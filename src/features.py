import copy
import importlib
import re
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from .imputation_strategies import get_imputer_from_config


def _load_transformer_backends():
    """Lazy import heavy text backends to avoid crashing non-text runs at import time."""
    transformers_mod = importlib.import_module('transformers')
    torch_mod = importlib.import_module('torch')
    return transformers_mod.AutoTokenizer, transformers_mod.AutoModel, torch_mod


def engineer_features(X: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Compute derived features based on config flags.
    Must be called AFTER imputation (so weight/height are non-NaN).

    Config keys (under features):
      add_weight_height_ratio: bool  → adds 'weight_height_ratio' = weight / height
      add_bmi: bool                  → adds 'bmi' = weight / (height/100)^2
      add_ponderal_index: bool       → adds 'ponderal_index' = weight / (height/100)^3
    """
    feat_cfg = config.get('features', {})
    X = X.copy()

    if feat_cfg.get('add_weight_height_ratio', False):
        height_safe = X['height'].replace(0, np.nan).fillna(1e-6)
        X['weight_height_ratio'] = X['weight'] / height_safe

    if feat_cfg.get('add_bmi', False):
        height_m = (X['height'] / 100.0).replace(0, np.nan).fillna(1e-6)
        X['bmi'] = X['weight'] / (height_m ** 2)

    if feat_cfg.get('add_ponderal_index', False):
        height_m = (X['height'] / 100.0).replace(0, np.nan).fillna(1e-6)
        X['ponderal_index'] = X['weight'] / (height_m ** 3)

    return X


def fit_pre_imputation_clip_bounds(
    X: pd.DataFrame,
    clip_cols,
    lower_percentile=1,
    upper_percentile=99,
):
    """Fit clipping bounds on train fold before imputation."""
    bounds = {}
    for col in clip_cols:
        if col not in X.columns:
            continue
        values = pd.to_numeric(X[col], errors='coerce').to_numpy(dtype=float)
        if np.all(np.isnan(values)):
            continue
        lower = float(np.nanpercentile(values, lower_percentile))
        upper = float(np.nanpercentile(values, upper_percentile))
        bounds[col] = (lower, upper)
    return bounds


def apply_pre_imputation_clip_bounds(X: pd.DataFrame, bounds: dict) -> pd.DataFrame:
    """Apply pre-imputation clipping bounds to dataframe columns."""
    X_out = X.copy()
    for col, (lower, upper) in bounds.items():
        if col not in X_out.columns:
            continue
        series = pd.to_numeric(X_out[col], errors='coerce')
        X_out[col] = series.clip(lower=lower, upper=upper)
    return X_out


class ClippingTransformer(BaseEstimator, TransformerMixin):
    """自訂 Transformer：針對數值型特徵進行極端值剪裁 (Clipping)"""
    def __init__(self, lower_percentile=1, upper_percentile=99):
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.lower_bounds_ = None
        self.upper_bounds_ = None

    def fit(self, X, y=None):
        self.lower_bounds_ = np.nanpercentile(X, self.lower_percentile, axis=0)
        self.upper_bounds_ = np.nanpercentile(X, self.upper_percentile, axis=0)
        return self

    def transform(self, X, y=None):
        # 建立副本以避免覆寫原始資料
        X_clipped = np.copy(X)
        for i in range(X.shape[1]):
            X_clipped[:, i] = np.clip(X_clipped[:, i], self.lower_bounds_[i], self.upper_bounds_[i])
        return X_clipped

    def get_feature_names_out(self, input_features=None):
        """Keep one-to-one feature names after clipping."""
        if input_features is None:
            return np.array([], dtype=object)
        return np.asarray(input_features, dtype=object)

class TextEmbeddingTransformer(BaseEstimator, TransformerMixin):
    """自訂 Transformer：將文本特徵轉換為數值向量（支援 MiniLM、TF-IDF、或兩者併用）"""
    _MINILM_CACHE = {}

    def __init__(self, Mini_LM=True, TF_IDF=False, use_both=False,
                 embedding_dim=384, tfidf_max_features=50,
                 use_pca=False, pca_n_components=30,
                 model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
                 cache_dir=None,
                 model_revision=None):
        self.Mini_LM = Mini_LM
        self.TF_IDF = TF_IDF
        self.use_both = use_both
        self.embedding_dim = embedding_dim
        self.tfidf_max_features = tfidf_max_features
        self.use_pca = use_pca
        self.pca_n_components = pca_n_components
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.model_revision = model_revision
        self.tokenizer_ = None
        self.model_ = None
        self.tfidf_vectorizer_ = None
        self.pca_ = None
        self.pca_scaler_ = None
        self.torch_ = None

    # ── Pickle 優化：不序列化 ~460MB 的模型權重 ──
    def __getstate__(self):
        state = self.__dict__.copy()
        state['tokenizer_'] = None
        state['model_'] = None
        state['torch_'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # 延遲到真正需要編碼時再載入模型，避免反序列化階段大量 IO。
        self.tokenizer_ = None
        self.model_ = None
        self.torch_ = None

    # ── Deepcopy 優化：跨 fold 共享同一份模型引用（推論模式安全）──
    def __deepcopy__(self, memo):
        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new
        for k, v in self.__dict__.items():
            if k in ('tokenizer_', 'model_', 'torch_'):
                setattr(new, k, v)  # 共享引用，不複製
            else:
                setattr(new, k, copy.deepcopy(v, memo))
        return new

    def _to_texts(self, X):
        """將各種輸入格式統一轉為文本列表"""
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, 0].fillna('').astype(str).tolist()
        if isinstance(X, pd.Series):
            return X.fillna('').astype(str).tolist()
        return [str(x) if x is not None else '' for x in X]

    def _fit_tfidf(self, texts):
        """訓練 TF-IDF vectorizer"""
        self.tfidf_vectorizer_ = TfidfVectorizer(
            max_features=self.tfidf_max_features,
            ngram_range=(1, 1),
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,
            token_pattern=r'(?u)\b\w+\b'
        )
        self.tfidf_vectorizer_.fit(texts)

    def _fit_minilm(self):
        """加載 MiniLM 模型（僅首次）"""
        if self.model_ is not None and self.tokenizer_ is not None and self.torch_ is not None:
            return

        cache_key = (
            self.model_name,
            self.cache_dir,
            self.model_revision,
        )
        cached = self._MINILM_CACHE.get(cache_key)
        if cached is not None:
            self.tokenizer_, self.model_, self.torch_ = cached
            return

        if self.model_ is None:
            AutoTokenizer, AutoModel, torch_mod = _load_transformer_backends()
            self.torch_ = torch_mod
            self.tokenizer_ = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                revision=self.model_revision,
            )
            self.model_ = AutoModel.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                revision=self.model_revision,
            )

            self.model_.eval()
            self._MINILM_CACHE[cache_key] = (self.tokenizer_, self.model_, self.torch_)

    def _encode_minilm(self, texts):
        """使用 MiniLM 將文本轉為嵌入向量"""
        encoded_input = self.tokenizer_(
            texts, padding=True, truncation=True,
            return_tensors='pt', max_length=128)
        with self.torch_.no_grad():
            model_output = self.model_(**encoded_input)
        return self.mean_pooling(
            model_output, encoded_input['attention_mask']).cpu().numpy()

    def fit(self, X, y=None):
        texts = self._to_texts(X)

        if self.use_both:
            self._fit_tfidf(texts)
            self._fit_minilm()
        elif self.TF_IDF:
            self._fit_tfidf(texts)
        elif self.Mini_LM:
            self._fit_minilm()

        # 如果啟用 PCA，在 fit 階段先產生嵌入向量再訓練 PCA
        if self.use_pca and (self.Mini_LM or self.use_both):
            raw = self._get_raw_embeddings(texts)
            self.pca_scaler_ = StandardScaler()
            scaled = self.pca_scaler_.fit_transform(raw)
            self.pca_ = PCA(n_components=self.pca_n_components)
            self.pca_.fit(scaled)

        return self

    def _get_raw_embeddings(self, texts):
        """產生原始嵌入向量（用於 PCA fit）"""
        if self.use_both:
            tfidf_feat = self.tfidf_vectorizer_.transform(texts).toarray()
            minilm_feat = self._encode_minilm(texts)
            return np.hstack([tfidf_feat, minilm_feat])
        return self._encode_minilm(texts)

    def mean_pooling(self, model_output, attention_mask):
        """對所有 token embeddings 進行平均池化"""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return self.torch_.sum(token_embeddings * input_mask_expanded, 1) / self.torch_.clamp(input_mask_expanded.sum(1), min=1e-9)

    def transform(self, X, y=None):
        """將文本轉換為嵌入向量（返回 numpy array）"""
        texts = self._to_texts(X)

        if self.use_both:
            tfidf_feat = self.tfidf_vectorizer_.transform(texts).toarray()
            minilm_feat = self._encode_minilm(texts)
            result = np.hstack([tfidf_feat, minilm_feat])
            if self.use_pca and self.pca_ is not None:
                result = self.pca_.transform(self.pca_scaler_.transform(result))
            return result

        if self.TF_IDF:
            return self.tfidf_vectorizer_.transform(texts).toarray()

        if self.Mini_LM:
            result = self._encode_minilm(texts)
            if self.use_pca and self.pca_ is not None:
                result = self.pca_.transform(self.pca_scaler_.transform(result))
            return result

        raise ValueError("必須啟用 Mini_LM、TF_IDF 或 use_both")

    def apply_PCA(self, embeddings, n_components=None):
        """對嵌入向量應用 PCA 降維"""
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)
        pca = PCA(n_components=n_components)
        embeddings_reduced = pca.fit_transform(embeddings_scaled)
        return embeddings_reduced, pca, scaler


class TextHandcraftedTransformer(BaseEstimator, TransformerMixin):
    """從 self_intro 文本萃取低維度手工特徵的 Transformer

    產出特徵 (15 維)：
      - text_len: 文字總長度
      - word_count: 詞數
      - avg_word_len: 平均每個詞的長度
      - is_empty: 是否為空文本
      - chinese_ratio: 中文字元佔比
      - upper_ratio: 大寫字母佔比
      - punctuation_count: 標點符號數量
      - has_self_ref: 是否包含第一人稱 (I/i'm/i am)
      - has_male_keyword: 是否包含男性傾向關鍵詞 (handsome/man/boy/cool)
      - has_female_keyword: 是否包含女性傾向關鍵詞 (beautiful/girl/brilliant)
      - sentiment_positive: 正向形容詞數量
      - ends_with_period: 是否以句號結尾
      - is_all_lower: 全小寫文本 (p=0.006, 男性傾向)
      - has_number: 含數字 (p=0.002, 100%男性)
      - is_short_simple: ≤8字元且無空格 (p=0.015, 男性傾向)
    """

    # ── 編譯一次即可，所有實例共享 ──
    _RE_CHINESE = re.compile(r'[\u4e00-\u9fff]')
    _RE_MALE_KW = re.compile(
        r'\b(?:handsome|man|boy|cool|nerd|hard|hungry|super|foolish)\b', re.I)
    _RE_FEMALE_KW = re.compile(
        r'\b(?:beautiful|girl|brilliant|cute|introvert|never)\b', re.I)
    _RE_SELF_REF = re.compile(r"\bi\b|\bi'm\b|\bi am\b", re.I)
    _RE_POSITIVE = re.compile(
        r'\b(?:happy|good|nice|smart|positive|love|great|wonderful|amazing)\b', re.I)
    _RE_HAS_NUMBER = re.compile(r'\d')
    _PUNCT = set('.,!?;:~-')

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        texts = self._to_texts(X)
        features = np.array([self._extract(t) for t in texts], dtype=np.float64)
        return features

    def _to_texts(self, X):
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, 0].fillna('').astype(str).tolist()
        if isinstance(X, pd.Series):
            return X.fillna('').astype(str).tolist()
        return [str(x) if x is not None else '' for x in X]

    @classmethod
    def _extract(cls, text: str) -> list:
        """從單筆文本萃取手工特徵，回傳 list[float]"""
        length = len(text)
        words = text.split()
        word_count = len(words)

        return [
            # 基本文本統計
            float(length),
            float(word_count),
            float(np.mean([len(w) for w in words])) if words else 0.0,
            float(length == 0),

            # 字元組成
            len(cls._RE_CHINESE.findall(text)) / max(length, 1),
            sum(1 for c in text if c.isupper()) / max(length, 1),
            float(sum(1 for c in text if c in cls._PUNCT)),

            # 語意 / 關鍵詞
            float(bool(cls._RE_SELF_REF.search(text))),
            float(bool(cls._RE_MALE_KW.search(text))),
            float(bool(cls._RE_FEMALE_KW.search(text))),
            float(len(cls._RE_POSITIVE.findall(text))),

            # 風格
            float(text.rstrip().endswith('.')),

            # 新增統計顯著特徵
            float(text == text.lower() and length > 0),   # is_all_lower
            float(bool(cls._RE_HAS_NUMBER.search(text))),  # has_number
            float(length <= 8 and ' ' not in text and length > 0),  # is_short_simple
        ]

    def get_feature_names_out(self, input_features=None):
        return [
            'text_len', 'word_count', 'avg_word_len', 'is_empty',
            'chinese_ratio', 'upper_ratio', 'punctuation_count',
            'has_self_ref', 'has_male_keyword', 'has_female_keyword',
            'sentiment_positive', 'ends_with_period',
            'is_all_lower', 'has_number', 'is_short_simple',
        ]


def clip_min_value(X, min_value=0):
    """
    將值 clip 到下界（用於 log1p 前處理）
    這是一個普通函數（非 lambda），可以被 pickle 序列化
    """
    return np.clip(X, min_value, None)


def get_scaler_from_config(scaler_name):
    if scaler_name == 'standard':
        return StandardScaler()
    if scaler_name == 'minmax':
        return MinMaxScaler()
    if scaler_name == 'robust':
        return RobustScaler()
    if scaler_name == 'none':
        return 'passthrough'
    raise ValueError(f"不支援的 scaler 設定: {scaler_name}")


def build_preprocessor(config):
    """
    根據 config 設定，建立完整的 Pipeline：
    1. 數值特徵 (一般): 中位數補值 -> 剪裁 (1%-99%) -> StandardScaler
    2. 數值特徵 (長尾): 中位數補值 -> 移除負值 (clip to 0) -> log(1+x) -> StandardScaler
       ⚠️ 重要: 數據中 fb_friends 存在 -1000 等負值，必須在 log1p 前處理，否則會產生 NaN
    3. 無序類別特徵: 可切換新版/舊版補值 -> One-Hot Encoding
    4. 有序類別特徵: 可切換新版/舊版補值 -> 維持原數值大小 (不做 One-Hot)
    """

    prep_cfg = config.get('preprocessing', {})
    imputation_mode = str(prep_cfg.get('imputation_mode', 'new')).lower()

    if imputation_mode in {'new', 'v2'}:
        default_cat_strategy = 'constant'
        default_cat_fill = '-1'
        default_ord_strategy = 'constant'
        default_ord_fill = -1
    elif imputation_mode in {'old', 'legacy', 'v1'}:
        default_cat_strategy = 'most_frequent'
        default_cat_fill = None
        default_ord_strategy = 'most_frequent'
        default_ord_fill = None
    else:
        raise ValueError("preprocessing.imputation_mode 僅支援: new | old")

    numeric_imputer_strategy = prep_cfg.get('numeric_imputer_strategy', 'median')
    # 允許手動覆寫策略；未設定時由 imputation_mode 決定
    categorical_imputer_strategy = prep_cfg.get('categorical_imputer_strategy', default_cat_strategy)
    categorical_fill_value = prep_cfg.get('categorical_fill_value', default_cat_fill)
    ordinal_imputer_strategy = prep_cfg.get('ordinal_imputer_strategy', default_ord_strategy)
    ordinal_fill_value = prep_cfg.get('ordinal_fill_value', default_ord_fill)
    clipping_lower = prep_cfg.get('clipping_lower_percentile', 1)
    clipping_upper = prep_cfg.get('clipping_upper_percentile', 99)
    log_clip_min = prep_cfg.get('log_clip_min', 0)
    scaler_name = prep_cfg.get('scaler', 'standard')
    onehot_handle_unknown = prep_cfg.get('onehot_handle_unknown', 'ignore')
    onehot_sparse_output = prep_cfg.get('onehot_sparse_output', False)

    # 1. 數值特徵 Pipeline (Clipping + Scaling)
    num_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=numeric_imputer_strategy)),
        ('clipper', ClippingTransformer(lower_percentile=clipping_lower, upper_percentile=clipping_upper)),
        ('scaler', get_scaler_from_config(scaler_name))
    ])

    # 2. 數值特徵長尾分佈 Pipeline (Log1p + Scaling)
    # ⚠️ 注意：必須先移除負值，否則 log1p 會產生 NaN 導致 SMOTE 失敗
    # Pipeline: 中位數補值 → clip 負值到 0 → log(1+x) → StandardScaler
    log_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=numeric_imputer_strategy)),
        ('clip_min', FunctionTransformer(
            clip_min_value,
            kw_args={'min_value': log_clip_min},
            validate=False,
            feature_names_out='one-to-one'
        )),
        ('log1p', FunctionTransformer(
            np.log1p,
            validate=False,
            feature_names_out='one-to-one'
        )),
        ('scaler', get_scaler_from_config(scaler_name))
    ])

    # 3. 無序類別特徵 Pipeline (OneHot Encoding)
    cat_imputer_kwargs = {'strategy': categorical_imputer_strategy}
    if categorical_imputer_strategy == 'constant':
        cat_imputer_kwargs['fill_value'] = categorical_fill_value

    cat_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(**cat_imputer_kwargs)),
        ('onehot', OneHotEncoder(handle_unknown=onehot_handle_unknown, sparse_output=onehot_sparse_output))
    ])

    # 4. 有序類別特徵 Pipeline (只做補值，因在 data_loader 已轉成 Float)
    ord_imputer_kwargs = {'strategy': ordinal_imputer_strategy}
    if ordinal_imputer_strategy == 'constant':
        ord_imputer_kwargs['fill_value'] = ordinal_fill_value

    ord_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(**ord_imputer_kwargs))
    ])

    # 5. 文本特徵 Pipeline（如果配置中有 text_cols）
    feat_cfg = config.get('features', {})
    text_cols = feat_cfg.get('text_cols', [])
    
    transformers_list = [
        ('num', num_pipeline, config['features']['numeric_cols']),
        ('log', log_pipeline, config['features']['numeric_log_cols']),
        ('cat', cat_pipeline, config['features']['categorical_cols']),
        ('ord', ord_pipeline, config['features']['ordinal_cols'])
    ]
    
    # ✅ 如果配置中有文本特徵，添加文本 pipeline
    if text_cols:
        text_method = str(feat_cfg.get('text_embedding_method', 'minilm')).lower()
        tfidf_max_features = feat_cfg.get('tfidf_max_features', 50)
        text_model_name = feat_cfg.get('text_model_name', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        text_model_cache_dir = feat_cfg.get('text_model_cache_dir', None)
        text_model_revision = feat_cfg.get('text_model_revision', None)
        model_source = text_model_name

        use_pca = bool(feat_cfg.get('use_pca', False))
        pca_n_components = int(feat_cfg.get('pca_n_components', 30))

        # text_embedding_method = 'handcrafted' 時只用手工特徵，不走嵌入
        if text_method != 'handcrafted':
            if text_method in {'both', 'combined', 'hybrid'}:
                embedder = TextEmbeddingTransformer(
                    Mini_LM=False, TF_IDF=False, use_both=True,
                    tfidf_max_features=tfidf_max_features,
                    use_pca=use_pca, pca_n_components=pca_n_components,
                    model_name=model_source,
                    cache_dir=text_model_cache_dir,
                    model_revision=text_model_revision)
            elif text_method in {'tfidf', 'tf-idf', 'tf_idf'}:
                embedder = TextEmbeddingTransformer(
                    Mini_LM=False, TF_IDF=True,
                    tfidf_max_features=tfidf_max_features)
            else:
                embedder = TextEmbeddingTransformer(
                    Mini_LM=True, TF_IDF=False,
                    use_pca=use_pca, pca_n_components=pca_n_components,
                    model_name=model_source,
                    cache_dir=text_model_cache_dir,
                    model_revision=text_model_revision)

            text_pipeline = Pipeline(steps=[('text_embedder', embedder)])
            transformers_list.append(('text', text_pipeline, text_cols))

    # ✅ 手工文本特徵 Pipeline
    #   啟用方式: use_text_handcrafted: true  或  text_embedding_method: "handcrafted"
    use_handcrafted = feat_cfg.get('use_text_handcrafted', False)
    if text_cols and (use_handcrafted or text_method == 'handcrafted'):
        handcrafted_pipeline = Pipeline(steps=[
            ('text_handcrafted', TextHandcraftedTransformer())
        ])
        transformers_list.append(('text_hc', handcrafted_pipeline, text_cols))

    # 組合所有 Pipeline 到 ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=transformers_list,
        remainder='drop'  # 沒有設定到的特徵直接丟棄
    )

    return preprocessor
