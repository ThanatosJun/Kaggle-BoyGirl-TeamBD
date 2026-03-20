# Boy or Girl 2026 NEW - Kaggle Competition
End-to-end Machine Learning strategy and complete execution pipeline for the Kaggle TeamBD Competition context.

## 🚀 Completely Reproducible Pipeline

To guarantee exactly identical mathematical evaluations and output behaviors across machines, we use a fully locked-down Conda virtual environment alongside heavily seeded pipeline scripts.

### 1. Environment Deployment
Make sure you have Miniconda or Anaconda installed, then run the following command directly at the project root to reproduce the identical dependency matrix:

```bash
conda env create -f environment.yml
```

### 2. Enter the Working Environment
```bash
conda activate boygirl_env
```

### 3. Automatic End-to-End Execution
Run the monolithic pipeline to derive features, compute structural models, aggregate soft-voting logic, and format submissions:
```bash
python src/pipeline.py
```

### Execution Strategy Summary
- **Module A (Data Defense & Base Pipeline)**: Handles duplicate eradication, anomaly clipping, string stripping, and 5-fold Stratified partitioning.
- **Module B (Feature Engineering)**: Derives body stat norms, residual deviations, noise pruning, and embeds textual semantic extraction up to 1000 features.
- **Module C (Robust Model Training)**: Isolates standardization iteratively within individual folds preventing leakage, trains 5 randomized LightGBM configurations natively suppressing over-fit patterns.
- **Module D (Tactical Resolution)**: Synthesize `submission_final.csv` translating 1/0 binary boundaries back strictly into Kaggle required gender codes using 0-bias validation-optimized threshold grids.
