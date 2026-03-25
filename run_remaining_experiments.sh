#!/bin/bash
cd /home/thanatos/data_science/Kaggle-BoyGirl-TeamBD
python main_train.py --config configs/exp1_p99_method2_paper_range.yaml
python main_train.py --config configs/exp1_p99_method3_grouped_mean.yaml
