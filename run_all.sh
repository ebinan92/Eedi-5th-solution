#!/bin/bash
# generate synthetic data
python src/generate_question.py

# split data into 5 folds
python src/split_fold.py

# knowledge distillation
python src/distill.py

# training biencoder model
python src/train_biencoder.py --fold 4
python src/train_biencoder.py --fold 3
python src/train_biencoder.py --fold 2
python src/train_biencoder.py --fold 1
python src/train_biencoder.py --fold 0

# inference biencoder model and get topk ids
python src/inference_biencoder.py

# training listwise model
python src/train_listwise.py --fold 4