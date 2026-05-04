#!/bin/bash
# Fine-tune minBERT on SST and CF-IMDB.
# Outputs (predictions, model checkpoints, training logs) are written to ./outputs/
mkdir -p outputs

# 1. SST — finetune all BERT params
PREF='sst'
python classifier.py \
    --use_gpu \
    --option finetune \
    --lr 1e-5 \
    --seed 1234 \
    --train "data/${PREF}-train.txt" \
    --dev "data/${PREF}-dev.txt" \
    --test "data/${PREF}-test.txt" \
    --dev_out "outputs/${PREF}-dev-output.txt" \
    --test_out "outputs/${PREF}-test-output.txt" \
    --filepath "outputs/${PREF}-model.pt" | tee outputs/${PREF}-train-log.txt

# 2. CF-IMDB — finetune all BERT params
PREF='cfimdb'
python classifier.py \
    --use_gpu \
    --option finetune \
    --lr 1e-5 \
    --seed 1234 \
    --train "data/${PREF}-train.txt" \
    --dev "data/${PREF}-dev.txt" \
    --test "data/${PREF}-test.txt" \
    --dev_out "outputs/${PREF}-dev-output.txt" \
    --test_out "outputs/${PREF}-test-output.txt" \
    --filepath "outputs/${PREF}-model.pt" | tee outputs/${PREF}-train-log.txt

# 3. Ensemble on SST (multi-seed, averaged predictions)
PREF='sst'
python classifier.py \
    --use_gpu \
    --option ensemble \
    --lr 1e-5 \
    --epochs 5 \
    --train "data/${PREF}-train.txt" \
    --dev "data/${PREF}-dev.txt" \
    --test "data/${PREF}-test.txt" \
    --dev_out "outputs/${PREF}-ensemble-dev-output.txt" \
    --test_out "outputs/${PREF}-ensemble-test-output.txt" \
    --filepath "outputs/${PREF}-ensemble-model.pt" | tee outputs/${PREF}-ensemble-train-log.txt

# 4. Ensemble on CF-IMDB
PREF='cfimdb'
python classifier.py \
    --use_gpu \
    --option ensemble \
    --lr 1e-5 \
    --epochs 5 \
    --train "data/${PREF}-train.txt" \
    --dev "data/${PREF}-dev.txt" \
    --test "data/${PREF}-test.txt" \
    --dev_out "outputs/${PREF}-ensemble-dev-output.txt" \
    --test_out "outputs/${PREF}-ensemble-test-output.txt" \
    --filepath "outputs/${PREF}-ensemble-model.pt" | tee outputs/${PREF}-ensemble-train-log.txt
