# minBERT — Additional Implementations and Experiments

Beyond the base BERT implementation and standard fine-tuning pipeline, I implemented four additional techniques aimed at improving sentence classification accuracy:

- **(A) Multi-Seed Ensemble Prediction**
- **(B) Enhanced Classification Head with Multi-Pool Aggregation**
- **(C) Learning Rate Warmup with Linear Decay Schedule**
- **(D) Label Smoothing**

These are accessed via `--option ensemble` and `--option improved_finetune` in `classifier.py`.

## Implemented techniques

### (A) Multi-Seed Ensemble Prediction

The most effective technique. Instead of relying on a single model trained with one random seed, I train 3 independent models with seeds `{1234, 11711, 42}` and average their log-probability outputs at inference time. Each model sees the same data but learns slightly different decision boundaries due to random initialization and shuffling, so the ensemble reduces variance and corrects individual model errors.

Implementation: `train_ensemble()` and `ensemble_eval()` in `classifier.py`. Run via `--option ensemble`.

Reference: Dietterich (2000), *Ensemble Methods in Machine Learning*, MCS.

### (B) Enhanced Classification Head (`BertSentClassifierImproved`)

The baseline classifier uses only the `[CLS]` pooler output. The improved classifier concatenates three representations:

- `[CLS]` pooler output (contextualized sentence embedding via tanh projection)
- Mean pooling over all token hidden states (mask-aware, ignoring padding)
- Max pooling over all token hidden states (mask-aware, padding set to `-1e9`)

This produces a `3 × hidden_size` (2304-dim) feature vector passed through dropout and a linear classification layer. Mean pooling provides an average semantic representation while max pooling highlights the most salient features.

Reference: Reimers & Gurevych (2019), *Sentence-BERT*, EMNLP.

### (C) Learning Rate Warmup + Linear Decay

Linear warmup over the first 10% of total training steps, followed by linear decay to zero. Prevents early instability from large gradient updates on randomly initialized layers.

Reference: Devlin et al. (2019), *BERT*, NAACL.

### (D) Label Smoothing

Distributes a small fraction (`epsilon=0.1`) of probability uniformly across all classes instead of concentrating on the ground-truth label, acting as a regularizer:

```
loss = (1 - epsilon) * NLL(target) + epsilon * uniform_loss / num_classes
```

Reference: Szegedy et al. (2016), *Rethinking the Inception Architecture*, CVPR.

## Experimental setup

All experiments use `batch_size=8`, `hidden_dropout_prob=0.3`.

| Setting | Command |
|---|---|
| Baseline finetune | `--option finetune --lr 1e-5 --seed 1234 --epochs 10` |
| Improved finetune | `--option improved_finetune --lr 2e-5 --seed 1234 --epochs 10` |
| Ensemble | `--option ensemble --lr 1e-5 --epochs 5` (seeds 1234, 11711, 42) |

## Results

### SST sentiment classification (5-class)

| Model              | Dev Acc   | Test Acc  |
|--------------------|-----------|-----------|
| Reference          | 0.515     | 0.526     |
| Baseline finetune  | 0.523     | 0.545     |
| Improved finetune  | 0.522     | 0.531     |
| **Ensemble (3 seeds)** | **0.531** | 0.538     |

### CFIMDB binary classification

| Model              | Dev Acc   | Test Acc  |
|--------------------|-----------|-----------|
| Reference          | 0.966     | —         |
| Baseline finetune  | 0.963     | 0.512     |
| Improved finetune  | 0.955     | 0.527     |
| **Ensemble (3 seeds)** | **0.967** | 0.508     |

The ensemble achieves the best dev accuracy on both datasets:
- **SST:** +0.8% over baseline (0.531 vs 0.523)
- **CFIMDB:** +0.4% over baseline (0.967 vs 0.963)

## Analysis

Ensemble prediction is the most effective technique for this setting. Individual models trained with different seeds show variance in their best dev accuracy (SST: 0.523, 0.511, 0.530 for seeds 1234, 11711, 42 respectively). By averaging predictions, the ensemble smooths out individual model errors and consistently outperforms any single model.

The improved classification head (multi-pool) with warmup and label smoothing did **not** outperform the baseline. Analysis suggests this is because:

1. The 3× wider head has more parameters requiring more data to converge
2. Label smoothing overlaps with the already-aggressive dropout (0.3)
3. Warmup delays early learning, and SST's best checkpoints tend to be early

These techniques would likely show more benefit with larger datasets or longer training. On SST (8544 examples) and CFIMDB (1707 examples), the standard `[CLS]` classifier is already a strong baseline.

**Key takeaway:** For small fine-tuning datasets, multi-seed ensembling is the most reliable path to accuracy improvement, as it addresses model variance rather than model capacity.

## Training logs

Full per-epoch training logs are included in this directory:

- `sst-train-log.txt`, `cfimdb-train-log.txt` — baseline
- `sst-improved-train-log.txt`, `cfimdb-improved-train-log.txt` — improved finetune
- `sst-ensemble-train-log.txt`, `cfimdb-ensemble-train-log.txt` — ensemble
