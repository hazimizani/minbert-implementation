# minBERT — implementation of the missing pieces

This repository is **my implementation** of the unfinished components of `minBERT`, a minimalist educational BERT scaffold. The original scaffold is *not* mine — see "Attribution" below.

The exercise: given the surrounding code (config, tokenizer, base class, training loop, sanity checks), implement the actual transformer math (multi-head self-attention, the BERT layer, the embedding stack, the AdamW optimizer, and the sentence classifier on top), then fine-tune the resulting model on SST and CF-IMDB.

## What I implemented

| File | What I wrote |
|------|--------------|
| `bert.py` | `BertSelfAttention.attention`, `BertLayer.add_norm`, `BertLayer.forward`, `BertModel.embed` |
| `optimizer.py` | `AdamW.step` — decoupled weight decay, "efficient" bias correction (Kingma & Ba 2014), no LR schedule |
| `classifier.py` | `BertSentClassifier` — pooler → dropout → linear projection; pretrain vs. finetune handling; ensemble option |

The remaining files (`base_bert.py`, `config.py`, `tokenizer.py`, `utils.py`, `sanity_check.py`) are part of the scaffold and were not modified.

## How the pieces fit

```
input ids
  ↓ BertModel.embed   ← word embeddings + positional embeddings + LayerNorm + dropout
  ↓ BertLayer × N     ← each: self-attention → add_norm → feed-forward → add_norm
  ↓ pooler ([CLS] → tanh-projected)
  ↓ BertSentClassifier  ← dropout → linear → class logits
  ↓ AdamW with decoupled weight decay
```

The two add-norm blocks per layer are what people usually mean when they say "transformers are residual networks." Skipping the residual connections breaks gradient flow on deep stacks; that's why the sanity check pins layer-by-layer outputs against a reference.

## Setup

```bash
bash setup.sh
```

This installs PyTorch and downloads the pre-trained BERT base weights that `BertModel.from_pretrained` consumes (parameter names get re-mapped from HuggingFace's naming in `base_bert.py`).

### Data

The SST and CF-IMDB datasets are not bundled. Each line is `tag ||| sentence`. Place under `data/`:

```
data/sst-train.txt    data/sst-dev.txt    data/sst-test.txt
data/cfimdb-train.txt data/cfimdb-dev.txt data/cfimdb-test.txt
```

## Verify the implementation

```bash
python sanity_check.py     # checks BERT layer outputs against reference embeddings
python optimizer_test.py   # checks AdamW step against reference trajectory
```

## Train

```bash
bash run_exp.sh
```

Runs four experiments: SST + CF-IMDB, each with single-seed fine-tuning and a multi-seed ensemble. Outputs land in `outputs/`.

Reference accuracies (from the original assignment):

| Setting | Dev | Test |
|---|---|---|
| SST pretrain (frozen BERT)      | 0.391 ± 0.007 | 0.403 ± 0.008 |
| SST finetune                    | 0.515 ± 0.004 | 0.526 ± 0.008 |
| CF-IMDB finetune                | 0.966 ± 0.007 | — |

## Attribution

The scaffold (everything outside the three files above, and the structure within them) is from CMU's **CS11-747: Neural Networks for NLP**, written by **Shuyan Zhou, Zhengbao Jiang, Ritam Dutt, and Brendon Boldt**. Several files are derived from HuggingFace's [`transformers`](https://github.com/huggingface/transformers) library and are covered under the Apache License 2.0 (see `LICENSE`).

This repo exists to show *my work on the missing pieces*; the surrounding educational scaffold belongs to the authors above.

## References

- Vaswani et al., *Attention Is All You Need*, 2017 — https://arxiv.org/abs/1706.03762
- Devlin et al., *BERT*, 2018 — https://arxiv.org/abs/1810.04805
- Loshchilov & Hutter, *Decoupled Weight Decay Regularization*, 2017 — https://arxiv.org/abs/1711.05101
- Kingma & Ba, *Adam*, 2014 — https://arxiv.org/abs/1412.6980
- `structure.md` — design notes from the original assignment authors
