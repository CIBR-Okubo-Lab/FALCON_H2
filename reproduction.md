# FALCON_H2 Reproduction Guide

This document walks you through reproducing the experimental results from the FALCON_H2 paper from scratch.

---

## Table of Contents

- [FALCON\_H2 Reproduction Guide](#falcon_h2-reproduction-guide)
  - [Table of Contents](#table-of-contents)
  - [1. Environment Setup](#1-environment-setup)
    - [Dependencies](#dependencies)
    - [FALCON Challenge Baseline Code](#falcon-challenge-baseline-code)
  - [2. Download the Dataset](#2-download-the-dataset)
    - [FALCON H2 Dataset](#falcon-h2-dataset)
    - [Data Preprocessing](#data-preprocessing)
  - [3. Download Language Models](#3-download-language-models)
    - [3-gram Language Model](#3-gram-language-model)
    - [GPT-2 XL](#gpt-2-xl)
  - [4. Stage 1: Train the GRU Backbone](#4-stage-1-train-the-gru-backbone)
  - [5. Stage 2: Train the MAML Aligner](#5-stage-2-train-the-maml-aligner)
  - [6. Pooled-Training Aligner (Ablation Baseline)](#6-pooled-training-aligner-ablation-baseline)
  - [7. Conventional Baseline: Joint Training with Session-Specific Layers](#7-conventional-baseline-joint-training-with-session-specific-layers)
  - [8. Held-Out Session Rapid Adaptation](#8-held-out-session-rapid-adaptation)
  - [9. EvalAI Submission](#9-evalai-submission)
    - [9.1 Local Evaluation](#91-local-evaluation)
    - [9.2 Remote Evaluation (EvalAI)](#92-remote-evaluation-evalai)
    - [9.3 Key Files](#93-key-files)
  - [10. Using Existing Checkpoints](#10-using-existing-checkpoints)
    - [10.1 Checkpoint List](#101-checkpoint-list)
    - [10.2 Loading a Checkpoint](#102-loading-a-checkpoint)
    - [10.3 Performing Rapid Adaptation from a Checkpoint](#103-performing-rapid-adaptation-from-a-checkpoint)
  - [FAQ](#faq)
    - [Q: I'm running out of GPU memory during training](#q-im-running-out-of-gpu-memory-during-training)
    - [Q: What is the data format?](#q-what-is-the-data-format)
    - [Q: MAML training is very slow](#q-maml-training-is-very-slow)
    - [Q: How do I change the local validation split?](#q-how-do-i-change-the-local-validation-split)
  - [Licenses](#licenses)

---

## 1. Environment Setup

### Dependencies

The project is built on PyTorch + PyTorch Lightning, with Hydra for configuration management.

```bash
pip install torch pytorch-lightning hydra-core omegaconf wandb
```

If you need LM decoding (3-gram + GPT-2 XL), also install:

```bash
pip install transformers
```

The LM decoding pipeline follows the CORP [LanguageModelDecoder](https://github.com/cffan/CORP/tree/master/LanguageModelDecoder).

For a full list of dependencies, see `falcon_submit/maml_3gram/requirements_3gram.txt`.

> **⚠️ Important: PyTorch version**
>
> This project uses two different PyTorch versions for different parts of the pipeline:
>
> - **PyTorch 2.7.1** — Required for `maml_aligner/` training. This version provides `torch.nn.utils.stateless.functional_call`, which is used by the MAML inner-loop implementation. Install with:
>   ```bash
>   pip install torch==2.7.1
>   ```
> - **PyTorch 1.13** — Required for Stage 1 training, EvalAI submission, and LM decoding. The LM decoding environment (CORP `LanguageModelDecoder`) is built around this older version. Install with:
>   ```bash
>   pip install torch==1.13
>   ```
>
> It is recommended to use **separate conda/virtual environments** for the two stages to avoid conflicts. The `falcon_submit/maml_3gram/` Docker image uses PyTorch 1.13.

### FALCON Challenge Baseline Code

The EvalAI submission portion of this project depends on the official [FALCON challenge code package](https://github.com/snel-repo/falcon-challenge):

```bash
pip install falcon-challenge
```

Or clone from GitHub and install:

```bash
git clone https://github.com/snel-repo/falcon-challenge.git
cd falcon-challenge
pip install -e .
```

---

## 2. Download the Dataset

### FALCON H2 Dataset

Download from the [DANDI Archive](https://dandiarchive.org/dandiset/000950):

```bash
pip install dandi
dandi download https://dandiarchive.org/dandiset/000950/0.241029.1403
```

The dataset contains:
- **21 held-in sessions** (for training)
- **5 held-out sessions** (only 3 calibration sentences released per session; evaluation via EvalAI)

### Data Preprocessing

Run the provided notebook to convert the raw NWB files into the pickle format expected by the training scripts:

```bash
cd rnn/
jupyter notebook formatData_remote.ipynb
```

This notebook:
1. Loads NWB files from the DANDI dataset (`../data/000950/`)
2. Computes per-session mean and std for z-score normalization
3. Extracts trial-level neural features and character labels
4. Saves the result as a pickle file (`data_remote`) containing `train` and `val` splits

After running, set `datasetPath` in the config files to point to the generated pickle file.

**Key config paths** (update to your actual paths):
- `rnn/conf/config.yaml` → `datasetPath` — Stage 1 and conventional-model training data
- `maml_aligner/conf/config.yaml` → `datasetPath` — MAML training data
- `rnn/conf/align_config.yaml` → `datasetPath` — Pooled-training aligner data

---

## 3. Download Language Models

LM decoding requires two pretrained language models:

### 3-gram Language Model

Download from the [CORP GitHub repository](https://github.com/cffan/CORP/tree/master/LanguageModelDecoder). The CORP repo provides a link to the 3-gram language model weights (trained on OpenWebText2, released under CC0).

### GPT-2 XL

Download from HuggingFace:

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2-xl")
tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2-xl")
```

Or:

```bash
pip install huggingface_hub
huggingface-cli download openai-community/gpt2-xl
```

> **Note**: GPT-2 XL uses the Modified MIT License. We use it at inference time only — we do not train, fine-tune, or redistribute it.

---

## 4. Stage 1: Train the GRU Backbone

Train a session-agnostic GRU backbone (no session-specific layer).

> **Note:** `rnn/neural_decoder_trainer.py` is used for both the **Stage 1 session-agnostic backbone** and the **conventional model with session-specific layers**. The difference is controlled by the model import — `model_no_daylayer` for Stage 1, `model` (affine + softsign) or `model_linear` (affine only) for the conventional model.

```bash
cd rnn/

# Edit the config file to set datasetPath and outputDir
# vi conf/config.yaml

# Run training
python neural_decoder_trainer.py
```

**Training hyperparameters** (defaults; see Table 3 of paper):

| Parameter | Value |
|-----------|-------|
| Batch size | 48 |
| Steps | 20,000 |
| Optimizer | Adam |
| Learning rate | 0.02 |
| L2 weight decay | 1e-5 |
| Gradient clipping norm | 10 |

After training, model weights are saved under `out/run_YYYYMMDD-HHMMSS/modelWeights`.

---

## 5. Stage 2: Train the MAML Aligner

Meta-learn a lightweight input aligner on top of the frozen Stage 1 backbone. Single-layer (linear) vs two-layer (nonlinear) is controlled in `conf/config.yaml` via the `single_layer` flag.

```bash
cd maml_aligner/

# Edit config to set datasetPath, outputDir, pretrain, and aligner settings
# vi conf/config.yaml
#   pretrain: /path/to/run_20250908-101130/modelWeights
#   single_layer: True   # for linear aligner
#   single_layer: False  # for 2-layer aligner
#   align_hidden: 92     # hidden size for 2-layer aligner

# Run training
python neural_decoder_trainer.py
```

**MAML training hyperparameters** (defaults; see Table 4 of paper):

| Parameter | Single-layer | 2-layer |
|-----------|-------------|---------|
| Steps | 100,000 | 100,000 |
| Inner-loop LR | 0.005 | 0.07 |
| Outer-loop LR | 0.002 | 0.007 |
| Inner-loop steps | 3 | 3 |
| Tasks per batch | 3 | 3 |
| Support set size | 5 | 5 |
| Query set size | 5 | 5 |

> **Note**: MAML training does **not** use distributed training (`devices: 1`) because inner-loop updates are sequential.

---

## 6. Pooled-Training Aligner (Ablation Baseline)

Train the pooled-training aligner — this shares the same first half (backbone + aligner architecture) as the MAML aligner, but trains the aligner differently: all held-in sessions are pooled together and the aligner is trained directly, rather than treating each session as a separate MAML task.

```bash
cd rnn/

# Edit align_config.yaml to set datasetPath, outputDir, and pretrain
# vi conf/align_config.yaml
#   pretrain: /path/to/run_20250908-101130/modelWeights

# Run training
python joint_aligner_trainer.py

# The key difference from MAML:
#   - MAML (maml_aligner/neural_decoder_trainer.py): each held-in session is a separate task,
#     with support/query splits and inner/outer loop optimization
#   - Pooled training (rnn/joint_aligner_trainer.py): all sessions mixed together,
#     standard training with only the aligner trainable (backbone frozen)
```

**Training hyperparameters** (defaults; see Table 4 / Appendix F.1):

| Parameter | Value |
|-----------|-------|
| Batch size | 30 (matches total samples: 3 tasks × 5 support × 2) |
| Steps | 100,000 (same as MAML for fair comparison) |
| Optimizer | Adam |
| Outer learning rate | 0.007 (2-layer) |
| L2 weight decay | 1e-5 |
| Gradient clipping norm | 10 |

The resulting checkpoint (`run_20260402-125444`) corresponds to the "pooled-training initialization" baseline in Table 9 / Fig 3a of the paper (WER w/ LM: 9.16%).

---

## 7. Conventional Baseline: Joint Training with Session-Specific Layers

Train the conventional model that jointly learns session-specific input layers and a shared GRU decoder (Fig 2a of paper). This model has a per-session affine (or affine + softsign) input transform before the shared GRU, trained together with the backbone.

```bash
cd rnn/

# Edit config to set datasetPath and outputDir
# vi conf/config.yaml
# Change the model import from `model_no_daylayer` to `model` (affine + softsign)
# or `model_linear` (affine only)

# Run training
python neural_decoder_trainer.py
```

> **Note**: The conventional model uses `rnn/neural_decoder_trainer.py` (same script as Stage 1), but imports a different model class (`model.py` or `model_linear.py` instead of `model_no_daylayer.py`). Make sure to switch the import accordingly.

**Training hyperparameters** (defaults; see Table 5 of paper):

| Parameter | Value |
|-----------|-------|
| Batch size | 48 |
| Steps | 20,000 |
| Optimizer | Adam |
| Learning rate | 0.04 |
| L2 weight decay | 1e-5 |
| Gradient clipping norm | 10 |

After training, you can fine-tune the session-specific layer for held-out sessions using the 3 released calibration sentences (learning rate 0.1 and 20 update steps, as selected in Appendix C.3).

---

## 8. Held-Out Session Rapid Adaptation

Rapidly adapt the aligner for a new held-out session (only the aligner is updated; backbone stays frozen).

```bash
cd maml_aligner/

# Open fast_adaptation.ipynb
# Set the following:
#   - model_path: path to the trained MAML aligner checkpoint
#   - backbone_path: path to the Stage 1 backbone checkpoint
#   - calibration_sentences: 3 calibration sentences
#   - adaptation_steps: 25 (default)
#   - adaptation_lr: same as inner-loop LR

# Run the notebook cell by cell
```

**Rapid Adaptation Parameters**:

| Parameter | Value |
|-----------|-------|
| Adaptation steps | 25 |
| Learning rate | Same as inner-loop LR (linear: 0.005, 2-layer: 0.07) |
| Updated parameters | Aligner only (~0.15% of total) |
| Frozen parameters | GRU backbone |

> **Tip**: You can also perform rapid adaptation + evaluation directly in `evaluation.ipynb`.

---

## 9. EvalAI Submission

Use the code in `falcon_submit/` to submit results to the FALCON benchmark EvalAI platform.

### 9.1 Local Evaluation

```bash
cd falcon_submit/maml_3gram/

python maml_3gram_sample.py \
  --evaluation local \
  --model-path /path/to/adapted/aligner/ \
  --split h2 \
  --phase minival
```

### 9.2 Remote Evaluation (EvalAI)

```bash
# Build the Docker image
docker build -f maml_3gram_sample.Dockerfile -t maml_3gram_submit .

# Submit the image following EvalAI instructions
# See: https://snel-repo.github.io/falcon/
```

### 9.3 Key Files

- `maml_3gram_decoder.py` — Defines the `MAMLLMDecoder` class, containing the full neural decoding + LM decoding pipeline
- `utils/lmDecoderUtils.py` — LM decoding utilities (3-gram + GPT-2 XL beam search)
- `requirements_3gram.txt` — Python dependencies for the Docker image


---

## 10. Using Existing Checkpoints

If you don't want to train from scratch, download the pretrained weights from the [GitHub Release](https://github.com/CIBR-Okubo-Lab/FALCON_H2_two_stage/releases). They are not included in the git repository. Extract the archives into `model_checkpoints/`.

### 10.1 Checkpoint List

| Checkpoint | Purpose | WER w/ LM ↓ |
|-----------|---------|------------|
| `run_20260203-104333/` | Session-specific layer (affine + softsign) | 26.99% (ZS); 17.54% (FSS) |
| `run_20250908-101130/` | Stage 1 GRU Backbone | 12.10% |
| `run_20260320-104023/` | 2-layer MAML Aligner (main method) | **7.41%** |
| `run_20260402-125444/` | Pooled-training Aligner (ablation) | 9.16% |
| `run_20260417-124107/` | Linear MAML Aligner (ablation) | 10.80% |

### 10.2 Loading a Checkpoint

```python
# Stage 1 Backbone
backbone = GRUDecoder.load_from_checkpoint(
    "model_checkpoints/run_20250908-101130/modelWeights"
)

# MAML Aligner
aligner = MAMLAligner.load_from_checkpoint(
    "model_checkpoints/run_20260320-104023/last.ckpt",
    pretrain_model="model_checkpoints/run_20250908-101130/modelWeights"
)
```

### 10.3 Performing Rapid Adaptation from a Checkpoint

1. Open `maml_aligner/fast_adaptation.ipynb`
2. Set `model_path` to the corresponding checkpoint directory
3. Set `backbone_path` to `model_checkpoints/run_20250908-101130/modelWeights`
4. Run the notebook

---

## FAQ

### Q: I'm running out of GPU memory during training

- Stage 1 training uses up to 4× RTX 4090 (24 GB). If using a single GPU, try reducing `batchSize` in the config file.
- 7-layer and 9-layer GRU configurations may OOM on 24 GB. The 5-layer default is recommended.

### Q: What is the data format?

Data is preprocessed into a pickle file with keys such as `loadedData["train"]`, etc. Each session contains neural feature sequences (192 channels, 20 ms bins) and corresponding text labels. See `neural_decoder/dataset.py` for the `SpeechDataModule` and `MAMLDataModule` implementations.

### Q: MAML training is very slow

MAML training takes ~40 hours (single GPU) because inner-loop updates are sequential and difficult to parallelize. You can try:
- Reducing `nSteps` (may affect convergence)
- Reducing `inner_steps` (though the paper uses 3 steps)
- Using a GPU with more memory to support a larger batch

### Q: How do I change the local validation split?

The default local validation split uses sessions 1–18 as local held-in and sessions 19–21 as local held-out (Figure 4). This is set during data preprocessing; see the original data-processing code.

---

## Licenses

- **Code**: Based on FALCON challenge code package (MIT License)
- **Data**: FALCON H2 Dandiset 000950 (CC BY 4.0)
- **3-gram LM**: CC0
- **GPT-2 XL**: Modified MIT License
