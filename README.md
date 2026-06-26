# BrainFuse-LLM

Official repository for **BrainFuse-LLM: Subject-Agnostic fMRI-to-Text Decoding with Neuro-Enhanced Encoding and Expert Modulation**.

BrainFuse-LLM is a subject-agnostic brain-to-language framework for decoding voxel-level fMRI signals into natural-language outputs. The model supports both brain captioning and instruction-conditioned fMRI question answering by combining a topology-aware neural encoder with parameter-efficient large-language-model adaptation.

## Highlights

- Subject-agnostic fMRI-to-text decoding for captioning and question answering.
- Neuro-Enhanced Encoder for fusing voxel responses, spatial coordinates, and neuroanatomical priors.
- Neuro LoRA-MoE for neural- and text-conditioned expert modulation of frozen LLMs.
- Evaluation on NSD captioning, fMRI-QA, unseen-subject transfer, and selected reasoning tasks.

## Method Overview

BrainFuse-LLM contains two core components:

1. **Neuro-Enhanced Encoder**: aggregates variable-length voxel-level fMRI responses into fixed-length neural tokens using multi-scale value fusion, coordinate-aware key construction, region-level priors, and attention-based aggregation.
2. **Neuro LoRA-MoE**: inserts lightweight LoRA experts into selected feed-forward layers of a frozen language model and routes tokens to experts according to neural-language hidden states.

The projected fMRI tokens are concatenated with optional instruction tokens and decoded by a frozen Vicuna-style language backbone equipped with trainable adapters.

## Installation

```bash
conda create -n brainfuse python=3.10 -y
conda activate brainfuse
pip install -r requirements.txt
pip install -e .
```

For full-scale training with large language models, install the CUDA-compatible PyTorch build and optional acceleration libraries for your hardware.

## Dataset

Experiments in the paper are based on the Natural Scenes Dataset (NSD). The repository does not redistribute raw NSD fMRI data or third-party VQA annotations. Please obtain them from the original providers and place processed files under `data/` following:

```text
data/
  nsd/
    subject01/
    subject02/
    ...
  annotations/
    coco_captions/
    vqa/
    tdiuc/
```

See [docs/dataset.md](docs/dataset.md) for expected fields and preprocessing notes.

## Quick Synthetic Check

The repository includes a lightweight synthetic forward-pass check for the proposed encoder and adapter modules:

```bash
python scripts/run_synthetic_demo.py
```

This does not reproduce paper numbers; it verifies that the core BrainFuse-LLM modules can process variable-size voxel inputs and produce fixed-length neural tokens.

## Usage

### Preprocess NSD

```bash
python scripts/prepare_nsd_metadata.py \
  --nsd-root /path/to/nsd \
  --out data/processed/nsd_metadata.jsonl
```

### Train BrainFuse-LLM

```bash
python scripts/train_brainfuse.py \
  --config configs/brainfuse_vicuna7b.yaml \
  --data-root data/processed \
  --output-dir outputs/brainfuse
```

### Evaluate Captioning and fMRI-QA

```bash
python scripts/evaluate_brainfuse.py \
  --checkpoint outputs/brainfuse/checkpoint_last.pt \
  --task captioning

python scripts/evaluate_brainfuse.py \
  --checkpoint outputs/brainfuse/checkpoint_last.pt \
  --task a-okvqa
```

The training and evaluation entrypoints are intentionally documented as reproducibility interfaces. Full dataset preprocessing and checkpoint release details will be updated with the camera-ready version.

## Paper Results

BrainFuse-LLM is evaluated on NSD brain captioning, fMRI question answering, unseen-subject generalization, and selected TDIUC reasoning tasks. Representative results include:

- NSD brain captioning: CIDEr 62.30.
- A-OKVQA fMRI-QA: accuracy 55.37.
- Leave-one-subject-out evaluation: consistent gains over subject-agnostic baselines.

See [docs/reproducibility.md](docs/reproducibility.md) and [docs/baselines.md](docs/baselines.md).

## Related Work and Baselines

The repository structure and documentation style are inspired by recent public brain-decoding repositories such as MindLLM, BrainChat, UniBrain, UMBRAE, and MindBridge. We only reuse public documentation conventions and do not copy implementation code from these projects.

## Citation

```bibtex
@article{shen2026brainfuse,
  title={BrainFuse-LLM: Subject-Agnostic fMRI-to-Text Decoding with Neuro-Enhanced Encoding and Expert Modulation},
  author={Shen, Zehao and Jiang, Maowei and Qi, Shuo and Zeng, Weiming},
  journal={Information Fusion},
  year={2026},
  note={Under review}
}
```

## License

This repository is released under the MIT License. Dataset use is governed by the licenses and terms of the original data providers.
