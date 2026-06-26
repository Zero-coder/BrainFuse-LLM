# Reproducibility

This document summarizes the experimental protocol reported in the BrainFuse-LLM manuscript.

## Tasks

- Brain captioning: generate natural-language captions from fMRI responses.
- fMRI question answering: answer textual questions using fMRI responses as implicit visual evidence.
- Unseen-subject generalization: train on seven NSD subjects and evaluate on a held-out subject.
- Selected reasoning tasks: evaluate affective, affordance, activity, and counting reasoning categories.

## Model Configuration

- LLM backbone: Vicuna-7B.
- fMRI tokens: 128.
- Learnable latent queries: 1024.
- Encoder hidden dimension: 256.
- LoRA-MoE experts: 4.
- Routing: token-wise top-1.
- Trainable parameters: fMRI encoder, projection layers, LoRA experts, and router.
- Frozen parameters: backbone LLM.

## Optimization

- Optimizer: AdamW.
- Learning rate: 5e-5.
- Betas: 0.9, 0.996.
- Weight decay: 0.009.
- Batch size: 64.
- Inference: greedy decoding for stable evaluation.

## Metrics

- Captioning: BLEU, METEOR, ROUGE, CIDEr, SPICE.
- fMRI-QA: accuracy, ANLS, and RMSE where appropriate.
- TDIUC selected categories: arithmetic and harmonic mean accuracy over selected tasks.

## Representative Results

- NSD captioning CIDEr: 62.30.
- COCO-QA accuracy: 51.23.
- VQA-v2 accuracy: 54.62.
- A-OKVQA accuracy: 55.37.
- TallyQA RMSE: 1.57.

These numbers correspond to the manuscript tables and require the complete processed NSD/VQA resources.
