# Dataset Notes

BrainFuse-LLM experiments use the Natural Scenes Dataset (NSD) and annotations derived from COCO/VQA-style resources. Raw fMRI data and third-party annotations are not redistributed in this repository.

## NSD Source

NSD contains multi-subject 7T fMRI responses to natural images. In the paper, each subject has 24,980 training samples and 982 test samples, with repeated test responses averaged for stable evaluation. The number of valid voxels varies across subjects.

Users should follow the official NSD access process and comply with the NSD data-use terms.

## Expected Processed Schema

Each processed sample should contain:

```json
{
  "sample_id": "subj01_000001",
  "subject": 1,
  "image_id": 123456,
  "voxel_path": "data/processed/voxels/subj01_000001.npy",
  "coord_path": "data/processed/coords/subj01.npy",
  "region_path": "data/processed/regions/subj01.npy",
  "task": "captioning",
  "instruction": "Describe the perceived image.",
  "target": "a short caption or answer"
}
```

For fMRI-QA, `instruction` stores the question and `target` stores the normalized answer. For captioning, `instruction` may be a fixed prompt.

## Splits

The paper follows standard NSD train/test splits. For unseen-subject generalization, models are trained on seven subjects and evaluated on the held-out eighth subject.

## Data Availability Statement

The repository provides code, configuration templates, and preprocessing interfaces. Raw data should be downloaded from the original sources.
