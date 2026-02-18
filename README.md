
# Artificial intelligence differentiates prefibrotic primary myelofibrosis with thrombocytosis from essential thrombocythemia using digitized bone marrow biopsy images

This repository contains the official code and model for the manuscript **"Artificial intelligence differentiates prefibrotic primary myelofibrosis with thrombocytosis from essential thrombocythemia using digitized bone marrow biopsy images"** (Srisuwananukorn et al., *Leukemia* 2026).

It provides a minimal, reproducible pipeline to:
1.  **Preprocess** whole slide images (WSIs) into feature bags using the exact normalization parameters from the study.
2.  **Predict** the probability of prePMF vs. ET using the trained Multiple Instance Learning (MIL) model.

## Citation

If you use this code or model in your research, please cite:

> Srisuwananukorn A, et al. Artificial intelligence differentiates prefibrotic primary myelofibrosis with thrombocytosis from essential thrombocythemia using digitized bone marrow biopsy images. *Leukemia*. 2026. DOI: 10.1038/s41375-026-02893-7.

## Data Requirements

### Annotation Manifest (Optional)
Provision of an annotation CSV is optional and only required if performance metrics (e.g., Accuracy, AUC) are desired. If provided, the file must contain the following columns:

| Column           | Description                                                          |
| :--------------- | :------------------------------------------------------------------- |
| `slide`          | **Required**. The unique filename of the WSI (without extension).    |
| `bmbx_diagnosis` | **Required for Metrics**. The ground truth label (`prePMF` or `ET`). |

Example format:
```csv
slide,bmbx_diagnosis
slide_001,prePMF
slide_002,ET
```

## Quick Start (Docker)

The provided Docker environment ensures a stable and reproducible execution of the pipeline.

### 1. Build the Docker Image
```bash
docker build -t prepmf_et_classifier .
```

### 2. Implementation: Preprocessing
The `preprocess.py` script automates tile extraction (302 µm) and feature generation (RetCCL) using study-specific normalization parameters.
```bash
docker run --gpus all \
    -v /path/to/wsis:/data \
    -v $(pwd)/output:/app/output \
    prepmf_et_classifier \
    python preprocess.py --wsi_dir /data --output_dir /app/output/features
```

### 3. Implementation: Prediction
The `predict.py` script executes the Attention-MIL model on the generated feature bags.
```bash
docker run --gpus all \
    -v $(pwd)/output:/app/output \
    prepmf_et_classifier \
    python predict.py --feature_dir /app/output/features --output_file /app/output/predictions.csv
```

## Local Environment Setup

For execution in a local Python environment (3.8+), ensuring GPU support and appropriate drivers is recommended.

1.  **Dependency Installation**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Preprocessing Workflow**:
    ```bash
    python preprocess.py --wsi_dir /path/to/slides --output_dir ./features
    ```

3.  **Inference Workflow**:
    ```bash
    python predict.py --feature_dir ./features --output_file predictions.csv
    ```

## Model Architecture Details

- **Feature Backbone**: RetCCL (Residual Convolutional Curation and Learning) trained on large-scale histopathology datasets.
- **Normalization**: Macenko spectral normalization utilizing a fixed target stain matrix derived from the primary training cohort.
- **Classifier**: Attention-based Multiple Instance Learning (MIL) Aggregation.
- **Resolution**: 299 px tiles at 1.01 µm/px (~302 µm field-of-view).
- **Environment**: Optimized for Slideflow 2.2.1 (verified repo environment).
