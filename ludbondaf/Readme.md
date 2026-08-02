# DBond-AF

This project implements the DBond-AF model proposed in the paper Enhancing Mirror-Image Peptide Bond Cleavage Prediction via Attention-Based Feature Fusion for mirror-image peptide bond cleavage prediction. The repository also provides ablation variants without the attention module, without feature concatenation, and without both, making reproduction and comparative analysis easier.

## Project Structure

```
.
├── Dockerfile                # Docker environment configuration
├── LICENSE                   # License
├── best_model/               # Best model weights
│   └── dbond_m_exp_af/
├── checkpoint/               # Model checkpoints generated during training
│   └── dbond_m_exp_af/
├── data_utils_dbond_af.py    # DBond-AF data processing utilities
├── dataset/                  # Dataset files
│   ├── *.csv
├── dbond_m_exp_af.py         # Main DBond-AF model script
├── dbond_m_exp_af_rm_attn.py # Ablation model without the attention module
├── dbond_m_exp_af_rm_cat.py  # Ablation model without feature concatenation
├── dbond_m_exp_af_rm_attn_cat.py # Ablation model without attention and feature concatenation
├── dbond_m_exp_af_config/    # DBond-AF configuration files
│   └── default.yaml
├── evaluate.dbond_m.exp_af.py # DBond-AF evaluation script
├── evaluate.dbond_m.exp_af_rm_attn.py # Evaluation script for the ablation model
├── evaluate.dbond_m.exp_af_rm_cat.py # Evaluation script for the ablation model
├── evaluate.dbond_m.exp_af_rm_attn_cat.py # Evaluation script for the ablation model
├── multi_label_metrics.py    # Multi-label evaluation metrics
├── result/                   # Output results
│   ├── metric/
│   └── pred/
├── tensorboard/              # TensorBoard logs
│   └── dbond_m_exp_af/
├── train.dbond_m.exp_af.py   # DBond-AF training script
├── train.dbond_m.exp_af_rm_attn.py # Training script for the ablation model
├── train.dbond_m.exp_af_rm_cat.py # Training script for the ablation model
├── train.dbond_m.exp_af_rm_attn_cat.py # Training script for the ablation model
└── Readme.md                 # Project documentation
```

## Main Files

- `dbond_m_exp_af.py`: Main DBond-AF model definition
- `dbond_m_exp_af_rm_attn.py` / `dbond_m_exp_af_rm_cat.py` / `dbond_m_exp_af_rm_attn_cat.py`: Ablation model definitions
- `train.dbond_m.exp_af.py`: DBond-AF training pipeline
- `evaluate.dbond_m.exp_af.py`: DBond-AF evaluation pipeline
- `multi_label_metrics.py`: Multi-label classification metrics
- `data_utils_dbond_af.py`: Data loading, encoding, and feature processing

## Quick Start

### 1. Environment

Prebuilt runtime images have been published to Docker Hub and are automatically built and pushed via GitHub Actions. Pulling the image directly is recommended:

```bash
docker pull LoserLus/dbond_af_env:latest
```

You can also pull an image tagged with the first 5 characters of a commit SHA to reproduce a specific version of the environment:

```bash
docker pull LoserLus/dbond_af_env:<short_sha>
```

Alternatively, build the image locally:

```bash
docker build -t dbond_af_env:local .
```

### 2. Data Preparation

Place the raw dataset in the `dataset/` directory. Use the existing CSV files as formatting references.

Training now expects three explicit splits in the config: `train_dataset_path`, `validation_dataset_path`, and `test_dataset_path`. The validation split is used for early stopping and best-model selection, while the test split should be reserved for final evaluation only.

### 3. Train the Model

Using the main DBond-AF model as an example:

```bash
python train.dbond_m.exp_af.py --config dbond_m_exp_af_config/default.yaml
```

Before training, make sure `validation_dataset_path` in the config points to a real validation CSV file.

For ablation experiments, use the corresponding training scripts:

```bash
python train.dbond_m.exp_af_rm_attn.py --config dbond_m_exp_af_config/default.yaml
python train.dbond_m.exp_af_rm_cat.py --config dbond_m_exp_af_config/default.yaml
python train.dbond_m.exp_af_rm_attn_cat.py --config dbond_m_exp_af_config/default.yaml
```

### 4. Evaluate the Model

#### DBond-AF Evaluation

```bash
python evaluate.dbond_m.exp_af.py \
	--in_model_weight_path best_model/dbond_m_exp_af/2026_04_14_17_25_default_0.pt \
	--in_model_comfig_path dbond_m_exp_af_config/default.yaml \
	--in_csv_to_predict_path dataset/dbond_m.test.csv \
	--out_multi_label_pred_dir result/pred/dbond_m_exp_af/multi/ \
	--out_multi_label_metric_dir result/metric/dbond_m_exp_af/multi/
```

#### Ablation Model Evaluation

```bash
python evaluate.dbond_m.exp_af_rm_attn.py \
	--in_model_weight_path checkpoint/dbond_m_exp_af/2026_04_14_17_25_default_0.pt \
	--in_model_comfig_path dbond_m_exp_af_config/default.yaml \
	--in_csv_to_predict_path dataset/dbond_m.test.csv \
	--out_multi_label_pred_dir result/pred/dbond_m_exp_af/multi/ \
	--out_multi_label_metric_dir result/metric/dbond_m_exp_af/multi/
```

For the other ablation variants, replace the evaluation script with `evaluate.dbond_m.exp_af_rm_cat.py` or `evaluate.dbond_m.exp_af_rm_attn_cat.py` while keeping the other arguments unchanged.

#### Argument Description

- `--in_model_weight_path`: Path to the model weight file
- `--in_model_comfig_path`: Path to the model configuration file (YAML)
- `--in_csv_to_predict_path`: Path to the CSV file for prediction
- `--out_multi_label_pred_dir`: Directory for saving multi-label prediction results
- `--out_multi_label_metric_dir`: Directory for saving multi-label evaluation metrics

The evaluation scripts automatically save prediction results and metrics to the specified directories.