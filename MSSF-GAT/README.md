# MSSF-GAT

Official implementation of **MSSF-GAT** for multi-label classification on software model repositories built from the **ModelSet** dataset.

This project focuses on repository-level tagging/classification for **UML** and **Ecore** models. It provides:

- end-to-end data preprocessing from ModelSet exports
- node feature construction for software model graphs
- graph neural network training and evaluation
- the proposed **MS²F-GAT** model
- ablation study pipelines
- standard baseline implementations for **GCN**, **GAT**, **GraphSAGE**, **FFNN**, and **SVM**

---

## 1. Overview

Software models contain both rich textual semantics and non-trivial structural relations.  
This repository implements **MS²F-GAT**, a structure-aware multi-label classification framework that jointly exploits:

- **semantic information encoding**
- **structural-semantic fusion encoding**
- **multi-level feature fusion**

The framework is evaluated on processed **UML** and **Ecore** subsets derived from **ModelSet**.

---

## 2. Main Features

- Support for **UML** and **Ecore** datasets
- Multi-label graph classification with **PyTorch** and **PyTorch Geometric**
- Joint text-based node representations from model element names and types
- Baseline graph classifiers:
  - GCN
  - GAT
  - GraphSAGE
- Non-graph baselines:
  - FFNN
  - SVM
- Optuna-based hyperparameter optimization
- Ablation study support for:
  - w/o Semantic Information Encoding
  - w/o Structural-Semantic Fusion Encoding
  - w/o Multi-Level Feature Fusion

---

## 3. Repository Structure

```text
MODELSET_GAT_PROJECT
├── configs/                  # configuration files
├── data/                     # processed data, features, PyG datasets
├── fig/                      # figures used in the paper
├── results/                  # experiment outputs
├── scripts/                  # hyperparameter search and experiment scripts
├── src/
│   ├── Baselines/            # graph-level feature export for FFNN/SVM
│   ├── data_prep/            # metadata filtering / graph extraction / tag cleaning
│   ├── dataset/              # PyG dataset construction and label utilities
│   ├── engine/               # evaluator and trainer
│   ├── features/             # node feature construction scripts
│   └── models/               # model definitions
├── trainer_ablation.py       # ablation training entry
├── run_all_ablation.bat      # helper script for ablation experiments
└── requirements.txt
```

---

## 4. Environment

The experiments in our paper were conducted with the following environment:

- **Python**: 3.9.23
- **PyTorch**: 2.8.0+cu128
- **PyTorch Geometric**: 2.6.1
- **CUDA**: 12.8

Hardware:

- **GPU**: NVIDIA GeForce RTX 4070 SUPER (11.99 GB)
- **CPU**: 14 physical cores / 20 logical cores

> The code should also work in similar Python/PyTorch/PyG environments, but the above setup is the one used for our reported results.

---

## 5. Installation

### 5.1 Clone the repository

```bash
git clone https://github.com/WhatHowWhy1/MSSF-GAT.git
cd MSSF-GAT
```

### 5.2 Create environment

Using conda:

```bash
conda create -n modelset python=3.9
conda activate modelset
pip install -r requirements.txt
```

---

## 6. Dataset Preparation

This repository assumes that the raw data are derived from **ModelSet** and then processed into graph classification inputs.

### 6.1 Recommended pipeline

For each dataset (`uml` or `ecore`), the pipeline is:

1. **Filter metadata**
2. **Extract graphs to CSV**
3. **Clean tags**
4. **Split dataset**
5. **Build node features**
6. **Build PyG datasets**

Typical scripts include:

- `src/data_prep/01_filter_metadata_uml.py`
- `src/data_prep/01_filter_metadata_ecore.py`
- `src/data_prep/02_extract_graph_to_csv_uml.py`
- `src/data_prep/02_extract_graph_to_csv.py`
- `src/data_prep/03_clean_tags_uml.py`
- `src/data_prep/03_clean_tags.py`
- `src/dataset/split_dataset.py`
- `src/features/01_build_type_features.py`
- `src/features/04_build_joint_text_features.py`
- `src/dataset/build_pyg_dataset.py`

### 6.2 Processed split used in our experiments

The processed datasets follow this split ratio:

- **Train**: 72%
- **Validation**: 8%
- **Test**: 20%

The task is **multi-label graph classification**.

---

## 7. Node Features

This project supports multiple node feature settings.

### 7.1 Type feature
One-hot encoding of node types.

### 7.2 Name feature
Text embedding derived from node names.

### 7.3 Joint text feature
The main setting used in our best model.  
Each node is converted to a short text template such as:

```text
name: <node_name>; type: <node_type>
```

The text is then encoded into dense semantic vectors.

### 7.4 Graph-level pooled features
For FFNN and SVM baselines, graph-level features are exported from node-level semantic features via mean/max pooling.

---

## 8. Training the Proposed Model

The main proposed model is **MS²F-GAT**.

Example training command:

```bash
python trainer_ablation.py --dataset uml --config configs/full_gatv2_uml.yaml
python trainer_ablation.py --dataset ecore --config configs/full_gatv2_ecore.yaml
```

---

## 9. Ablation Study

This repository includes scripts and configs for the following ablations:

- **w/o Semantic Information Encoding**
- **w/o Structural-Semantic Fusion Encoding**
- **w/o Multi-Level Feature Fusion**

Example:

```bash
python trainer_ablation.py --dataset uml --config configs/ablation_wo_semantic_uml.yaml
python trainer_ablation.py --dataset uml --config configs/ablation_wo_structural_semantic_uml.yaml
python trainer_ablation.py --dataset uml --config configs/ablation_wo_fusion_uml.yaml
```

You can also use:

```bash
run_all_ablation.bat
```

to batch run ablation experiments on Windows.

---

## 10. Baselines

### 10.1 Graph baselines

Standard graph classification baselines implemented in `src/models/`:

- `baseline_gcn_classifier.py`
- `baseline_gat_classifier.py`
- `baseline_graphsage_classifier.py`

Hyperparameter search scripts:

- `scripts/hyperparameter_tune_baseline_gcn.py`
- `scripts/hyperparameter_tune_baseline_gat.py`
- `scripts/hyperparameter_tune_baseline_graphsage.py`

Example:

```bash
python -m scripts.hyperparameter_tune_baseline_gcn --dataset uml --config configs/full_gatv2_uml.yaml --n_trials 30 --study_name baseline_gcn_uml
python -m scripts.hyperparameter_tune_baseline_gat --dataset uml --config configs/full_gatv2_uml.yaml --n_trials 30 --study_name baseline_gat_uml
python -m scripts.hyperparameter_tune_baseline_graphsage --dataset uml --config configs/full_gatv2_uml.yaml --n_trials 30 --study_name baseline_graphsage_uml
```

### 10.2 FFNN and SVM baselines

First export graph-level pooled features:

```bash
python src/Baselines/export_graph_level_jointtext_features.py --dataset uml --config configs/full_gatv2_uml.yaml
python src/Baselines/export_graph_level_jointtext_features.py --dataset ecore --config configs/full_gatv2_ecore.yaml
```

Then run:

```bash
python -m scripts.hyperparameter_tune_baseline_ffnn --dataset uml --config configs/full_gatv2_uml.yaml --n_trials 30 --study_name baseline_ffnn_uml
python -m scripts.hyperparameter_tune_baseline_svm --dataset uml --config configs/full_gatv2_uml.yaml --n_trials 30 --study_name baseline_svm_uml
```

---

## 11. Hyperparameter Optimization

We use **Optuna** for hyperparameter search.

Typical search targets include:

- hidden dimension
- number of layers
- dropout
- learning rate
- weight decay
- batch size
- class imbalance weighting

For fair comparison, all competing models are tuned under the same validation protocol.

---

## 12. Evaluation

Reported metrics include:

- **Precision**
- **Recall**
- **Micro-F1**
- **Macro-F1**

Threshold selection is performed on the validation set, and the final selected model is evaluated on the held-out test set.

---

## 13. Results

The proposed **MS²F-GAT** consistently outperforms or matches strong baselines on both UML and Ecore, especially on **Micro-F1** and **Macro-F1**, demonstrating the effectiveness of jointly modeling semantic information and structure-aware interactions.

---


## 15. Acknowledgement

This project is built on top of the **ModelSet** dataset and related tooling from the model-driven engineering community.

---

## 16. Contact

For questions, issues, or collaboration, please open an issue on GitHub:

- https://github.com/WhatHowWhy1/MSSF-GAT
