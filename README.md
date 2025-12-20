# SynFS : Discovering Features with Synergistic Interactions in Multiple Views

SynFS is a deep learning framework for disentangling synergistic and non-synergistic features in multi-view data.
It is designed for scenarios where predictive performance arises not only from individual views, but from interactions across views.

SynFS explicitly learns:

Synergistic features: features that are predictive only when combined across views

Non-synergistic features: features that are predictive independently within a single view

## Architiecture Overview

image.png

## Citation

if you add SynFS in your research, please cite: 

@inproceedings{kim2024discovering,
  title={Discovering features with synergistic interactions in multiple views},
  author={Kim, Chohee and Van Der Schaar, Mihaela and Lee, Changhee},
  booktitle={Forty-first International Conference on Machine Learning},
  year={2024}
}


## How to Use 

### Installation
```bash
git clone https://github.com/choheeK/SynFS.git
```

### Set Enviroment 

#### step 1. create env
```bash
conda create -n synfs python=3.10 -y
conda activate synfs
```

#### step 2. upgrade pip tooling
```
python -m pip install --upgrade pip setuptools wheel
```

#### step 3. install requirements
```
pip install -r requirements.txt
```

### Data Format
SynFS supports any number of views.

Example configuration:
/config/data/default.yaml
```
data:
  train:
    views:
      - path: SynFS/synthetic_dummy_data/view1.csv
      - path: SynFS/synthetic_dummy_data/view2.csv
    labels: SynFS/synthetic_dummy_data/y.csv

  val:
    views: null
    labels: null

views_dims: [10, 10]
batch_size: 64
```
Each view is shape of (N, D_v).

## Run single experiment 

```
python train.py
```

- you cana track your experiments with mlflow

```bash
mllfow ui
```

- For hyperparameter tuning use optuna sweeper 
- from scripts/run_sweep.sh, set the range you want to sweep run below cmd. it will log the the experiments to your mlflow and return best hyperparameteres
```bash
bash run_sweep.sh 
```

## Synthetic Data (Quick Start)


If no data paths are provided, SynFS automatically generates synthetic multi-view data:
```
from src.data.synfs_synthetic import generate_multi_dataset

views, y, meta = generate_multi_dataset(
    n=1000,
    dims=[10, 10],
    seed=42,
)

```
- if you want to use your own data revise datapath from config/data/default.yaml


## Inspecting Selected Features
- use get_important_features from src/utils/feature_importance.py 
- select which interaction to see "synergistic" | "non_synergistic"
- set threshold you over which select
- it will return the index of important features (you can also see name of feature if you pass featre_names)
- example output
```
[synergistic] threshold=0.7
  total features     : 500
  selected features  : 2
  indices: [  0 251]
```

## Repository Structure

```text
SynFS/
├── src/
│   ├── models/        # SynFS model & selectors
│   ├── trainer/       # Training logic
│   ├── data/          # Multi-view datasets
│   ├── utils/         # Logging, seeding
├── config/            # Hydra configs
├── synthetic_dummy_data/
├── train.py
└── README.md
```
