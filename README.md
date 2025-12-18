# SynFS
Discovering Synergistic Features in Multi-View Data

## Set Enviroment 

### step 1. create env
```bash
conda create -n synfs python=3.10 -y
conda activate synfs
```

### step 2. upgrade pip tooling
```
python -m pip install --upgrade pip setuptools wheel
```

### step 3. install requirements
```
pip install -r requirements.txt
```

## Run single experiment 

```
python train.py
```

## To use your own data
- revise datapath from config/data/default.yaml


## Use MLFLOW to track experiments

```bash
mllfow ui
```

## Use optuna sweeper
- from scripts/run_sweep.sh, set the range you want to sweep run below cmd. it will log the the experiments to your mlflow and return best hyperparameteres
```bash
bash run_sweep.sh 
```

## Get feature importance 
- use get_important_features from src/utils/feature_importance.py 
- select which interaction to see "synergistic" | "non_synergistic"
- set threshold you over which select
- it will return the index of important features (you can also see name of feature if you pass featre_names)
- example output

[synergistic] threshold=0.7
  total features     : 500
  selected features  : 2
  indices: [  0 251]
