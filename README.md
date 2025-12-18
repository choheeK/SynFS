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

## Use MLFLOW to track experiments

```bash
mllfow ui
```

## Use optuna sweeper
- from scripts/run_sweep.sh, set the range you want to sweep run below cmd. it will log the the experiments to your mlflow and return best hyperparameteres
```bash
bash run_sweep.sh 
```
