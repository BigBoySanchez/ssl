# CAHSI Disaster AI Training
---
## ☁️ Run on Colab

### 1. Open colab_cahsi_disaster_train.ipynb in Colab
Search for the file in the shared student folder on Google Drive


---
### (The following steps are also outlined in the notebook)

### 2. Change the file paths to point to the colab_root folder
This folder will be in the same folder as `colab_cahsi_disaster_train.ipynb` 
```py
ROOT_FOLDER = "path/to/colab/root/" # <- CHANGE THIS
```

### 3, (optional) Modify the `sweep_ids.txt` file
Do this if you need to run less sweeps or a subset of sweeps (i.e. 50lb/cl only, kerala_floods only).

### 4. (optional) Change the `COUNT` variable
This determines how many runs each sweep does before switching to the next. NOTE: Sweeps won't run infinitely here, that only happens when running on the GPU server.

### 5. Log into wandb
Use the `wandb.login()` cell to do this. I recommend going to [https://wandb.ai/authorize]() to get your API key.

### 6. Run all cells

### 7. Monitor runs on wandb

---
## wandb Tips
- TERMS:
    - Run
        - An object representing the result of training a model using train.py (has config, score, artifacts)
    - Project
        - A group of runs representing an experiment
    - Sweep 
        - An object with a configuration for running multiple runs. Uses optimization methods like grid search, random, bayes. Random and bayes may run infinitely, or the sweep can be forced to "finish" optimization using 

- to organize runs, tags and sweeps can be used
    - ex: I group a project by sweeps, then use the lbcl tag to get the runs I want
- the only time to make new sweeps is when search space "shrinks"