# Hierarchical Clustering-based Personalized Federated Learning for Robust and Fair Human Activity Recognition

## PyTorch Version

**Note**: We aim to faithfully reproduce the algorithms, and if there are any errors, please submit an issue or contact us via email for discussion.

Additionally, we are providing running examples only on the WISDM dataset and including data partitioning code for reference. We recommend readers to employ grid search to discover optimal hyperparameters. We suggest a search range of 1-10 for "initial rounds," 2-10 for "n_clusters," and {0.01, 0.1, 1, 2} for $\mu$ (corresponding to $\lambda$ in the paper). Please modify the parameters in XXX_config.yaml.

For ease of visualization, the PyTorch version supports the visualization tool WandB.

### Running Examples:

```python
pip install -r requirements.txt

# Debug
FedCHAR: python3 system/main.py --algorithm FedCHAR --mode debug --project FedCHAR --device_id 0

FedCHAR-DC: python3 system/main.py --algorithm FedCHAR_DC --mode debug --project FedCHAR --device_id 0

# Using wandb
FedCHAR: python3 system/main.py --algorithm FedCHAR --mode single_wandb --project FedCHAR --device_id 0 --tag "FedCHAR"

FedCHAR-DC: python3 system/main.py --algorithm FedCHAR_DC --mode single_wandb --project FedCHAR --device_id 0 --tag "FedCHAR-DC"
```

