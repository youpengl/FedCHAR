# Debug
FedCHAR: python3 system/main.py --algorithm FedCHAR --mode debug --project FedCHAR --device_id 0

FedCHAR-DC: python3 system/main.py --algorithm FedCHAR_DC --mode debug --project FedCHAR --device_id 0

# Using wandb
FedCHAR: python3 system/main.py --algorithm FedCHAR --mode single_wandb --project FedCHAR --device_id 0 --tag "FedCHAR"

FedCHAR-DC: python3 system/main.py --algorithm FedCHAR_DC --mode single_wandb --project FedCHAR --device_id 0 --tag "FedCHAR-DC"