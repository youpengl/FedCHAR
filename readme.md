# FedCHAR

### Dependencies

`pip install -r requirements.txt`

### Data Preparation

1. Download from: https://drive.google.com/file/d/1Q-i7n3PqEQ_8jcoJ4gQqlojqz5iohqRY/view?usp=sharing
2. Unzip data.zip and replace the existing empty folder 'data' with the decompressed folder 'data'.

### Code Structure

```Python
|-- FedCHAR
    |-- LICENSE
    |-- main.py
    |-- readme.md
    |-- requirements.txt
    |-- run.txt
    |-- data
    |   |-- readme.txt
    |-- flearn
    |   |-- models
    |   |   |-- Depth.py
    |   |   |-- FMCW.py
    |   |   |-- HARBox.py
    |   |   |-- IMU.py
    |   |   |-- MobiAct.py
    |   |   |-- UWB.py
    |   |   |-- WISDM.py
    |   |   |-- __init__.py
    |   |   |-- client.py
    |   |-- trainers
    |   |   |-- CHARbase.py // base for FedCHAR
    |   |   |-- FedCHAR-DC.py
    |   |   |-- FedCHAR.py
    |   |   |-- Newbase.py // base for FedCHAR-DC
    |   |-- utils
    |       |-- __init__.py
    |       |-- model_utils.py
    |       |-- tf_utils.py
    |-- record
        |-- Dataset
            |-- Depth
            |-- FMCW
            |-- HARBox
            |-- IMU
            |-- MobiAct
            |-- UWB
            |-- WISDM
```



### Run

In run.txt, we give some examples.

### Info

We will continue to update the codebase. 

If there are any questions, please feel free to contact with me. 

Email: youpengcs@gmail.com & youpengl@stu.xidian.edu.cn