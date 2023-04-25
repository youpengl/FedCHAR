# Hierarchical Clustering-based Personalized Federated Learning for Robust and Fair Human Activity Recognition

**News:** PyTorch Version is comming soon :)

This repository contains the code for our paper that has been accepted by **Ubicomp/IMWUT 2023**:

> [Hierarchical Clustering-based Personalized Federated Learning for Robust and Fair Human Activity Recognition](https://dl.acm.org/doi/10.1145/3580795)

**Abstract:** 

Currently, federated learning (FL) can enable users to collaboratively train a global model while protecting the privacy of user data, which has been applied to human activity recognition (HAR) tasks. However, in real HAR scenarios, deploying an FL system needs to consider multiple aspects, including system accuracy, fairness, robustness, and scalability. Most existing FL frameworks aim to solve specific problems while ignoring other properties. In this paper, we propose FedCHAR, a personalized FL framework with a hierarchical clustering method for robust and fair HAR, which not only improves the accuracy and the fairness of model performance by exploiting the intrinsically similar relationship between users but also enhances the robustness of the system by identifying malicious nodes through clustering in attack scenarios. In addition, to enhance the scalability of FedCHAR, we also propose FedCHAR-DC, a scalable and adaptive FL framework which is featured by dynamic clustering and adapting to the addition of new users or the evolution of datasets for realistic FL-based HAR scenarios. We conduct extensive experiments to evaluate the performance of FedCHAR on seven datasets of different sizes. The results demonstrate that FedCHAR could obtain better performance on different datasets than the other five state-of-the-art methods in terms of accuracy, robustness, and fairness. We further validate that FedCHAR-DC exhibits satisfactory scalability on three large-scale datasets regardless of the number of participants.

## Two Steps to Start

### 1. Install the dependencies

`pip install -r requirements.txt`

### 2. Run the provided examples

In run.txt, we give some examples.

## BibTeX Citation

```
@article{10.1145/3580795,
author = {Li, Youpeng and Wang, Xuyu and An, Lingling},
title = {Hierarchical Clustering-Based Personalized Federated Learning for Robust and Fair Human Activity Recognition},
year = {2023},
issue_date = {March 2023},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {7},
number = {1},
url = {https://doi.org/10.1145/3580795},
doi = {10.1145/3580795},
month = {mar},
articleno = {20},
numpages = {38},
keywords = {fairness, Human activity recognition, federated learning, attack and defense}
}
```

## Acknowledgements

- Code base: [Ditto](https://github.com/litian96/ditto)

- Dataset base: [IMU/UWB/Depth/HARBox](https://github.com/xmouyang/FL-Datasets-for-HAR), [FMCW](https://github.com/DI-HGR/cross_domain_gesture_dataset), [MobiAct](https://bmi.hmu.gr/the-mobifall-and-mobiact-datasets-2/), [WISDM](https://www.cis.fordham.edu/wisdm/dataset.php)

## Keep in touch with me

Please feel free to contact me with any questions about this paper or for the in-depth discussion and collaboration.

Email: youpengcs@gmail.com
