conda :

```
conda create --name confounder_free python=3.8
conda activate confounder_free
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
conda install pandas numpy scikit-learn
conda install conda-forge::tqdm
conda install -c conda-forge matplotlib
conda install anaconda::seaborn
conda install conda-forge::optuna
```

# Addressing Confounder Effects in Microbiome Studies

## Overview

Confounding variables can obscure the true relationships in microbiome research. This project focuses on:

## Objectives

1. **Enhancing Model Generalizability:** Improve model robustness to ensure it works well across different datasets with varying metadata.
   
2. **Identifying True Biomarkers:** Discover true biomarkers that affect host phenotypes without confounding influences.

## Goals

Our aim is to improve the accuracy and reliability of microbiome studies by effectively handling confounder effects.



For now please run baseline2 both for train_v2 and train v3 version (train v2 is imbalanced labels and v3 is balance labels )

*YH: baseline_v2 is great. I decreased the learning rate to 0.0001 and added parameter initialization in the model to avoid local optimization. The performance.png looks slightly better now, where the test accuracy and AUC do not tremor. Please note that the model is overfitting after about 10 epochs. We can address this later. Also, I wrote baseline_v2_kfold to check the data. Everthing looks good.* 

*MM: Increase the Learning Rate:
Try increasing the learning rate. If you're currently using a very small learning rate (e.g., 1e-4), try raising it to something like 1e-2 or even 1e-1 to amplify the impact of these small gradients.

Numerical Precision and Optimizer Mechanics:
Optimizers like Adam rely on first and second moments of gradients (i.e., the running averages of gradients and their squares). Even if gradients for both weight and bias are similar, differences in their momentum values might cause weights to have a very small or negligible update, while biases are updated.
If the magnitude of gradients is close to the optimizer's threshold for change, weights might effectively remain unchanged due to rounding or numerical stability issues.

Bias Correction in Adam:
The Adam optimizer uses bias correction for the running estimates of gradient moments. This correction might affect weight and bias parameters differently depending on the gradient history and the specific iteration.
*

*MM: when I just removed confounders using correlation coefficient directly on features I got better results comparing to this adversarial approach*
/// try to add waight initialization
## To-Do List

- [ ] Find projects for train and test dataset that has different distribution for metadata (e.g. gender), and focus on bindary classification

--> I tried to get the samples for gut microbiome from Mbodysample database; Large intestine, Small intestine and stomache; But it's not good because there is no phenotype that exists in 2 different projects and mostly just gender was available

--> GMrepo relativeve abundance info: #please note the abundance statistics, i.e. mean, median and sd were calculated in samples in which the corresponding taxons were detected.
#please consult our web site for more details.

https://evolgeniusteam.github.io/gmrepodocumentation/usage/downloaddatafromgmrepo/

- [ ] follow the toy example visualization

- [ ] See if you can have few metadata at the same time for confounders

- [ ] Use more metrics for training and evaluation such as accuracy

- [ ] make sure baseline has the same optimizer, architecture and etc as confounder_free model

- [ ] we could use the new architecture of neural networks (KAN): https://arxiv.org/abs/2404.19756 [optional]

- [ ] they fixed the database and I have access to the database again. I can download the data and do the preprocessing from the begining [https://mbodymap.microbiome.cloud/#/health&diseases]

random question: Ask Yuhui how she made awsome repo and ask Yuzhen if she thinks it's a good idea to do that for metagenomics




results so far:
### confounder_free_age_linear_correlation results:
```
train result --> Accuracy: 0.9785330948121646, Loss: 0.07159556448459625, AUC: 0.9989710144927536

eval result --> Accuracy: 0.9428571428571428, Loss: 0.18021175265312195, AUC: 0.96775
```
For balanced test dataset:
```
test result --> Accuracy: 0.765625, Loss: 0.668499231338501, AUC: 0.909423828125
```
For non balanced test dataset:
```
test result --> Accuracy: 0.5757575757575758, Loss: 1.5847922563552856, AUC: 0.7847118263473054
```

### confounder_free_age results:

```
train result --> Accuracy: 0.962432915921288, Loss: 0.14968928694725037, AUC: 0.9909710144927536

eval result --> Accuracy: 0.9285714285714286, Loss: 0.20735128223896027, AUC: 0.9784999999999999
```
For balanced test dataset:
```
test result --> Accuracy: 0.734375, Loss: 0.5888912081718445, AUC: 0.882080078125
```
For non balanced test dataset:
```
test result --> Accuracy: 0.5800865800865801, Loss: 1.1535238027572632, AUC: 0.8347679640718562
```


### baseline results:
```
train result --> Accuracy: 1.0000, Loss: 0.0019, AUC: 1.0000
eval result --> Accuracy: 0.9476, Loss: 0.1531, AUC: 0.9954
```
For balanced test dataset:
```
test result --> Accuracy: 0.7344, Loss: 0.9605, AUC: 0.8418
```
For non balanced test dataset:
```
test result --> Accuracy: 0.6667, Loss: 1.3016, AUC: 0.7856
```

### Results Summary

| Model                            | Train (Accuracy, Loss, AUC)           | Eval (Accuracy, Loss, AUC)           | Test (Balanced) (Accuracy, Loss, AUC) | Test (Non-balanced) (Accuracy, Loss, AUC) |
|-----------------------------------|---------------------------------------|--------------------------------------|----------------------------------------|--------------------------------------------|
| **confounder_free_age_linear_correlation** | 0.9785, 0.0716, 0.9990                | 0.9429, 0.1802, 0.9678               | **0.7656**, **0.6685**, **0.9094**                 | 0.5758, 1.5848, 0.7847                     |
| **confounder_free_age**           | 0.9624, 0.1497, 0.9910                | 0.9286, 0.2074, 0.9785               | 0.7344, 0.5889, 0.8821                 | 0.5801, 1.1535, 0.8348                     |
| **baseline**                      | **1.0000**, **0.0019**, **1.0000**                | **0.9476**, **0.1531**, **0.9954**               | 0.7344, 0.9605, 0.8418                 | 0.6667, 1.3016, 0.7856                     |


GAN alone: Likely sufficient, as it should remove both linear and non-linear correlations.
Linear correlation remover: May be useful as a check but could be redundant if the GAN is well-trained.

**sep 20:**

- for age I use inv_correlation_loss instead of mse and based on BR_net paper.
- I also used z_score for age 
- clean the confounder_free_age in the confounder_free_age_clean file

**sep 25**

- clean code for age 

(I beleive overfit happened for baseline)

For test data

| Metric                            | Baseline Model           | Confounder-Free Age Model    | 
|-----------------------------------|---------------------------------------|--------------------------------------|
| **Average Accuracy** | 0.5222 ± 0.0228              | 0.6610 ± 0.0538 |
| **Average AUC**           | 0.5718 ± 0.0447                | 0.7602 ± 0.0412              | 
| **Average F1 Score**                      | 0.6633 ± 0.0757                | 0.5990 ± 0.0931               | 

