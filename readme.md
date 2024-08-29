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






