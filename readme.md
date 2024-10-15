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



*MM: when I just removed confounders using correlation coefficient directly on features I got better results comparing to this adversarial approach*
/// try to add waight initialization
## To-Do List

- [ ] Find projects for train and test dataset that has different distribution for metadata (e.g. gender), and focus on bindary classification

https://evolgeniusteam.github.io/gmrepodocumentation/usage/downloaddatafromgmrepo/

- [ ] 1. Use dropout
- [ ] 2. Optimize the model using hyperparameter_optimization.py script
- [ ] 3. Try to find more metrics rather than just use dcor and also figure out how to get one value for mutual information
- [ ] 4. Change the order of model's training based on the paper
- [ ] 5. Read their github code again
- [ ] 6. Read both papers and supplementry again
- [ ] 7. Add gender metadata to the code as well (or any other usefull data)
- [ ] 8. Read code and check that everything is correct
- [ ] 9. Use another metrics for early stop (rn I think baseline is overfitting)
- [ ] 10. Add prior knowledge
- [ ] 11. Add explainability methods to find unbiased biomarkers
- [ ] 12. Ask chatgpt to help ake th idea better
- [ ] 13. Find more data
- [ ] 14. From "A realistic benchmark for differential abundance testing and confounder adjustment in human microbiome studies" paper I found their confounder dataset which is here: https://zenodo.org/records/6242715



- [ ] we could use the new architecture of neural networks (KAN): https://arxiv.org/abs/2404.19756 [optional]



random question: Ask Yuhui how she made awsome repo and ask Yuzhen if she thinks it's a good idea to do that for metagenomics



### Results Summary            


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

| Metric                            | Baseline Model           | Confounder-Free BMI Model    | Confounder-Free age Model | Confounder_Free BMI & age Model|
|-----------------------------------|---------------------------------------|--------------------------------------| --------------------------------------| --------------------------------------|
| **Average Accuracy** |     0.7448 ± 0.0627         | 0.6742 ± 0.0877 |         0.7036 ± 0.0627     | 0.6826 ± 0.0306 |
| **Average AUC**           |       0.8103 ± 0.0552         | .0.7596 ± 0.1058             |       0.7867 ± 0.0798    | 0.7822 ± 0.0629 |
| **Average F1 Score**                      |   0.8369 ± 0.0245             | 0.7963 ± 0.0868               |      0.8208 ± 0.0526       | 0.8336 ± 0.0435 |


***oct 2***
tried to have one distiller and one regressor and changed the loss from correlation to mse and used GradientReversalFunction but the results on test are worst: 

Final Evaluation on Test Data:
Average Accuracy: 0.6559 ± 0.0269
Average AUC: 0.7538 ± 0.0600
Average F1 Score: 0.7748 ± 0.0663

the scriot is: multitask_model.py


results for projection one:

Final Evaluation on Test Data:
Average Accuracy: 0.6889 ± 0.0588
Average AUC: 0.7848 ± 0.0804
Average F1 Score: 0.8542 ± 0.0295

***oct 7***
Try a new dataset name MetaCardis. The information about the labels find in the supplementry table 1 of following paper: https://www.nature.com/articles/s41586-021-04177-9#Sec15


*** oct 9 *** 
Final Evaluation on Test Data: (confounder_free model)
Average Accuracy: 0.7460 ± 0.0122
Average AUC: 0.8443 ± 0.0174
Average F1 Score: 0.6529 ± 0.0195

Final Evaluation on Test Data (Baseline Model):
Average Accuracy: 0.7490 ± 0.0255
Average AUC: 0.8383 ± 0.0088
Average F1 Score: 0.6506 ± 0.0381

*** oct 15 ***
fix the gradients claculation for each part of the model



to Yuhui: check the MetaCardis_lib_new. models.py is the one that just train the distiller. models_main is the one that train GAN.

to run the model you have to run the following: 

```
python MetaCardis_lib_new/main.py
```
