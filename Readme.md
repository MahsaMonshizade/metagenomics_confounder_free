## Project Structure and Instructions

We are working with three datasets:

1. **MetaCardis** – Already included in the paper.
2. **IBS** – New dataset.
3. **CRC** – New dataset.

### Code Libraries

We have two libraries:

- **Fully Connected Network (FCNN)** – Located in `FCNN_lib/`
- **Confounder-Free Model** – Located in `FCNN_encoder_confounder_free_lib/`

Each library has its own `config.py` file. Inside these files, you’ll see that two datasets are currently commented out. To use a different dataset, **comment out the current one and uncomment the dataset you want to use**.

### Running the Code

To prepare the datasets: 

```bash
chmod +x prepare_dataset.sh
./prepare_dataset.sh
```

To run the models:

- For the **Fully Connected Network**:
```bash
  python FCNN_lib/main.py

  # With pre-training
  python FCNN_lib/pretrain_main.py
  python FCNN_lib/finetune_main.py
```

- For the **Confounder-Free Model**:
```bash
  python FCNN_encoder_confounder_free_lib/main.py

  # With pre-training
  python FCNN_encoder_confounder_free_lib/pretrain_main.py
  # It is important to keep the batchsize in pretrain and finetune the same. 
  python FCNN_encoder_confounder_free_lib/finetune_main.py
```

- For the **MicroKPNN Model**:
```bash
  python MicroKPNN_lib/run_pipeline.py
  python MicroKPNN_lib/main.py

  # With pre-training
  cp -r Results/MicroKPNN_plots/required_data Results/MicroKPNN_finetune_plots/
  python MicroKPNN_lib/pretrain_main.py
  python MicroKPNN_lib/finetune_main.py
```

- For the **MicroKPNN_Confounder-Free Model**: 
```bash
  python MicroKPNN_encoder_confounder_free_lib/run_pipeline.py
  python MicroKPNN_encoder_confounder_free_lib/main.py

  # With pre-training
  # run `python MicroKPNN_encoder_confounder_free_lib/run_pipeline.py` first
  cp -r Results/MicroKPNN_encoder_confounder_free_plots/required_data Results/MicroKPNN_encoder_confounder_free_finetune_plots/
  python MicroKPNN_encoder_confounder_free_lib/pretrain_main.py
  # It is important to keep the batchsize in pretrain and finetune the same. 
  python MicroKPNN_encoder_confounder_free_lib/finetune_main.py
```

### Hyperparameter Optimization

We provide scripts to perform hyperparameter tuning:

- For FCNN:
```bash
  python FCNN_lib/hyperparam_optimization.py
```

- For Confounder_free: 
```bash
  python FCNN_encoder_confounder_free_lib/hyperparam_optimization.py
  python FCNN_encoder_confounder_free_lib/finetune_hyperparam_optimization.py
```

- For MicroKPNN: 
```bash
  python MicroKPNN_lib/run_pipeline.py

  python MicroKPNN_lib/hyperparam_optimization.py 
  python MicroKPNN_lib/finetune_hyperparam_optimization.py # NOT GOOD
```

- For MicroKPNN_Confounder_free: 
```bash
  python MicroKPNN_encoder_confounder_free_lib/run_pipeline.py
  
  python MicroKPNN_encoder_confounder_free_lib/hyperparam_optimization.py # NOT GOOD
  python MicroKPNN_encoder_confounder_free_lib/finetune_hyperparam_optimization.py # NOT GOOD
```

Currently, the scripts are set to run all trials in parallel. 

## TODO List

**Mahsa**

1. Implement **early stopping** in the training process. (Mahsa)
2. ~~Add the last layer activatin funciton, latent dim, batch size and norm as hyper parameters (remove dropout as confounder parameter) (Mahsa)~~
3. ~~Try to find more crc data for pretraining they need to have gender (Mahsa)~~
4. ~~Add MicroKPNN confounder free (Mahsa)~~
5. ~~Clean MicroKPNN_Confounder_free_lib and create edges scripts (Mahsa)~~
6. Add EDA for each dataset (Mahsa)
7. Add explainability script (Mahsa)
8. ~~Add all benchmarks (MicroKPNN, SVM, RF) (Mahsa)~~
9. ~~Implement MicroKPNN-MT as a benchmark (Mahsa)~~
10. try our model on metacardis data again (Mahsa)
11. try to fine tune MicroKPNN_confounder_free (Mahsa)
12. try to fine tune FCNN_confounder_free (Mahsa)
13. try to fine tune microkpnn_mt and make sure the code is correct (Mahsa)

**Yuhui**

10. ~~Read train.py carefully (Yuhui)~~
11. ~~Pretraining on the wole crc dataset (Yuhui)~~
12. ~~Early stop control/Save the results of best epoch rather than the last epoch.~~ (Save the performance of the best epoch according to valid acc for each fold, and average the performances among the best epochs.)
13. ~~Hyper parameter optimization for FCNN_CF, and MicroKPN_CF~~

**Later**

11. Try different loss function instead of pearson loss function
13. For the **CRC dataset**, remove **age** and **BMI** as they are potential confounders.
14. Plot pca and tsne on both train and test to see samples distribution
15. Data Augmentation (phylomix)

Mahsa and Yuhui:

1. ~~Yuhui: hyper parameter optimization (FCNN_encoder_confounder_free)~~
2. ~~Mahsa: hyper parameter optimization (MicroKPNN_encoder confounder free)~~
3. Mahsa: run svm and rf
4. Mahsa and Yuhui: MicroKPNN FT check the code (Yuhui: I checked the code, and the performance has been improved from 0.579 to 0.581, still not good enough.)
5. ~~analysis result explanation write it down in readme~~
6. Mahsa and Yuhui: should text for significance, e.g. by DeLong test or other test, to demonstrate our approach is beneficial
7. read me the paper "" explanation for table and table 2
8. The manuscript would benefit from a discussion on potentioal strategies to optimize both accuracy and interpretability
9. Explainability figures for this new dataset
10. MicroKPNN-MT for the meformine dataset (The one that we have in the paper already )

<!--| MicroKPNN FT    | 1.000     | 1.000     | 1.000     | 1.000     | 1.000     | 0.990      | 0.991     | 0.992     | 0.992     | 0.991     | 0.581     | 0.667     | 0.672     | 0.598     | 0.753     |--->
<!--| FCNN FT         | 1.000     | 1.000     | 1.000     | 1.000     | 1.000     | 0.990      | 0.992     | 1.000     | 0.992     | 0.991     | 0.615     | 0.728     | 0.770     | 0.610     | **0.903** |--->
<!--| FCNN-CF         | 0.927     | 0.932     | 0.981     | 0.949     | 0.916     | 0.866      | 0.881     | 0.947     | 0.881     | 0.884     | 0.606     | 0.699     | 0.760     | 0.612     | 0.815     |--->
<!--| MicroKPNN-CF    | 0.998     | 0.999     | 1.000     | 0.999     | 0.998     | 0.996      | 0.996     | 0.987     | 0.997     | 0.995     | 0.635     | 0.722     | 0.689     | 0.642     | **0.828** |--->


**CRC data**

| Model           | Train     |           |           |           |           | Validation |           |           |           |           | Test      |           |           |           |           |
|-----------------|-----------|-----------|-----------|-----------|-----------|------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
|                 | Accuracy  | F1 Score  | AUC PR    | Precision | Recall    | Accuracy   | F1 Score  | AUC PR    | Precision | Recall    | Accuracy  | F1 Score  | Precision | Recall    | 
| SVM             | 1.000     | 1.000     | 1.000     | 1.000     | 1.000     | 0.952     | 0.957     | 0.993     | 0.957     | 0.957     | 0.594     | 0.690     | 0.603     | 0.804     |
| RF              | 1.000     | 1.000     | 1.000     | 1.000     | 1.000     | 0.984     | 0.985     | 0.999     | 0.985     | 0.986     | 0.581     | 0.705     | 0.588     | **0.881**     |
| FCNN            | 1.000     | 1.000     | 1.000     | 1.000     | 1.000     | 0.989      | 0.991     | 0.999     | 0.990     | 0.992     | 0.607     | 0.713     | 0.609     | 0.861     |
| MicroKPNN       | 1.000     | 1.000     | 1.000     | 1.000     | 1.000     | 0.991      | 0.991     | 0.997     | 0.995     | 0.988     | 0.642     | 0.715     | 0.643     | 0.807     |
| MicroKPNN-MT    | 1.000     | 1.000     | 1.000     | 1.000     | 1.000     | 0.981      | 0.983     | 0.999     | 0.980     | 0.987     | 0.617     | 0.707     | 0.628     | 0.810     |
| FCNN-CF FT      | 0.873     | 0.868     | 0.964     | 0.946     | 0.804     | 0.842      | 0.834     | 0.940     | 0.923     | 0.769     | **0.663** | 0.727     | **0.671** | 0.799     |
| MicroKPNN-CF FT | 0.979     | 0.981     | 0.995     | 0.987     | 0.974     | 0.959      | 0.962     | 0.978     | 0.967     | 0.959     | 0.643     | **0.728** | 0.650     | 0.813     |

- Epoch Selection Method: Best epoch determined by validation accuracy, averaged across folds.
- Limitation: Validation-based epoch selection introduces overfitting risk - may **not** select optimal test performance. 

## 🧪 Gender Confounder Analysis Summary

**Chi-squared P-value (Gender vs. CRC):** `1.0290e-02`  
**Significant Microbial Features (P < 0.05):** `57`  
**Total Features Tested:** `171`  
**Result File:** `gender_microbiome_association.csv`

### 📊 Interpretation

The chi-squared test reveals a statistically significant association between gender and colorectal cancer (CRC) status (**p = 0.01029**), suggesting that **gender may act as a potential confounder** in microbiome-based CRC analyses.

Additionally, out of the 171 microbial features tested, **57 features** were found to be significantly associated with gender (**p < 0.05**). This indicates that gender has a notable impact on the microbiome composition.

> ⚠️ To avoid confounded or biased results, **gender should be accounted for** in any downstream analysis or predictive modeling involving this dataset.

For detailed feature-level results, see: [`gender_microbiome_association.csv`](./gender_microbiome_association.csv)



**T2D Metformine data**

| Model           | Train     |           |           |           |           | Validation |           |           |           |           | Test      |           |           |           |           |
|-----------------|-----------|-----------|-----------|-----------|-----------|------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
|                 | Accuracy  | F1 Score  | AUC PR    | Precision | Recall    | Accuracy   | F1 Score  | AUC PR    | Precision | Recall    | Accuracy  | F1 Score  | AUC PR    | Precision | Recall    | 
| SVM              | 1.0000    | 1.0000    | 1.0000    | 1.0000    | 1.0000    | 0.6782     | 0.8174    | 0.8972    | 0.8360    | 0.8000    | 0.7005    | 0.5918    | 0.6240    | 0.4964    | 0.7364    |
| RF               | 1.0000    | 1.0000    | 1.0000    | 1.0000    | 1.0000    | 0.6319     | 0.8748    | 0.9482    | 0.7943    | 0.9737    | 0.6174    | 0.5328    | 0.7453    | 0.3731    | 0.9318    |
| FCNN             | 0.9538    | 0.9624    | 0.9969    | 0.9888    | 0.9374    | 0.7231     | 0.8279    | 0.9270    | 0.8691    | 0.7939    | 0.7245    | 0.6171    | 0.6684    | 0.5371    | 0.7318    |
| MicroKPNN        | 0.9221    | 0.9398    | 0.9893    | 0.9758    | 0.9066    | 0.7513     | 0.8589    | 0.9221    | 0.8756    | 0.8444    | 0.7689    | 0.6770    | 0.7128    | 0.6314    | 0.7318    |
| MicroKPNN-MT     | 0.9952    | 0.9952    | 0.9997    | 1.0000    | 0.9904    | 0.7335     | 0.8832    | 0.9416    | 0.8554    | 0.9135    | 0.7762    | 0.6722    | 0.7554    | 0.5575    | 0.8542    |
| FCNN-CF          | 0.9502    | 0.9660    | 0.9847    | 0.9812    | 0.9515    | 0.7763     | 0.8657    | 0.9171    | 0.8913    | 0.8424    | 0.7927    | 0.7078    | 0.7368    | 0.6678    | 0.7591    |
| MicroKPNN-CF     | 0.8604    | 0.8953    | 0.9670    | 0.9477    | 0.8500    | 0.7909     | 0.8657    | 0.9254    | 0.9055    | 0.8323    | 0.7601    | 0.6667    | 0.7375    | 0.6842    | 0.6636    |


## 🧪 Confounder Analysis: Metformin

We performed a two-step statistical analysis to assess whether **metformin usage** may act as a confounding variable in our **T2D vs. control classification**:

### 1. 📊 Association Between Metformin and Disease Label

A Chi-squared test revealed a **strong, statistically significant association** between metformin usage and T2D diagnosis:

- **Chi-squared P-value**: `3.03e-66`

This result indicates that metformin is **not independently distributed** across T2D and control groups. Individuals with T2D are far more likely to be on metformin, suggesting that **metformin is a potential confounder**.

### 2. 🧬 Microbiome Feature Associations with Metformin

To understand how metformin influences microbial composition, we performed **Mann–Whitney U tests** for each microbial feature:

- **Total Microbial Features Tested**: `360`  
- **Significantly Associated Features (P < 0.05)**: `88`

This shows that **24% of microbial features** differ significantly between metformin users and non-users. These differences could **bias microbiome-based classifiers** if not properly controlled.

### ✅ Interpretation & Recommendation

Metformin is:

- **Associated with the disease label (T2D)**
- **Associated with significant microbial variation**

As a result, it should be **explicitly addressed as a confounder** in downstream analyses. Models that do not account for metformin may incorrectly attribute metformin-induced microbial changes as T2D-specific biomarkers.


## 🔍 Model Comparison Summary – AUC and Statistical Significance

We compared two models — `FCNN` and `FCNN Encoder Confounder Free` — on a shared test set to evaluate their predictive performance and determine if the observed difference in AUC (Area Under the ROC Curve) is statistically significant using DeLong's test.

### 📌 Sample Information
- **Total matched samples**: 143  
- **Positive samples**: 44 (30.8%)  
- **Negative samples**: 99 (69.2%)  
- **Common sample IDs**: 143  

### 📊 AUC Results
| Model                        | AUC     |
|-----------------------------|---------|
| FCNN                        | 0.8276  |
| FCNN Encoder Confounder Free | 0.8370  |
| **AUC Difference**           | **-0.0094** |

- **95% Confidence Interval**: [-0.0957, 0.0769]

### 🧪 DeLong’s Test for Statistical Significance
- **Z-statistic**: -0.2138  
- **Standard Error**: 0.0440  
- **P-value**: 0.8307  
- **Significance threshold**: *p* < 0.05  
- **Result**: AUC difference is **not statistically significant**

### ✅ Conclusion
While `FCNN Encoder Confounder Free` achieved a slightly higher AUC than the baseline `FCNN`, the improvement is small and **not statistically significant** according to DeLong’s test. This indicates that the observed AUC difference may be due to random chance rather than a true performance gain.




FCNN_CF:
 "training": {
        
        "num_epochs": 150,
        
        "batch_size": 64,
        
        "learning_rate": 0.0001,             # For disease classifier optimizer
        
        "encoder_lr": 0.002,                 # For encoder (e.g., for distillation phase)
        
        "classifier_lr": 0.005,              # For confounder classifier (e.g., 'drug' branch)
        
        "weight_decay": 0, #1e-4,
        
        "device": "cuda:0"                   # Change to "cpu" if GPU is unavailable
    },
    
    "model": {
        
        "latent_dim": 64,                    # Dimension of the latent space
        
        "num_encoder_layers": 3,             # Number of layers in the encoder (beyond initial projection)
        
        "num_classifier_layers": 2,          # Number of layers in each classifier branch
        
        "dropout_rate": 0.0,                 # Dropout probability (set to 0 to disable)
        
        "norm": "layer",                     # Normalization type ("batch" or "layer")
        
        "classifier_hidden_dims": [],        # Optional list; if empty, layers are created via halving
        
        "activation": "leaky_relu",                 # Activation function: options (e.g., "relu", "tanh", "leaky_relu")
        
        "last_activation": "tanh"
    },

MicroKPNN_CF:

"training": {

        "num_epochs": 150,
        
        "batch_size": 256,
        
        "learning_rate": 0.0002,             # For disease classifier optimizer
        
        "encoder_lr": 0.002,                 # For encoder (e.g., for distillation phase)
        
        "classifier_lr": 0.002,              # For confounder classifier (e.g., 'drug' branch)
        
        "weight_decay": 0, #1e-4,
        
        "device": "cuda:0"                   # Change to "cpu" if GPU is unavailable
    },
    "model": {
        
        "latent_dim": 64,                    # Dimension of the latent space
        
        "num_encoder_layers": 3,             # Number of layers in the encoder (beyond initial projection)
        
        "num_classifier_layers": 2,          # Number of layers in each classifier branch
        
        "dropout_rate": 0.0,                 # Dropout probability (set to 0 to disable)
        
        "norm": "layer",                     # Normalization type ("batch" or "layer")
        
        "classifier_hidden_dims": [],        # Optional list; if empty, layers are created via halving
        
        "activation": "relu",                 # Activation function: options (e.g., "relu", "tanh", 
        
        "leaky_relu")
        
        "last_activation": "tanh"}


### main Todo list:
1. reRun all the models for metformine (T2D) dataset
2. calculate delong
3. do the pca and tsne again
4. probably do interpretation again
5. Add the results for crc adn interpretation as well
6. If confusion looks good for removing false positive we can add that to the paper as well
**Key Insights**

- **Fine-tuning Benefits**: The FT version shows the value of transfer learning, achieving the best overall test performance despite slightly lower training metrics compared to the base model. This suggests better generalization and reduced overfitting.
- **Confounder-Free Advantage**: The CF approach helps the model focus on true predictive signals rather than spurious correlations, which is especially important in imbalanced datasets where confounders can lead to biased predictions toward the majority class. 
- **Precision-Recall Trade-off**: MicroKPNN-CF FT achieves the most favorable precision-recall balance for imbalanced data, delivering the highest precision (65.0%) while maintaining strong recall (81.3%). This indicates the model effectively handles class imbalance by being both selective (high precision) and comprehensive (high recall) in its predictions, resulting in the best F1 score (72.2%) - a metric particularly valuable for imbalanced datasets. 
