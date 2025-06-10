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


**CRC data**

| Model           | Train     |           |           |           |           | Validation |           |           |           |           | Test      |           |           |           |           |
|-----------------|-----------|-----------|-----------|-----------|-----------|------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
|                 | Accuracy  | F1 Score  | AUC PR    | Precision | Recall    | Accuracy   | F1 Score  | AUC PR    | Precision | Recall    | Accuracy  | F1 Score  | AUC PR    | Precision | Recall    | 
| SVM             | 1.000     | 1.000     | 1.000     | 1.000     | 1.000     | 0.952     | 0.957     | 0.993     | 0.957     | 0.957     | 0.594     | 0.690     | 0.781 | 0.603     | 0.804     |
| FCNN            | 1.000     | 1.000     | 1.000     | 1.000     | 1.000     | 0.989      | 0.991     | 0.999     | 0.990     | 0.992     | 0.607     | 0.713     | **0.799** | 0.609     | 0.861     |
| FCNN FT         | 1.000     | 1.000     | 1.000     | 1.000     | 1.000     | 0.990      | 0.992     | 1.000     | 0.992     | 0.991     | 0.615     | 0.728     | 0.770     | 0.610     | **0.903** |
| MicroKPNN       | 1.000     | 1.000     | 1.000     | 1.000     | 1.000     | 0.991      | 0.991     | 0.997     | 0.995     | 0.988     | 0.642     | 0.715     | 0.728     | 0.643     | 0.807     |
| MicroKPNN FT    | 1.000     | 1.000     | 1.000     | 1.000     | 1.000     | 0.990      | 0.991     | 0.992     | 0.992     | 0.991     | 0.581     | 0.667     | 0.672     | 0.598     | 0.753     |
| MicroKPNN-MT    | 1.000     | 1.000     | 1.000     | 1.000     | 1.000     | 0.981      | 0.983     | 0.999     | 0.980     | 0.987     | 0.617     | 0.707     | 0.718     | 0.628     | 0.810     |
| FCNN-CF         | 0.927     | 0.932     | 0.981     | 0.949     | 0.916     | 0.866      | 0.881     | 0.947     | 0.881     | 0.884     | 0.606     | 0.699     | 0.760     | 0.612     | 0.815     |
| FCNN-CF FT      | 0.873     | 0.868     | 0.964     | 0.946     | 0.804     | 0.842      | 0.834     | 0.940     | 0.923     | 0.769     | **0.663** | 0.727     | 0.792     | **0.671** | 0.799     |
| MicroKPNN-CF    | 0.998     | 0.999     | 1.000     | 0.999     | 0.998     | 0.996      | 0.996     | 0.987     | 0.997     | 0.995     | 0.635     | 0.722     | 0.689     | 0.642     | **0.828** |
| MicroKPNN-CF FT | 0.979     | 0.981     | 0.995     | 0.987     | 0.974     | 0.959      | 0.962     | 0.978     | 0.967     | 0.959     | 0.643     | **0.728** | 0.756     | 0.650     | 0.813     |

- Epoch Selection Method: Best epoch determined by validation accuracy, averaged across folds.
- Limitation: Validation-based epoch selection introduces overfitting risk - may **not** select optimal test performance. 

**Key Insights**

- **Fine-tuning Benefits**: The FT version shows the value of transfer learning, achieving the best overall test performance despite slightly lower training metrics compared to the base model. This suggests better generalization and reduced overfitting.
- **Confounder-Free Advantage**: The CF approach helps the model focus on true predictive signals rather than spurious correlations, which is especially important in imbalanced datasets where confounders can lead to biased predictions toward the majority class. 
- **Precision-Recall Trade-off**: MicroKPNN-CF FT achieves the most favorable precision-recall balance for imbalanced data, delivering the highest precision (65.0%) while maintaining strong recall (81.3%). This indicates the model effectively handles class imbalance by being both selective (high precision) and comprehensive (high recall) in its predictions, resulting in the best F1 score (72.2%) - a metric particularly valuable for imbalanced datasets. 
