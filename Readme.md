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
```
  python FCNN_lib/main.py
```

- For the **Confounder-Free Model**:
```
  python FCNN_encoder_confounder_free_lib/main.py
```

- For the **Confounder-Free Model** with **pre-training**:
```bash
  python FCNN_encoder_confounder_free_lib/pretrain_main.py
  # It is important to keep the batchsize in pretrain and finetune the same. 
  python FCNN_encoder_confounder_free_lib/finetune_main.py
```

- For the **MicroKPNN_Confounder-Free Model**: 
```
  python MicroKPNN_encoder_confounder_free_lib/run_pipeline.py
  python MicroKPNN_encoder_confounder_free_lib/main.py
```

- For the **MicroKPNN_Confounder-Free Model** with **pre-training**:
```bash
  # run `python MicroKPNN_encoder_confounder_free_lib/run_pipeline.py` first
  cp -r Results/MicroKPNN_encoder_confounder_free_plots/required_data Results/MicroKPNN_encoder_confounder_free_finetune_plots/
  python MicroKPNN_encoder_confounder_free_lib/pretrain_main.py
  # It is important to keep the batchsize in pretrain and finetune the same. 
  python MicroKPNN_encoder_confounder_free_lib/finetune_main.py
```

### Hyperparameter Optimization
We provide scripts to perform hyperparameter tuning:

For FCNN:
```
  python FCNN_lib/hyperparam_optimization.py
```

For Confounder_free
```
  python FCNN_encoder_confounder_free_lib/hyperparam_optimization.py
```

For MicroKPNN_Confounder_free
```
  python MicroKPNN_encoder_confounder_free_lib/run_pipeline.py
  python MicroKPNN_encoder_confounder_free_lib/hyperparam_optimization.py
```

Currently, the scripts are set to run 10 trials. You can increase this number as needed to improve optimization results.

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
13. Hyper parameter optimization for FCNN_CF, and MicroKPN_CF

**Later**

11. Try different loss function instead of pearson loss function
13. For the **CRC dataset**, remove **age** and **BMI** as they are potential confounders.
14. Plot pca and tsne on both train and test to see samples distribution
15. Data Augmentation (phylomix)


Mahsa:
1. add fine tuning to FCNN, MicroKPNN and MicroKPN_MT
2. Modufy code to sav the best epoch (based on accuracy) for FCNN, MicroKPNN and MicroKPN_MT (*Yuhui: I modified the performance saving for FCNN, but please double-check it.*)
3. hyper parameter optimization for FCNN, MicroKPNN and MicroKPN_MT
4. add the final results to readme for FCNN, MicroKPNN and MicroKPN_MT



### Results:

| Model | Train     |           |           |           |           | Validation |           |           |           |           | Test      |           |           |           |           |
|-------|-----------|-----------|-----------|-----------|-----------|------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
|       | Accuracy  | F1 Score  | AUC PR    | Precision | Recall    | Accuracy   | F1 Score  | AUC PR    | Precision | Recall    | Accuracy  | F1 Score  | AUC PR    | Precision | Recall    |
| RF    | 1.000     | 1.000     | 1.000     | 1.000     | 1.000     | 0.984      | 0.985     | 0.999     | 0.986     | 0.986     | 0.683     | 0.732     | 0.838     | 0.686     | 0.786     |
| SVM   | 0.999     | 0.999     | 0.999     | 1.000     | 0.999     | 0.990      | 0.991     | 0.999     | 0.993     | 0.989     | 0.654     | 0.711     | 0.796     | 0.660     | 0.771     |
| FCNN  | 1.000     | 1.000     | 1.000     | 1.000     | 1.000     | 0.985      | 0.987     | 0.999     | 0.982     | 0.993     | 0.667     | 0.712     | 0.794     | 0.677     | 0.751     |

**CRC data**

| Model           | Train     |           |           |           |           | Validation |           |           |           |           | Test      |           |           |           |           |
|-----------------|-----------|-----------|-----------|-----------|-----------|------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
|                 | Accuracy  | F1 Score  | AUC PR    | Precision | Recall    | Accuracy   | F1 Score  | AUC PR    | Precision | Recall    | Accuracy  | F1 Score  | AUC PR    | Precision | Recall    |
| FCNN            | 1.000     | 1.000     | 1.000     | 1.000     | 1.000     | 0.989      | 0.991     | 0.999     | 0.990     | 0.992     | 0.607     | 0.713     | **0.799** | 0.609     | 0.861     |
| FCNN-CF         | 0.993     | 0.993     | 0.999     | 0.999     | 0.986     | 0.985      | 0.986     | 0.996     | 0.990     | 0.981     | 0.593     | 0.707     | 0.734     | 0.597     | **0.865** |
| FCNN-CF FT      | 0.899     | 0.898     | 0.974     | 0.955     | 0.847     | 0.876      | 0.877     | 0.964     | 0.930     | 0.832     | 0.615     | 0.692     | 0.707     | 0.632     | 0.764     |
| MicroKPNN-CF    | 0.982     | 0.984     | 0.996     | 0.985     | 0.984     | 0.946      | 0.952     | 0.983     | 0.950     | 0.955     | 0.617     | 0.709     | 0.715     | 0.619     | 0.830     |
| MicroKPNN-CF FT | 0.979     | 0.981     | 0.995     | 0.987     | 0.974     | 0.959      | 0.962     | 0.978     | 0.967     | 0.959     | **0.643** | **0.722** | 0.756     | **0.650** | 0.813     |

- Now, the best epoch is calculated by validation accuracy. The average caculation is applied on the best epoch across folds.
- It is important to acknowledge that epoch selection based on validation set performance will inevitably introduce some degree of overfitting, meaning that optimal performance on the test set may **not** be selected. 
- The above table demonstrates that the FT models exhibit smaller gaps between validation and test performance, indicating better generalizability. Furthermore, the FT versions of each architecture outperform their non-FT counterparts on the test set. 
