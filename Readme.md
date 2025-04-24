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

- For the **MicroKPNN_Confounder-Free Model**:
```
  python MicroKPNN_encoder_confounder_free_lib/run_pipeline.py
  python MicroKPNN_encoder_confounder_free_lib/main.py
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

1. Implement **early stopping** in the training process. (Mahsa)
2. ~~Add the last layer activatin funciton, latent dim, batch size and norm as hyper parameters (remove dropout as confounder parameter) (Mahsa)~~
3. ~~Try to find more crc data for pretraining they need to have gender (Mahsa)~~
4. ~~Add MicroKPNN confounder free (Mahsa)~~
5. ~~Clean MicroKPNN_Confounder_free_lib and create edges scripts (Mahsa)~~
6. Add EDA for each dataset (Mahsa)
7. Add explainability script (Mahsa)
8. ~~Add all benchmarks (MicroKPNN, SVM, RF) (Mahsa)~~
9. Implement MicroKPNN-MT as a benchmark (Mahsa)
10. try our model on metacardis data again

9. Read train.py carefully (Yuhui)
10. Pretraining on the wole crc dataset (Yuhui)

11. Try different loss function instead of pearson loss function
12. For the **CRC dataset**, remove **age** and **BMI** as they are potential confounders.
13. Plot pca and tsne on both train and test to see samples distribution
14. Data Augmentation (phylomix)



### 📂 For Yuhui

You can find the pretraining data for CRC in the `dataset/pretrain_CRC_data/` directory.

Note that this dataset contains **more features** than the one we use for training (`dataset/CRC_data/new_crc_abundance_PRJEB6070.csv`).  
Feel free to remove any extra columns that are not present in the training dataset if you think it would improve consistency or simplify the modeling.
