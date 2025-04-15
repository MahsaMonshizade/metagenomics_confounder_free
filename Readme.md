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

To run the models:

- For the **Fully Connected Network**:
```
  python FCNN_lib/main.py
```

- For the **Confounder-Free Model**:
```
  python FCNN_encoder_confounder_free_lib/main.py
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

Currently, the scripts are set to run 10 trials. You can increase this number as needed to improve optimization results.

## TODO List

1. **Check the code** and modify it if anything looks incorrect. *(Assigned: Yuhui)*
2. Perform **hyperparameter optimization** for all datasets.
3. For the **CRC dataset**, remove **age** and **BMI** as they are potential confounders.
4. **Find additional test data for CRC**. *(Assigned: Mahsa)*
5. **Find additional test data for IBS**. *(Assigned: Mahsa)*
6. **Search for more datasets in general**. *(Assigned: Mahsa)*
7. Investigate why the **results vary significantly across folds**.
8. Implement **early stopping** in the training process.
9. If the **feature count is high**, apply **feature engineering** to remove less important features before training the model.
10. From train data we can get some samples for test with different gender distribution

we should try to increase recall

11. To change BCEwithlogit with BCEloss
12. Try different loss function instead of pearson loss function
13. plot pca and tsne on both train and test to see samples distribution
14. check utils.py carefully (assigned: Yuhui)

In FCNN_encoder_confounder_free_lib I set encoder learning rate and classfier learnin rate to 0 but still recall decreased why?