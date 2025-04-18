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

1. ~~**Check the code** and modify it if anything looks incorrect. *(Assigned: Yuhui)*~~
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
13. Plot pca and tsne on both train and test to see samples distribution
14. ~~Check utils.py carefully (assigned: Yuhui)~~
15. Read train.py carefully

In FCNN_encoder_confounder_free_lib I set encoder learning rate and classfier learnin rate to 0 but still recall decreased why?

why in FCNN_Plots we see that it converge very fast although we have around 1000 samples wirhg about 170 features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
I added the freeze and unfreeze for batch normalizarition in phase 1 and 2 in training.py in fcnn_encoder_confounder_free cause of following chatgpt suggested:   

Below is an explanation of some potential reasons why your confounder‑free training procedure does not yield the same results as your plain FCNN training even when you “disable” the confounder‐related updates by setting their learning rates to zero.


1. Differences in Training Procedure and Batch Handling

FCNN Training:

You use a single training loop that only updates the encoder and the disease classifier using the disease classification loss.
Each batch is used directly for the main prediction task, and the forward pass is “clean” (there’s only one forward–backward cycle per mini‑batch).
Confounder‑Free Training:

Your training loop is divided into three phases per epoch:
Phase 1 (Confounder Classifier Training):
– You freeze the encoder and update the confounder branch (classifier) using its loss.
Phase 2 (Distillation):
– You freeze the classifier and update the encoder with the Pearson correlation loss.
Phase 3 (Disease Classifier Training):
– You update both the encoder and the disease classifier with the disease classification loss.
Even if you set the learning rates for the encoder and confounder classifier updates (in phases 1 and 2) to zero, you are still performing additional forward passes on different batches.
These extra passes can affect layers that have internal state (for example, batch normalization or dropout layers) by altering running statistics or dropout masks. In other words, even if no weight update occurs during phases 1 and 2, the computation of statistics may differ from the FCNN loop.

2. Interactions Between Freezing/Unfreezing Parameters

In your confounder‑free training, you are freezing and unfreezing different parts of the network (encoder and classifier) across phases.
Freezing by setting param.requires_grad = False stops gradient updates but does not necessarily “cancel out” the effect of doing extra forward passes.
When you unfreeze the parameters later, the encoder’s state (or the batch norm running means/variances) may have changed relative to the single forward pass of the FCNN training.
This additional manipulation of gradients and parameter freezing/unfreezing can introduce discrepancies in how the encoder learns the features for the main task.
3. Impact on Normalization Layers and Other Internal States

Batch Normalization and Dropout:
During training, batch normalization layers update their running averages based on the mini‑batch statistics.
In your confounder‑free training, if you perform multiple forward passes (even without weight updates) the running statistics will be updated differently compared to the FCNN training (which has only one forward pass per batch).
Similarly, dropout is applied randomly during training. Extra forward passes might lead to different effective dropout masks, which can subtly change the behavior of the network.


Summary

Even if you set the learning rate to zero for the branches responsible for confounder prediction or for the distillation phase, the overall training procedure in your confounder‑free model is different from that of the plain FCNN:

Multiple phases and extra forward passes cause different updates to layer statistics (especially in batch normalization or dropout).
The freeze–unfreeze approach can alter the effective state of the encoder.
The order and frequency of updates are different, so the encoder and disease classifier are not updated in exactly the same way as in the single‑phase FCNN training.
These factors combined can explain why you are observing different results between your two training procedures.


# Training Script Comparison

This document summarizes the key differences between **`train1.py`** and **`train2.py`**, both of which implement a three‑phase training loop (confounder classifier, adversarial distillation, disease classifier) but differ in how they handle layer freezing, dropout, optimizers, and internal state.

---

## 1. Freezing / Unfreezing Encoder Layers

| Aspect                            | train1.py (`freeze_encoder_stats`)              | train2.py (`freeze_batchnorm`)                   |
|-----------------------------------|-------------------------------------------------|--------------------------------------------------|
| Layers frozen in Phase 1 & 2      | **BatchNorm1d, BatchNorm2d AND Dropout**        | **BatchNorm1d & BatchNorm2d only**               |
| Method                            | Iterates `module.modules()`, calls `.eval()/ .train()` on matching layers | Recursively walks `module.children()`, calls `.eval()/ .train()` only on BatchNorm |
| Restoring training mode           | `restore_encoder_train` returns both BN & Dropout to train mode | `unfreeze_batchnorm` returns only BN to train mode |

---

## 2. Weight vs. Stat Freezing

| Phase & Action                    | train1.py                                        | train2.py                                                   |
|-----------------------------------|---------------------------------------------------|-------------------------------------------------------------|
| **Phase 1 (confounder classifier)** | 1. Freeze BN+Dropout stats<br>2. `encoder.requires_grad=False`<br>3. update classifier | 1. Freeze BN stats only<br>2. rely on optimizer scope<br>3. update classifier |
| **Phase 2 (distillation)**        | 1. Freeze BN+Dropout<br>2. `classifier.requires_grad=False`<br>3. update encoder | 1. Freeze BN only<br>2. `classifier.requires_grad=False`<br>3. update encoder |

---

## 3. Dropout Handling

- **train1.py**  
  - Explicitly freezes **Dropout** in phases 1 & 2 so no units are dropped.
- **train2.py**  
  - Does **not** touch Dropout—units continue to be randomly dropped in all phases.

---

## 4. Optimizer vs. Manual `requires_grad`

- **train1.py**  
  - Uses manual `p.requires_grad = False/True` on encoder and classifier to control gradients.
- **train2.py**  
  - Relies primarily on separate optimizer parameter groups to isolate updates, toggles only classifier’s `requires_grad` in Phase 2.

---

## 5. Zeroing Gradients

- **train1.py**  
  - Calls both `model.zero_grad()` and the relevant `optimizer*.zero_grad()` before every backward pass.
- **train2.py**  
  - Uses `model.zero_grad()` and `optimizer.zero_grad()` but does not always zero the classifier or disease optimizer in the same order.

---

## 6. Mode Switching for Phase 3

- **train1.py**  
  - After Phase 2 calls `model.encoder.train()`, re‑enabling both BN and Dropout.
- **train2.py**  
  - Returns BN to train mode, but Dropout was never disabled so remains active throughout.

---

## 7. Hidden Features Collection

- **train1.py**  
  - Appends Phase 2 encoder outputs into `hidden_feats` for distance‑correlation computation.
- **train2.py**  
  - Uses `hidden_activations_list` but otherwise identical in purpose.

---

## Conclusion

To unify behavior between these two scripts, ensure that:

1. **The same layers** (BatchNorm ± Dropout) are frozen and unfrozen in the same phases.  
2. **Gradient control** (manual `requires_grad` vs. optimizer scoping) is applied consistently.  
3. **Dropout** is either always active or always frozen whenever you intend.  
4. **Gradient zeroing order** is consistent across all optimizers.  

With these alignments, both implementations will produce identical training dynamics for the adversarial confounder‑removal workflow.```
