# Log

## 05/05/2025

- Why when setting different 'num_epochs', the results are significantly different? 
- Different folds converge at different epochs, so an early stopping strategy needs to be implemented
**Add to preprocessing**: 
- Remove 2251-166 features
- Remove the samples where sum(166 features) < 20

feature size: 2251
feature columns used in pre-training: 

```bash
['562', '573', '729', '817', '818', '820', '821', '823', '851', '853', '876', '901', '905', '907', '1245', '1264', '1302', '1304', '1305', '1308', '1309', '1318', '1328', '1343', '1351', '1352', '1358', '1382', '1502', '1512', '1522', '1531', '1535', '1547', '1582', '1584', '1596', '1660', '1680', '1681', '1686', '1689', '1735', '1736', '1744', '2173', '2317', '28025', '28026', '28037', '28111', '28116', '28117', '28118', '28123', '29361', '29391', '29466', '33033', '33035', '33038', '33039', '35833', '39483', '39485', '39486', '39488', '39490', '39491', '39492', '39496', '39777', '39778', '40518', '40519', '40520', '40545', '43675', '45851', '46124', '46228', '46503', '46506', '47678', '52226', '53345', '53443', '61171', '68892', '74426', '84112', '84135', '88431', '100886', '102148', '106588', '113107', '116085', '154046', '154288', '165179', '166486', '169435', '187327', '204516', '208479', '214856', '216816', '218538', '239935', '246787', '261299', '291644', '291645', '292800', '301301', '310297', '310298', '322095', '328812', '328813', '328814', '329854', '333367', '338188', '341694', '357276', '358743', '360807', '363265', '371601', '387090', '387661', '410072', '411489', '446660', '454154', '454155', '457412', '457421', '457422', '469610', '469614', '471189', '487173', '487174', '487175', '552398', '626929', '626932', '626935', '626940', '649756', '658081', '658082', '658085', '658086', '658087', '658089', '658655', '658657', '665950', '665951', '674529', '751585', '1070699', '1099853', '1118060', '1118061', '1161942', '1288121', '-1', '197', '287', '306', '539', '546', '550', '569', '571', '576', '623', '624', '727', '739', '824', '827', '847', '849', '856', '860', '1255', '1260', '1261', '1262', '1334', '1338', '1354', '1363', '1366', '1379', '1381', '1383', '1579', '1583', '1590', '1598', '1599', '1613', '1623', '1625', '1632', '1633', '1656', '1683', '1685', '1701', '1703', '1727', '2047', '2051', '2702', '28125', '28126', '28127', '28130', '28133', '28197', '29347', '31971', '33031', '33037', '33043', '33959', '33968', '34028', '35517', '36834', '37734', ...]
```

feature size: 166
feature columns used in fine-tuning: 

```bash
['100886', '102148', '106588', '1070699', '1099853', '1118060', '113107', '116085', '1161942', '1245', '1264', '1288121', '1302', '1304', '1305', '1308', '1309', '1318', '1328', '1343', '1351', '1352', '1358', '1382', '1512', '1522', '1531', '1535', '154046', '154288', '1547', '1582', '1584', '1596', '165179', '1660', '166486', '1680', '1681', '1686', '1689', '169435', '1735', '1744', '187327', '204516', '208479', '214856', '216816', '2173', '218538', '2317', '239935', '246787', '261299', '28025', '28026', '28037', '28111', '28116', '28117', '28118', '28123', '291644', '291645', '292800', '29361', '29391', '29466', '301301', '310297', '310298', '322095', '328812', '328813', '328814', '329854', '33033', '33035', '33038', '33039', '333367', '338188', '341694', '357276', '35833', '358743', '360807', '363265', '371601', '387090', '387661', '39483', '39485', '39486', '39488', '39490', '39491', '39492', '39496', '39777', '39778', '40518', '40519', '40520', '40545', '410072', '411489', '43675', '446660', '454154', '454155', '457412', '457421', '457422', '45851', '46124', '46228', '46503', '46506', '469610', '469614', '471189', '47678', '487173', '487174', '487175', '52226', '53345', '53443', '552398', '562', '573', '61171', '626929', '626932', '626935', '626940', '649756', '658081', '658082', '658085', '658086', '658087', '658089', '658655', '658657', '665950', '665951', '674529', '68892', '729', '74426', '817', '818', '820', '821', '823', '84112', '84135', '851', '853', '876', '88431', '901', '905']
```

## 04/18/2025

- It seems that the poor generalizability is caused by normalization in the model, where the normalization parameters needed for training, validation, and test are very different. So, optimizing the following hyperparameters may help with this issue. 
- Optimization: increase `batch_size` to `256`, decrease `weight_decay` to `0`, decrease `dropout_rate` to `0`
- DEBUG: add a augment (`activation=model_cfg["activation"]`) for GAN in `./FCNN_encoder_confounder_free_lib/main.py`
- PROBLEM: the results of running `python FCNN_encoder_confounder_free_lib/hyperparam_optimization_test.py` are significantly different

```bash
# 1st run: 
Best trial:
  Final test F1: 0.7211383566634877
  Best hyperparameters:
    num_encoder_layers: 1
    num_classifier_layers: 3
    dropout_rate: 0.5
    learning_rate: 0.0005
    encoder_lr: 0.001
    classifier_lr: 1e-05
    activation: relu
# 2rd run: 
Best trial:
  Final test F1: 0.7033427692464809
  Best hyperparameters:
    num_encoder_layers: 3
    num_classifier_layers: 1
    dropout_rate: 0.5
    learning_rate: 0.0001
    encoder_lr: 0.001
    classifier_lr: 0.0005
    activation: tanh
# 3rd run:
Best trial:
  Final test F1: 0.7041237995889771
  Best hyperparameters:
    num_encoder_layers: 1
    num_classifier_layers: 2
    dropout_rate: 0.5
    learning_rate: 0.001
    encoder_lr: 0.001
    classifier_lr: 0.0001
    activation: relu
```

**A backup for ChatGPT's suggestion**

```
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

With these alignments, both implementations will produce identical training dynamics for the adversarial confounder‑removal workflow.
```

