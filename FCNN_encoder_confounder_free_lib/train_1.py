import torch
import numpy as np
import dcor
from sklearn.metrics import (
    balanced_accuracy_score, f1_score,
    precision_recall_curve, auc,
    precision_score, recall_score,
    confusion_matrix
)


def freeze_encoder_stats(module):
    """
    Put BatchNorm and Dropout layers of the encoder into eval mode
    so they do not update running stats or drop units.
    """
    for m in module.modules():
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.Dropout)):
            m.eval()


def restore_encoder_train(module):
    """
    Return BatchNorm and Dropout layers of the encoder to train mode
    so they update running stats and apply dropout.
    """
    for m in module.modules():
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.Dropout)):
            m.train()


def train_model(
     model, criterion, optimizer, data_loader, data_all_loader, data_val_loader, data_all_val_loader,
    data_test_loader, data_all_test_loader, num_epochs, 
    criterion_classifier, optimizer_classifier, 
    criterion_disease_classifier, optimizer_disease_classifier, device
):
    """
    Three-phase training per epoch:
      1) Confounder classifier (freeze encoder stats, freeze encoder weights)
      2) Distillation (update encoder, freeze classifier & stats)
      3) Disease classification (update encoder & disease head, normal stats)

    Validation and test only run the disease branch in eval mode.
    """

    # Prepare storage
    results = {
       "train": {
            "gloss_history": [],      # g_loss: distillation phase loss
            "loss_history": [],       # c_loss: disease classification loss
            "dcor_history": [],       # Distance correlation measure
            "accuracy": [],
            "f1_score": [],
            "auc_pr": [],
            "precision": [],
            "recall": [],
            "confusion_matrix": []
        },
        "val": {
            "loss_history": [],
            "dcor_history": [],
            "accuracy": [],
            "f1_score": [],
            "auc_pr": [],
            "precision": [],
            "recall": [],
            "confusion_matrix": []
        },
        "test": {
            "loss_history": [],
            "dcor_history": [],
            "accuracy": [],
            "f1_score": [],
            "auc_pr": [],
            "precision": [],
            "recall": [],
            "confusion_matrix": []
        }
    }

    # Move everything to device
    model.to(device)
    criterion.to(device)
    criterion_classifier.to(device)
    criterion_disease_classifier.to(device)

    for epoch in range(num_epochs):
        model.train()
        epoch_gloss = 0.0
        epoch_disease_loss = 0.0

        train_preds, train_labels, train_probs = [], [], []
        hidden_feats, hidden_targets = [], []

        conf_iter = iter(data_loader)
        all_iter = iter(data_all_loader)

        # ---- Training loop ----
        while True:
            # Phase 3 batch (disease)
            try:
                x_all, y_all = next(all_iter)
                x_all, y_all = x_all.to(device), y_all.to(device)
            except StopIteration:
                break

            # Phase 1 batch (confounder)
            try:
                x_conf, y_conf = next(conf_iter)
            except StopIteration:
                conf_iter = iter(data_loader)
                x_conf, y_conf = next(conf_iter)
            x_conf, y_conf = x_conf.to(device), y_conf.to(device)

            # -- Phase 1: Confounder classifier --
            freeze_encoder_stats(model.encoder)
            for p in model.encoder.parameters(): p.requires_grad = False
            for p in model.classifier.parameters(): p.requires_grad = True

            model.zero_grad()
            feats = model.encoder(x_conf)
            pred_conf = model.classifier(feats)
            loss_conf = criterion_classifier(pred_conf, y_conf)
            optimizer_classifier.zero_grad()
            loss_conf.backward()
            optimizer_classifier.step()

            # restore encoder for next phases
            restore_encoder_train(model.encoder)
            for p in model.encoder.parameters(): p.requires_grad = True

            # -- Phase 2: Distillation (adversarial) --
            for p in model.classifier.parameters(): p.requires_grad = False
            # freeze_encoder_stats(model.encoder)

            model.zero_grad()
            feats2 = model.encoder(x_conf)
            pred_conf2 = torch.sigmoid(model.classifier(feats2))
            g_loss = criterion(pred_conf2, y_conf)
            hidden_feats.append(feats2.detach().cpu())
            hidden_targets.append(y_conf.detach().cpu())

            optimizer.zero_grad()
            g_loss.backward()
            optimizer.step()
            epoch_gloss += g_loss.item()

            for p in model.classifier.parameters(): p.requires_grad = True
            # restore_encoder_train(model.encoder)

            # -- Phase 3: Disease classification --
            model.encoder.train()
            for p in model.encoder.parameters(): p.requires_grad = True
            for p in model.disease_classifier.parameters(): p.requires_grad = True

            model.zero_grad()
            feats_all = model.encoder(x_all)
            pred_all = model.disease_classifier(feats_all)
            loss_all = criterion_disease_classifier(pred_all, y_all)
            optimizer_disease_classifier.zero_grad()
            loss_all.backward()
            optimizer_disease_classifier.step()
            epoch_disease_loss += loss_all.item()

            # collect train metrics
            probs = torch.sigmoid(pred_all).detach().cpu()
            train_probs.extend(probs)
            train_preds.append((probs > 0.5).float())
            train_labels.append(y_all.cpu())

        # ---- End of training epoch: compute and store metrics ----
        n = len(data_all_loader)
        results["train"]["gloss_history"].append(epoch_gloss / n)
        results["train"]["loss_history"].append(epoch_disease_loss / n)

        preds = torch.cat(train_preds)
        labs  = torch.cat(train_labels)
        probs = torch.cat(train_probs)
        results["train"]["accuracy"].append(balanced_accuracy_score(labs, preds))
        results["train"]["f1_score"].append(f1_score(labs, preds))
        p, r, _ = precision_recall_curve(labs, probs)
        results["train"]["auc_pr"].append(auc(r, p))
        results["train"]["precision"].append(precision_score(labs, preds))
        results["train"]["recall"].append(recall_score(labs, preds))
        results["train"]["confusion_matrix"].append(confusion_matrix(labs, preds))

        feats_np = torch.cat(hidden_feats, 0).numpy()
        targs_np = torch.cat(hidden_targets, 0).numpy()
        results["train"]["dcor_history"].append(
            dcor.distance_correlation_sqr(feats_np, targs_np)
        )

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        val_preds, val_labels, val_probs = [], [], []
        val_hidden, val_targs = [], []

        with torch.no_grad():
            for x_v, y_v in data_all_val_loader:
                x_v, y_v = x_v.to(device), y_v.to(device)
                feats_v = model.encoder(x_v)
                pred_v = model.disease_classifier(feats_v)
                val_loss += criterion_disease_classifier(pred_v, y_v).item()
                p_v = torch.sigmoid(pred_v).cpu()
                val_probs.append(p_v)
                val_preds.append((p_v > 0.5).float())
                val_labels.append(y_v.cpu())

            for x_v, y_v in data_val_loader:
                x_v, y_v = x_v.to(device), y_v.to(device)
                val_hidden.append(model.encoder(x_v).cpu())
                val_targs.append(y_v.cpu())

        m = len(data_all_val_loader)
        results["val"]["loss_history"].append(val_loss / m)
        vp = torch.cat(val_probs)
        vpred = torch.cat(val_preds)
        vlab = torch.cat(val_labels)
        results["val"]["accuracy"].append(balanced_accuracy_score(vlab, vpred))
        results["val"]["f1_score"].append(f1_score(vlab, vpred))
        p_v, r_v, _ = precision_recall_curve(vlab, vp)
        results["val"]["auc_pr"].append(auc(r_v, p_v))
        results["val"]["precision"].append(precision_score(vlab, vpred))
        results["val"]["recall"].append(recall_score(vlab, vpred))
        results["val"]["confusion_matrix"].append(confusion_matrix(vlab, vpred))

        vh_np = torch.cat(val_hidden, 0).numpy()
        vt_np = torch.cat(val_targs, 0).numpy()
        results["val"]["dcor_history"].append(
            dcor.distance_correlation_sqr(vh_np, vt_np)
        )

        # ---- Test ----
        test_loss = 0.0
        test_preds, test_labels, test_probs = [], [], []
        test_hidden, test_targs = [], []

        with torch.no_grad():
            for x_t, y_t in data_all_test_loader:
                x_t, y_t = x_t.to(device), y_t.to(device)
                feats_t = model.encoder(x_t)
                pred_t = model.disease_classifier(feats_t)
                test_loss += criterion_disease_classifier(pred_t, y_t).item()
                p_t = torch.sigmoid(pred_t).cpu()
                test_probs.append(p_t)
                test_preds.append((p_t > 0.5).float())
                test_labels.append(y_t.cpu())

            for x_t, y_t in data_test_loader:
                x_t, y_t = x_t.to(device), y_t.to(device)
                test_hidden.append(model.encoder(x_t).cpu())
                test_targs.append(y_t.cpu())

        t = len(data_all_test_loader)
        results["test"]["loss_history"].append(test_loss / t)
        tp = torch.cat(test_probs)
        tpred = torch.cat(test_preds)
        tlab = torch.cat(test_labels)
        results["test"]["accuracy"].append(balanced_accuracy_score(tlab, tpred))
        results["test"]["f1_score"].append(f1_score(tlab, tpred))
        p_t, r_t, _ = precision_recall_curve(tlab, tp)
        results["test"]["auc_pr"].append(auc(r_t, p_t))
        results["test"]["precision"].append(precision_score(tlab, tpred))
        results["test"]["recall"].append(recall_score(tlab, tpred))
        results["test"]["confusion_matrix"].append(confusion_matrix(tlab, tpred))

        th_np = torch.cat(test_hidden, 0).numpy()
        tt_np = torch.cat(test_targs, 0).numpy()
        results["test"]["dcor_history"].append(
            dcor.distance_correlation_sqr(th_np, tt_np)
        )

        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] "
                  f"g_loss={results['train']['gloss_history'][-1]:.4f} "
                  f"d_loss={results['train']['loss_history'][-1]:.4f}")

    return results
