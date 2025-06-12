import importlib
import sys

# Import your config and finetune_main modules
import config as cfg
import finetune_main

# Define your list of learning rate tuples: (learning_rate, encoder_lr, classifier_lr)
lr_combinations = [
    (2e-05,0.0001,0.0005),
]

# Loop over each set of learning rates
for i, (lr, enc_lr, clf_lr) in enumerate(lr_combinations):
    print(f"\n\n====== Running Fine-Tuning with Set {i+1}: "
          f"LR={lr}, Encoder_LR={enc_lr}, Classifier_LR={clf_lr} ======")

    # Update config in memory
    cfg.config["finetuning_training"]["learning_rate"] = lr
    cfg.config["finetuning_training"]["encoder_lr"] = enc_lr
    cfg.config["finetuning_training"]["classifier_lr"] = clf_lr

    # Optionally: create a subdirectory for each run
    cfg.config["finetuning_training"]["results_dir_suffix"] = f"_run{i+1}_lr{lr}_enc{enc_lr}_clf{clf_lr}".replace(".", "")

    # Run the main fine-tuning process
    finetune_main.main()
