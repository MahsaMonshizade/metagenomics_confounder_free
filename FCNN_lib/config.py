config = {
    "data": {
        "train_abundance_path": "GMrepo/CRC_data/crc_abundance_PRJEB6070.csv",
        "train_metadata_path": "GMrepo/CRC_data/crc_metadata_PRJEB6070.csv",
        "test_abundance_path": "GMrepo/CRC_data/crc_abundance_PRJNA397219.csv",
        "test_metadata_path": "GMrepo/CRC_data/crc_metadata_PRJNA397219.csv",
        "disease_column": "disease",
        "confounder_column": "sex"
    },
    "training": {
        "learning_rate": 1e-4,
        "num_epochs": 100,
        "batch_size": 64,
        "device": "cuda:0",  # Use "cpu" if no GPU available
        "weight_decay": 1e-4
    },
    "model": {
        "latent_dim": 64,
        "num_encoder_layers": 1,    # One encoder layer beyond the initial layer
        "num_classifier_layers": 1, # One hidden layer for classifier networks
        "dropout_rate": 0.2,
        "norm": "batch",            # Options: "batch" or "layer"
        "classifier_hidden_dims": [],  # If provided, these override the halving rule
        "activation": "relu"        # Default activation function; can be 'relu', 'tanh', or 'leaky_relu'
    },
    "tuning": {
        # Discrete hyperparameter search space:
        "learning_rate": [1e-4, 5e-4, 1e-3],
        "dropout_rate": [0.0, 0.3, 0.5],
        "num_encoder_layers": [1, 2, 3],
        "num_classifier_layers": [1, 2, 3],
        "activation": ["relu", "tanh", "leaky_relu"]
    }
}
