config = {
    # ### Colorectal cancer data with gender as confounder (we also can use age and bmi as confounder as well) (coming from GMrepo database)
    # "data": {
    #     "train_abundance_path": "dataset/CRC_data/crc_abundance_PRJEB6070.csv",
    #     "train_metadata_path": "dataset/CRC_data/crc_metadata_PRJEB6070.csv",
    #     "test_abundance_path": "dataset/CRC_data/crc_abundance_PRJNA397219.csv",
    #     "test_metadata_path": "dataset/CRC_data/crc_metadata_PRJNA397219.csv",
    #     "disease_column": "disease",
    #     "confounder_column": "sex"
    # },

    # ### Colorectal cancer data with gender as confounder (we also can use age and bmi as confounder as well) (coming from GMrepo database) with new test
    # "data": {
    #     "train_abundance_path": "dataset/CRC_data/new_crc_abundance_PRJEB6070.csv",
    #     "train_metadata_path": "dataset/CRC_data/crc_metadata_PRJEB6070.csv",
    #     "test_abundance_path": "dataset/CRC_data/crc_abundance_PRJEB27928.csv",
    #     "test_metadata_path": "dataset/CRC_data/crc_metadata_PRJEB27928.csv",
    #     "disease_column": "disease",
    #     "confounder_column": "sex"
    # },

    # ### IBS data with gender as confounder (coming from GMrepo database)
    # "data": {
    #     "train_abundance_path": "dataset/IBS_data/new_train_filtered.csv",
    #     "train_metadata_path": "dataset/IBS_data/train_metadata.csv",
    #     "test_abundance_path": "dataset/IBS_data/new_test_filtered.csv",
    #     "test_metadata_path": "dataset/IBS_data/test_metadata.csv",
    #     "disease_column": "disease",
    #     "confounder_column": "sex"
    # },

    ### T2D data with metformine as confounder (coming frm Metacardis dataset)
    "data": {
        "train_abundance_path": "dataset/MetaCardis_data/new_train_T2D_abundance_with_taxon_ids.csv",
        "train_metadata_path": "dataset/MetaCardis_data/train_T2D_metadata.csv",
        "test_abundance_path": "dataset/MetaCardis_data/new_test_T2D_abundance_with_taxon_ids.csv",
        "test_metadata_path": "dataset/MetaCardis_data/test_T2D_metadata.csv",
        "disease_column": "PATGROUPFINAL_C",
        "confounder_column": "METFORMIN_C"
    },
    
    "training": {
        "learning_rate": 0.0001,
        "num_epochs": 100,
        "batch_size": 256,
        "device": "cuda:0",  # Use "cpu" if no GPU available
        "weight_decay": 0, 
    },
    "model": {
        "latent_dim": 64,
        "num_encoder_layers": 2,    # One encoder layer beyond the initial layer
        "num_classifier_layers": 1, # One hidden layer for classifier networks
        "dropout_rate": 0.2,
        "norm": "batch",            # Options: "batch" or "layer"
        "classifier_hidden_dims": [],  # If provided, these override the halving rule
        "activation": "relu"        # Default activation function; can be 'relu', 'tanh', or 'leaky_relu'
    },
    "tuning": {
        # Discrete hyperparameter search space:
        "learning_rate": [1e-5, 2e-5, 5e-5,
                          1e-4, 2e-4, 5e-4,
                          1e-3, 2e-3, 5e-3,
                          1e-2, 2e-2, 5e-2,
                          1e-1],
        "dropout_rate": [0.0, 0.3, 0.5],
        "num_encoder_layers": [1, 2, 3],
        "num_classifier_layers": [1, 2, 3],
        "activation": ["relu", "tanh", "leaky_relu"]
    }
}
