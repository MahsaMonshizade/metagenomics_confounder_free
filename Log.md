# Log

04/18/2025: 

- It seems that the poor generalizability is caused by normalization in the model, where the normalization parameters needed for training, validation, and test are very different. So, optimizing the following hyperparameters may help with this issue. 
- Optimization: increase `batch_size` to `256`, decrease `weight_decay` to `0`, decrease `dropout_rate` to `0`
- DEBUG: add a augment (`activation=model_cfg["activation"]`) for GAN in `./FCNN_encoder_confounder_free_lib/main.py`

