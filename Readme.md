# Libraries Overview

Here, we provide descriptions of different libraries included in this repository:

- **`FCNN_encoder_confounder_free_lib`**:  
  Code implementing a GAN-based approach for confounder-free modeling. The encoder part is a fully connected neural network.

- **`FCNN_lib`**:  
  Benchmark implementation. A fully connected neural network with the same number of layers as `FCNN_encoder_confounder_free_lib`.

- **`lib`**:  
  This library can be ignored.
 
- **`MicrokPNN_encoder_confounder_free_lib`**:  This is our model
  Code implementing a GAN-based approach for confounder-free modeling. The encoder part is similar to MicroKPNN. (the first layer of encoder is MicroKPNN and the rest is fully connected)

- **`MicroKPNN_lib`**:  
  MicroKPNN implementation without the confounder-free component.

- **`SVM_lib`**:  
  Benchmark implementation using Support Vector Machine (SVM).

- **`RF_lib`**:  
  Benchmark implementation using Random Forest (RF).

- **`Taxonomy_encoder_confounder_free_lib`**:  
  Code implementing a GAN-based approach for confounder-free modeling. The encoder uses a hierarchical taxonomy from genus to phylum.  
  **Note**: This implementation may have issues. The following warning occurs:  
  ```plaintext
  miniconda3/envs/confounder_free/lib/python3.8/site-packages/sklearn/metrics/_classification.py:1469: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
    _warn_prf(average, modifier, msg_start, len(result))
