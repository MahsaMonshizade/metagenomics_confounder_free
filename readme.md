conda :

```
conda create --name confounder_free python=3.8
conda activate confounder_free
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
conda install pandas numpy scikit-learn
conda install conda-forge::tqdm
```



August 2nd 2024

I just realized that my training dataset consists entirely of women. To address this, we have two options: create new training and test datasets, or change the confounder to age. I'll proceed with the first option.

After conducting a thorough exploratory data analysis (EDA), I determined that my current dataset does not contain any diseases featured in multiple projects with sufficient metadata to use as a confounder. The only disease identified was preterm birth (D047928), with age considered as a confounder. However, upon reviewing the literature, I realized that age directly impacts this phenotype.

to dos:

1. try to use idnmdb dataset since they come from 4 or 5 different sources and use gender as confounder
2. clean my code and add comments