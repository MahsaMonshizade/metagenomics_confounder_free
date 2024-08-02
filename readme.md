conda :

```
conda create --name confounder_free python=3.8
conda activate confounder_free
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
conda install pandas numpy scikit-learn
conda install conda-forge::tqdm
```

search for gan implementation in pytorch

I just realize my training has all women. what if we make another train and test or change the confounder to age. I'll go with First option