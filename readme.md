conda :

```
conda create --name confounder_free python=3.8
conda activate confounder_free
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
conda install pandas numpy scikit-learn
conda install conda-forge::tqdm
conda install -c conda-forge matplotlib
conda install anaconda::seaborn
```

I used scheduler just like one we have in microkpnn-mt

I use weighted bce loss for imbalance data but it just made the results worse.


to dos:

1. follow the toy example visualization
2. try to make the models better  e.g. by using schedule for learning rate
3. make age into regression instead of classification
4. try to realize why did you decide not to use the train and test from previous project (if you think you can use it write code for it)
5. See if you can have few metadata at the same time for confounders
6. Use more metrics for training and evaluation such as accuracy

7. make sure baseline has the same optimizer, architecture and etc as confounder_free model

8. we could use the new architecture of neural networks (KAN): https://arxiv.org/abs/2404.19756

9. they fixec the database and I have access to the database again. I can download the data and do the preprocessing from the begining [https://mbodymap.microbiome.cloud/#/health&diseases]

random question: Ask Yuhui how she made awsome repo and ask Yuzhen if she thinks it's a good idea to do that for metagenomics
