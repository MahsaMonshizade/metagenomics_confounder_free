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
to dos:

1. follow the toy example visualization
2. try to make the models better  e.g. by using schedule for learning rate
3. make age into regression instead of classification
4. try to realize why did you decide not to use the train and test from previous project (if you think you can use it write code for it)
5. See if you can have few metadata at the same time for confounders
6. Use more metrics for training and evaluation such as accuracy

