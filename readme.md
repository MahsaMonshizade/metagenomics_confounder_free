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

to do:
1. it's okay to use preterm data as well
2. try to use ibdmdb dataset since they come from 4 or 5 different sources and use gender as confounder
3. clean my code and add comments

August 3rd 2024

I added a new train and new test data for disease binary classification and confounder is age.

to do:
1. have a baseline
2. clean the code
3. add comments
4. make the performance better and check the accuracy as well
