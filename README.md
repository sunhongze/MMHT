# MMHT: object tracking model
We give a demo of our MMHT model for reviewers to check the reliability of our work. In our supplementary material, the entire training and test codes are given. Notably, limited by the size of file, we only give a example of our processed dataset.
##  Train on FE108 Dataset
1. Change dataset path at line 6 in 'ltr/admin/local.py'. 
2. Run ``` python train_mmht.py ```. You are free to change the training parameters in 'ltr/train_settings/MMHT/mmht.py'. 


##  Test on FE108 Dataset
1. Change dataset path at line 9 in 'pytracking/evaluation/local.py'.
2. Run ``` python run_tracker_mmht.py ```. You can set better test parameters in 'pytracking/parameter/fusion/mmht_para.py'. 
