
### Introduction
This repository is for our paper: 'Synergy-Guided Regional Supervision of Pseudo Labels for Semi-Supervised Medical Image Segmentation'.

#Data:
You could refer [MC-Net](https://github.com/ycwu1997/MC-Net) or [Co-BioNet](https://github.com/himashi92/Co-BioNet) for the preprocessed LA and Pancreas-CT Dataset and refer [SSL4MIS](https://github.com/HiLab-git/SSL4MIS) for BraTS2019 dataset.

#Training:
put the dataset in the LA folder,   
--Data   
----LA   
------2018LA_Seg_Training Set  
--------***/mri_norm2.h5  
     
----Pancreas    
------Pancreas_h5    
--------image****_norm.h5    

----BraTS2019    
------data   
--------BraTS19_****.h5   
    
```
cd code/
python train.py --labelnum 4 --gpu 0  --exp model1     
python test_3D.py --exp model1 --gpu 0     
```
      
### Acknowledgements:
Our code is origin from [SSL4MIS](https://github.com/HiLab-git/SSL4MIS) and [MC-Net](https://https://github.com/ycwu1997/MC-Net).Thanks for these authors for their valuable works.

### Citation
'''
@InProceedings{ WanTao_SynergyGuided_MICCAI2025,
                 author = { Wang, Tao AND Zhang, Xinlin AND Chen, Yuanbin AND Zhou, Yuanbo AND Zhao, Longxuan AND Tan, Tao AND Tong, Tong },
                 title = { { Synergy-Guided Regional Supervision of Pseudo Labels for Semi-Supervised Medical Image Segmentation } }, 
                 booktitle = {Medical Image Computing and Computer Assisted Intervention -- MICCAI 2025},
                 year = {2025},
                 publisher = {Springer Nature Switzerland},
                 volume = { LNCS 15967 },
                 month = {October},
                 pages = { 530 -- 540 },
              }
'''
