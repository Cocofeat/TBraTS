# TBraTS
* This repository provides the code for our accepted MICCAI'2022 paper "TBraTS: Trusted Brain Tumor Segmentation"
* Official implementation [TBraTS: Trusted Brain Tumor Segmentation](https://arxiv.org/abs/2206.09309)
* Journal version [[Paper]](https://arxiv.org/abs/2301.00349)  [[code]](https://github.com/Cocofeat/UMIS)
* Official video in [MICS2022](https://aim.nuist.edu.cn/events/mics2022.htm) of [TBraTS: Trusted Brain Tumor Segmentation](https://www.bilibili.com/video/BV1nW4y1a7Qp/?spm_id_from=333.337.search-card.all.click&vd_source=6ab19d355475883daafd34a6daae54a5) (**3rd Prize**)

## Introduction
Despite recent improvements in the accuracy of brain tumor segmentation, the results still exhibit low levels of confidence and robustness. Uncertainty estimation is one effective way to change this situation, as it provides a measure of confidence in the segmentation results. In this paper, we propose a trusted brain tumor segmentation network which can generate robust segmentation results and reliable uncertainty estimations without excessive computational burden and modification of the backbone network. In our method, uncertainty is modeled explicitly using subjective logic theory, which treats the predictions of backbone neural network as subjective opinions by parameterizing the class probabilities of the segmentation as a Dirichlet distribution. Meanwhile, the trusted segmentation framework learns the function that gathers reliable evidence from the feature leading to the final segmentation results. Overall, our unified trusted segmentation framework endows the model with reliability and robustness to out-of-distribution samples. To evaluate the effectiveness of our model in robustness and reliability, qualitative and quantitative experiments are conducted on the BraTS 2019 dataset.

<div align=center><img width="900" height="400" alt="Our TBraTS framework" src="https://github.com/Cocofeat/TBraTS/blob/main/image/Trust_E.gif"/></div>

## Requirements
Some important required packages include:  
Pytorch version >=0.4.1.  
Visdom  
Python == 3.7  
Some basic python packages such as Numpy.  

## Data Acquisition
- The multimodal brain tumor datasets (**BraTS 2019**) could be acquired from [here](https://ipp.cbica.upenn.edu/).

## Data Preprocess
After downloading the dataset from [here](https://ipp.cbica.upenn.edu/), data preprocessing is needed which is to convert the .nii files as .pkl files and realize date normalization.

Follow the `python3 data/preprocessBraTS.py ` which is referenced from the [TransBTS](https://github.com/Wenxuan-1119/TransBTS/blob/main/data/preprocess.py)

## Training & Testing 
Run the `python3 trainTBraTS.py ` : your own backbone with our framework(U/V/AU/TransBTS)

Run the `python3 train.py ` : the backbone without our framework

##  :fire: NEWS :fire:
* [09/17] More experiments on trustworthy medical image segmentation please refer to [UMIS](https://github.com/Cocofeat/UMIS). 
* [09/17] We released all the codes. 
* [06/05] We will release the code as soon as possible. 
* [06/13] We have uploaded the main part of our code. We will upload all the code after camera-ready.
* [06/22] Our pre-printed version of the paper is available at [TBraTS: Trusted Brain Tumor Segmentation](https://arxiv.org/abs/2206.09309)
## Citation
If you find our work is helpful for your research, please consider to cite:  
```
@InProceedings{Coco2022TBraTS,
  author    = {Zou, Ke and Yuan, Xuedong and Shen, Xiaojing and Wang, Meng and Fu, Huazhu},
  booktitle = {Medical Image Computing and Computer Assisted Intervention -- MICCAI 2022},
  title     = {TBraTS: Trusted Brain Tumor Segmentation},
  year      = {2022},
  address   = {Cham},
  pages     = {503--513},
  publisher = {Springer Nature Switzerland},
}
}
```
## Acknowledgement
Part of the code is revised from [TransBTS](https://github.com/Wenxuan-1119/TransBTS) and [TMC](https://github.com/hanmenghan/TMC)

## Contact
* If you have any problems about our work, please contact [me](kezou8@gmail.com) 
* Project Link: [TBraTS](https://github.com/Cocofeat/TBraTS/)
