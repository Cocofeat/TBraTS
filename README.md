# TBraTS
* This repository provides the code for our accepted MICCAI'2022 paper "TBraTS: Trusted Brain Tumor Segmentation"
* Official implementation of [TBraTS: Trusted Brain Tumor Segmentation](https://github.com/Cocofeat/TBraTS/)
## Introduction
Despite recent improvements in the accuracy of brain tumor segmentation, the results still exhibit low levels of confidence and robustness. Uncertainty estimation is one of effective way to change this situation, as it provides a measure of confidence in the segmentation results. Although many uncertainty estimation methods for brain tumor segmentation have been proposed, they focus excessively on segmentation accuracy. In this paper, a trusted brain tumor segmentation network is proposed to provide robust segmentation results and reliable uncertainty estimations without excessive computational burden and modification of the backbone network. Instead of using dropout at the test phase, we recommend using subjective logic theory to explicitly model uncertainty. We treat the predictions of the backbone neural network as subjective opinions by parameterizing the segmented class probabilities as the Dirichlet distributions. Further, the trusted segmentation framework learns the function that gathers reliable evidence from the data leading to the final segmentation. Overall, the unified trusted segmentation framework endows the model with reliability and robustness to out-of-distribution samples. To evaluate the effectiveness of our model in robustness and reliability, qualitative and quantitative experiments are conducted on the BraTS 2019 dataset.

![image](https://github.com/Cocofeat/TBraTS/blob/main/image/F1N.png)
## Requirements
Some important required packages include:  
Pytorch version >=0.4.1.  
Visdom  
Python == 3.7  
Some basic python packages such as Numpy.  
##  :fire: NEWS :fire:
* We will release the code as soon as possible. 
## Citation
If you find our work is helpful for your research, please consider to cite:  
```
@inproceedings{Coco2022TBraTS,
  title={TBraTS: Trusted Brain Tumor Segmentation},
  author={Zou, Ke and Yuan, Xuedong and Shen, Xiaojing and Wang, Meng and Fu, Huazhu},
  booktitle={International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI)},
  year={2022}
}
```
## Acknowledgement
Part of the code is revised from [TransBTS](https://github.com/Wenxuan-1119/TransBTS) 

## Contact
* If you have any problems about our work, please contact [me](https://mail.google.com/kezou8@gmail.com) 
* Project Link: [TBraTS](https://github.com/Cocofeat/TBraTS/)
