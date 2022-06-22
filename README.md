# TBraTS
* This repository provides the code for our accepted MICCAI'2022 paper "TBraTS: Trusted Brain Tumor Segmentation"
* Official implementation of [TBraTS: Trusted Brain Tumor Segmentation](https://arxiv.org/pdf/2206.09309.pdf)
## Introduction
Despite recent improvements in the accuracy of brain tumor segmentation, the results still exhibit low levels of confidence and robustness. Uncertainty estimation is one effective way to change this situation, as it provides a measure of confidence in the segmentation results. In this paper, we propose a trusted brain tumor segmentation network which can generate robust segmentation results and reliable uncertainty estimations without excessive computational burden and modification of the backbone network. In our method, uncertainty is modeled explicitly using subjective logic theory, which treats the predictions of backbone neural network as subjective opinions by parameterizing the class probabilities of the segmentation as a Dirichlet distribution. Meanwhile, the trusted segmentation framework learns the function that gathers reliable evidence from the feature leading to the final segmentation results. Overall, our unified trusted segmentation framework endows the model with reliability and robustness to out-of-distribution samples. To evaluate the effectiveness of our model in robustness and reliability, qualitative and quantitative experiments are conducted on the BraTS 2019 dataset.

<div align=center><img width="900" height="400" alt="Our TBraTS framework" src="https://github.com/Cocofeat/TBraTS/blob/main/image/F1N.png"/></div>

## Requirements
Some important required packages include:  
Pytorch version >=0.4.1.  
Visdom  
Python == 3.7  
Some basic python packages such as Numpy.  
##  :fire: NEWS :fire:
* [06/05] We will release the code as soon as possible. 
* [06/13] We have uploaded our code.
* [06/22] Our pre printed version of the paper is available at [TBraTS: Trusted Brain Tumor Segmentation](https://arxiv.org/pdf/2206.09309.pdf)
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
or
```
@misc{Coco2022TBraTSarxiv,
      title={TBraTS: Trusted Brain Tumor Segmentation}, 
      author={Ke Zou and Xuedong Yuan and Xiaojing Shen and Meng Wang and Huazhu Fu},
      year={2022},
      eprint={2206.09309},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
```
## Acknowledgement
Part of the code is revised from [TransBTS](https://github.com/Wenxuan-1119/TransBTS) and [TMC](https://github.com/hanmenghan/TMC)

## Contact
* If you have any problems about our work, please contact [me](https://mail.google.com/kezou8@gmail.com) 
* Project Link: [TBraTS](https://github.com/Cocofeat/TBraTS/)
