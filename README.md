# USSR



This repository is for the paper, "**Preception Consistency Ultarsound Images Super-resolution via Self-supervised CycleGAN**", accepted in  Neural Computing and Applications.

[Paper]([Perception consistency ultrasound image super-resolution via self-supervised CycleGAN | SpringerLink](https://link.springer.com/article/10.1007/s00521-020-05687-9))

<hr></hr>

## Datasets

+ CCA-US: http://splab.cz/en/download/databaze/ultrasound

+ US-CASE: http://www.ultrasoundcases.info/Cases-Home.aspx

  The US-Case image we used is available here:[百度网盘](https://pan.baidu.com/s/1V5fvczpYFVN_eR7L5zJGTA)，提取码：63yq

## 

## Results

Results on CCA-US dataset

<div align=center>
<img src="https://github.com/hengliusky/UltraSound_SSSR/blob/main/results/Results1.PNG"
</div>

<hr></hr>

Results on US-CASE dataset

<div align=center>
<img src="https://github.com/hengliusky/UltraSound_SSSR/blob/main/results/Results2.PNG">



## Requirements

+ python: 3.8.3
+ pytorch: 1.5.0
+ CUDA: 10.1
+ Ubuntu: 18.04

## Training && Testing

+ Split the dataset into training set and test set. In our method, we only need the test set.  
+ Use `generate_lr.m` to generate LR images and the corresponding HR images
+ Modify `input_path`  in `configs.py`
+ Run `python run_zssr_cycle.py  `
+  After the training, it will automatically test and get the final super-resolution result  



## Citing

If our method is useful for your research, please consider citing.

>```
>@article{liu2021perception,
>  title={Perception consistency ultrasound image super-resolution via self-supervised CycleGAN},
>  author={Liu, Heng and Liu, Jianyong and Hou, Shudong and Tao, Tao and Han, Jungong},
>  journal={Neural Computing and Applications},
>  pages={1--11},
>  year={2021},
>  publisher={Springer}
>}
>```

