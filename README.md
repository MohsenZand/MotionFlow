# [MotionFlow: Flow-based Spatio-Temporal Structured Prediction of Dynamics](https://arxiv.org/pdf/2104.04391.pdf)


## Dependencies
* [PyTorch](https://pytorch.org)

This code is tested under Ubuntu 18.04, CUDA 11.2, with one NVIDIA Titan RTX GPU.
Python 3.8.8 version is used for development.

## Datasets
* Weizmann Horse Database, Download [here](https://www.msri.org/people/members/eranb/)
* [CMU Mocap](http://mocap.cs.cmu.edu/) Download [here](https://github.com/chaneyddtt/Convolutional-Sequence-to-Sequence-Model-for-Human-Dynamics/tree/master/data/cmu_mocap)


## Results
### Binary Segmentation
Set 'data_dir' and 'run_dir' in 'experiments/img_seg/flags_img_seg.py'\
Run 'experiments/img_seg/train.py'
<!--- You can use the trained model: 
- [Download]() the trained model and save it in the 'checkpoints' directory
- Set 'load_weights'= 'True' in 'experiments/img_seg/flags_img_seg.py'
- Run  'experiments/img_seg/test.py' --->

[<img src="https://github.com/MohsenZand/MotionFlow/blob/main/experiments/img_seg/seg_result.jpg" width="400"/>](https://github.com/MohsenZand/MotionFlow/blob/main/experiments/img_seg/seg_result.jpg)


### CMU Mocap
Prepare the dataset by running 'experiments/motion/preprocess_datasets.py'\
Set 'data_dir' and 'run_dir' in 'experiments/motion/flags_motion.py'\
For training, run 'experiments/motion/train.py'\
For evaluation and visualization, run 'experiments/motion/evaluation.py'


## Citation
Please cite our paper if you use code from this repository:
```
@article{zand2021flow,
  title={Flow-based Spatio-Temporal Structured Prediction of Dynamics},
  author={Zand, Mohsen and Etemad, Ali and Greenspan, Michael},
  journal={arXiv preprint arXiv:2104.04391},
  year={2022}
}
```

## References
[Conditional-Glow](https://github.com/yolu1055/conditional-glow)\
[Locally Masked Convolution](https://github.com/ajayjain/lmconv)