

RegNet
==========

## Introduction
   In this study, we propose a neural network for cancer diagnosis and prognosis prediction in pathological images. To build 
a highly generalized and accurate AI aided diagnosis system, a lightweight convolution unit based on hierarchical split block (HSBlock) is carefully designed to improve the depth and width of the regular network (RegNet) for rich feature information extraction. The convolutional block attention module (CBAM) is introduced on the first and the last convolutions to better extract global features and local details.


## 1. Dependencies
- [Joblib](http://github.com/joblib/joblib) : Running Python functions as pipeline jobs.
- [Matplotlib](https://matplotlib.org/) A plotting library for the Python programming language and its numerical mathematics extension NumPy.
- [NumPy](http://www.numpy.org/) : General purpose array-processing package.
- [SimpleITK](http://www.simpleitk.org/) : Simplified interface to the Insight Toolkit for image registration and segmentation.
- [SciPy](https://www.scipy.org/) : A Python-based ecosystem of open-source software for mathematics, science, and engineering.
- [TensorFlow v1.x](https://www.tensorflow.org/) : TensorFlow helps the tensors flow.
- [xmltodict](https://github.com/martinblech/xmltodict) : Python module that makes working with XML feel like you are working with JSON.

## 2. Running RegNet
​     Run`RegNet.py`. Please note that current RegNet only works with pathology images.

### 2.1 Data
​     The 100,000 histological images of human colorectal cancer and healthy tissue dataset is applied to evaluate the proposed DTL-HS-RegNet. We chose a set of 11977 image patches of hematoxylin-eosin staining (HE) histological samples of human colorectal cancer which is accessible at http://dx.doi.org/10.5281/zenodo.1214456, containing three classes: adipose tissue and mucus (ADIMUC), stroma and muscle (STRMUS), and colorectal cancer epithelial tissue and stomach cancer epithelial tissue (TUMSTU). All images are 512 ×512 px at 0.5 μm / px.


### 2.2 Network
​    The proposed networks are given in Figure.
![image-20230627225124230](C:\Users\liu\AppData\Roaming\Typora\typora-user-images\image-20230627225124230.png)



