# Network-Compression

Network Compression is the technique of optimising a deep learning model by reducing the number of parameters. In our [paper](https://arxiv.org/pdf/1808.03833.pdf), we do this by computing the l2 norm of orcale ranking based on first order Taylor expansion for each layer followed by the removal of parameters which are less than a given threshold and retraining the network.   

This repository contains our TensorFlow implementation of Network-Compression for [AdapNet++](https://github.com/DeepSceneSeg/AdapNet-pp), which allows you to compress your own model on any dataset and evaluate results in terms of the mean IoU metric. 

If you find the code useful for your research, please consider citing our paper:
```
@article{valada18SSMA,
  author = {Valada, Abhinav and Mohan, Rohit and Burgard, Wolfram},
  title = {Self-Supervised Model Adaptation for Multimodal Semantic Segmentation},
  journal = {arXiv preprint arXiv:1808.03833},
  month = {August},
  year = {2018},
}
```

## Contacts
* [Abhinav Valada](http://www2.informatik.uni-freiburg.de/~valada/)
* [Rohit Mohan](https://github.com/mohan1914)

## System Requirements

#### Programming Language
```
Python 2.7
```

#### Python Packages
```
tensorflow-gpu 1.4.0
```

