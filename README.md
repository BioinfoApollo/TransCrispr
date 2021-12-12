# TransCrispr_ONT
Transformer based hybrid model for predicting CRISPR/Cas9 single guide RNA cleavage efficiency


## OS dependencies
We use [tensorflow](https://www.tensorflow.org/) as the backend for training and testing.

[ViennaRNA](http://rna.tbi.univie.ac.at/) should be downloaded and installed in advance in order to capture the important biological features of sgRNA.

The required packages are:
+ python==3.6.9
+ tensorflow-gpu==2.2.0
+ Keras==2.3.0
+ numpy==1.19.5
+ pandas==1.1.5
+ viennarna==2.4.18
+ hyperopt==0.2.5

## Tested demo with testsets
1. Only sgRNA sequence composition
```
python PureNet.py
```
2. Combination of sequence feature and biological feature
```
python BioNet.py
```

In addition, you can change the test datasets in folder [testsets](https://github.com/BioinfoApollo/TransCrispr_ONT/tree/main/testsets) (sequence only) / [testsets_withbiofeat](https://github.com/BioinfoApollo/TransCrispr_ONT/tree/main/testsets_withbiofeat) (features of fusion) and the corresponding test model weights in folder [models](https://github.com/BioinfoApollo/TransCrispr_ONT/tree/main/models)

## Files and directories description
+ [models](https://github.com/BioinfoApollo/TransCrispr_ONT/tree/main/models) the weights for the TransCrispr_ONT model trained by different datasets
+ [testsets](https://github.com/BioinfoApollo/TransCrispr_ONT/tree/main/testsets) datasets for tests with mononucleotide sequence feature
+ [testsets_withbiofeat](https://github.com/BioinfoApollo/TransCrispr_ONT/tree/main/testsets_withbiofeat) datasets for tests with combination of sequence feature and biological feature
+ [BioNet.py](https://github.com/BioinfoApollo/TransCrispr_ONT/blob/main/BioNet.py) code for sgRNA on-target activity prediction with sequence and biological features
+ [ParamsDetail.py](https://github.com/BioinfoApollo/TransCrispr_ONT/blob/main/ParamsDetail.py) detailed results of the hyperparameter search for different datasets
+ [PureNet.py](https://github.com/BioinfoApollo/TransCrispr_ONT/blob/main/PureNet.py) code for sgRNA on-target activity prediction with sequence features
+ [Transformer.py](https://github.com/BioinfoApollo/TransCrispr_ONT/blob/main/Transformer.py) code for Transformer
+ [utils.py](https://github.com/BioinfoApollo/TransCrispr_ONT/blob/main/utils.py) code for data preprocessing and evaluation metrics setting
