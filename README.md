# TransCrispr_ONT
Transformer based hybrid model for predicting CRISPR/Cas9 single guide RNA cleavage efficiency


## OS Dependencies
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
`python PureNet.py`
2. Combination of sequence feature and biological feature
`python BioNet.py`

In addition, you can change the test data sets in folder [testsets](https://github.com/BioinfoApollo/TransCrispr_ONT/tree/main/testsets) and the corresponding test model weights in folder [models](https://github.com/BioinfoApollo/TransCrispr_ONT/tree/main/models)
