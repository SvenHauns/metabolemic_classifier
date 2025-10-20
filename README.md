# AE-TABPFN RF-TABPFN

TabPFN based classifier for processing and classification of highly-dimensional metabolemic data


The current script uses two methods to pre-compute features for TabPFN, both improving the classification accuracy.

The first one use a random forest model to pre-select important features based on the contribution on the change to the entropy. The second makes use of a small autoencoder to reduce the redundancy in the data before classification. 


### Installation

Follow the instruction given by https://github.com/PriorLabs/TabPFN to first install TabPFN and required dependencies.

Additionally install matplotlib if you want to make use of the plotting functions. 


### Running AE-TABPFN and RF-TABPFN

The main logic of the script is implemented in the main.py file. 

Here you find functions that can load the provided example data in the data folder and execute either ae-tabpfn or rf-tapfn

```
python main.py --dataset {dataset_to_be_used} --setting {ae or rf} --size {size}
```

### Evaluation

Hyperparameter optimization of tabpfn input values and baseline models can be found in optimization. 
