# AE-TABPFN RF-TABPFN

TabPFN based classifier for processing and classification of highly-dimensional metabolemic data


The current script uses two methods to pre-compute features for TabPFN, both improving the classification accuracy.

The first one use a random forest model to pre-select important features based on the contribution on the change to the entropy. The second makes use of a small autoencoder to reduce the redundancy in the data before classification. 


### Installation

Follow the instruction given by https://github.com/PriorLabs/TabPFN to first install TabPFN and required dependencies.
Additionally install matplotlib if you want to make use of the plotting functions. 

Alternatively we provide a env.yml file that can be installed using:

```
conda env create -f env.yml
```


### Running AE-TABPFN and RF-TABPFN

The main logic of the script is implemented in the main.py file. 

Here you find functions that can load the provided example data in the data folder and execute either ae-tabpfn or rf-tapfn

```
python main.py --dataset {dataset_to_be_used} --setting {ae or rf} --size {size}
```

To run a prediction you can make use of: 

```
python main.py --test_dataset {./data/toy_predict.csv} --dataset {./data/toy_train.csv} --predict True --setting {ae or rf} --size {size}
```

### Evaluation

Hyperparameter optimization of Tabpfn input values and baseline models can be found in optimization. 

### Data sources

The following data was used be the study and is provided by the corresponding publications:

1. FI-TWIM-MS Serum: 61 PC patients + 42 controls(103 samples, 237 features) DOI:  10.1021/acs.analchem.8b04259
2. PSI-MS Urine: 40 PC patients + 40 controls(80 samples, 784 features) DOI: 10.1021/acs.analchem.1c00943/DOI: 10.1021/acs.analchem.1c04004
3. PSI-MS Biopsies: 204 malignant + 488 benign(692 samples, 494 features) DOI: https://doi.org/10.1038/s43856-025-00930-7
