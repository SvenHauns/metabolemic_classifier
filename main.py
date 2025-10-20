import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from run_ae import run_ae_tabpfn
from run_rf import run_rf_tabpfn
from sklearn.preprocessing import normalize
import argparse

def load_dataset_psi_ms(data_path = "./data/PSI_MS_Raw_Urine_Frederico.csv"):

    dataset = pd.read_csv(data_path)
    column_select = [c for c in dataset.columns if c not in ['Sample','']]
    dataset = dataset[column_select]
    dataset = dataset.iloc[:, 1:]

    labels = dataset.T.iloc[:,-1]
    dataset = dataset.T.iloc[:,:-1]

    x_dataset = dataset.values
    y_list = [0 if a == "Control" else 1 for a in list(labels.values)]


    x_dataset_or = []
    for dataline in x_dataset:
        dataline2 = [float(d.replace(",", ".")) for d in dataline]
        x_dataset_or.append(dataline2)
    
    
    return np.array(x_dataset_or), np.array(y_list)
    
    
def read_breast_cancer():

    
    dataset = pd.read_csv("../data/1995_0_data_set_7861791_ry39wp_model_construction.csv")
    y_list = list(dataset["Label"])
    y_list_or = y_list
    column_select = [c for c in dataset.columns if c not in ['Patient ID', 'Label']]
    datast2 = dataset[column_select]
    x_dataset_or = datast2.values



    dataset2 = pd.read_csv("../data/1995_0_data_set_7861792_ry39hp_to_validate.csv")
    y_list = list(dataset2["Label"])
    column_select = [c for c in dataset.columns if c not in ['Patient ID', 'Label', 'Acquired Date']]
    dataset2 = dataset2[column_select]
    x_dataset = dataset2.values
    x_dataset_or = list(x_dataset_or)
    x_dataset_or.extend(x_dataset)
    y_list_or.extend(y_list)


    return np.array(x_dataset_or), np.array(y_list_or)
    
def load_custom_dataset(datapath):

    dataset = pd.read_csv(data_path)
    X_data = []
    y_data = []
        
    for column in list(df.columns):
        y = df[column][1]
        X = np.array(df[column][2:])
        y_data.append(int(y))
        X_data.append(X)
        
    return np.array(y_data), np.array(X_data)
    

def load_dataset2(data_path = "./data/D-new2.csv"):

    df = pd.read_csv(data_path)
    X_data = []
    y_data = []
        
    for column in list(df.columns):
    
        if column == "Column1": continue
        y = df[column][1]
        
        if y == "Class:Healthy":
            y = 0
        elif y == "Class:Cancer":
            y = 1

        X = np.array(df[column][2:])
        y_data.append(y)
        X_data.append(X)
    
    return np.array(X_data), np.array(y_data)


if __name__ == '__main__':
    cmdline_parser = argparse.ArgumentParser('main logic used for training')

    cmdline_parser.add_argument('-d', '--dataset',
                                default="",
                                help='dataset',
                                required = True,
                                type=str)
    cmdline_parser.add_argument('-s', '--setting',
                                default="",
                                help='ae or rf',
                                required = True,
                                type=str)
    cmdline_parser.add_argument('-l', '--size',
                                default="",
                                help='latent or input size',
                                required = True,
                                type=int)
                                
    args, unknowns = cmdline_parser.parse_known_args()
    
    if args.dataset == "PSI-MS":
        X,y = load_dataset_psi_ms()
        lrdisc = 0.0001
        batch_size = 31
    elif args.dataset == "FI-TWIM-MS":
        X,y = load_dataset2()
        lrdisc = 0.0001
        batch_size = 32
    elif args.dataset == "breast-cancer":
        X,y = read_breast_cancer()
        lrdisc = 1.0
        batch_size = 32
    else:
        X,y = load_custom_dataset(args.dataset)
        batch_size = 32
    X = normalize(X)
    
    if args.setting == "rf":
        run_rf_tabpfn(X, y, args.size)
    elif args.setting == "ae":
        run_ae_tabpfn(X, y, args.size, class_discount = lrdisc, batch_size = batch_size)
    
    
    
