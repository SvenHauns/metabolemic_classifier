import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from run_ae import run_ae_tabpfn
from run_rf import run_rf_tabpfn
from sklearn.preprocessing import normalize

def load_dataset_psi_ms(data_path = "../data/PSI_MS_Raw_Urine_Frederico.csv"):

    dataset = pd.read_csv(data_path)
    column_select = [c for c in dataset.columns if c not in ['Sample','']]
    dataset = dataset[column_select]
    dataset = dataset.iloc[:, 1:]

    labels = dataset.T.iloc[:,-1]
    dataset = dataset.T.iloc[:,:-1]

    x_dataset = dataset.values
    y_list = [0 if a == "Control" else 1 for a in list(labels.values)]
    y_list_or = y_list

    x_dataset_or = []
    for dataline in x_dataset:
        dataline2 = [float(d.replace(",", ".")) for d in dataline]
        x_dataset_or.append(dataline2)
    
    
    return np.array(x_dataset_or), np.array(labels)

def load_dataset2(data_path = "../data/D-new2.csv"):

    df = pd.read_csv()
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
    
    if args.dataset == "psi-ms":
        X,y = load_dataset_psi_ms()
    elif: args.dataset == "psi":
        X,y = load_dataset2()
        
    X = normalize(X)
    
    if args.setting == "rf":
        run_rf_tabpfn(X, y, args.size)
    elif: args.setting == "ae":
        run_ae_tabpfn(X, y, args.size)
    
    
    
