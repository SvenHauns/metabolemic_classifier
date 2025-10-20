import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNClassifier
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier, RandomForestClassifier
import torch
import random
# must be set before importing torch in some cases
import os
import argparse

os.environ["PYTHONHASHSEED"] = "32"  # stable hashing affects set/dict orders
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # or ":4096:8"

random.seed(32)
np.random.seed(32)
torch.manual_seed(32)
torch.cuda.manual_seed_all(32)   # not just manual_seed
torch.use_deterministic_algorithms(True)  # raises if a nondeterministic op is used
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_float32_matmul_precision("high")  # avoid "medium"/TF32 variability                      

def get_kfold_splits(X, y):


    loo = StratifiedKFold(n_splits=10, shuffle=True, random_state = 32)
    
    splits = [(train_index, test_index) for train_index, test_index in loo.split(X, y)]
    
    importances_list = []
    
    for (train_idx, test_idx) in splits: 
    
        rf = RandomForestClassifier(n_estimators = 1000)
        rf.fit(np.array(X)[train_idx], np.array(y)[train_idx])
        importances = rf.feature_importances_
        importances_list.append(importances)
    

    return splits, importances_list
    
    
def read_breast_cancer(data_path = "../data/"):

    
    dataset = pd.read_csv(data_path + "1995_0_data_set_7861791_ry39wp_model_construction.csv")
    y_list = list(dataset["Label"])
    y_list_or = y_list
    column_select = [c for c in dataset.columns if c not in ['Patient ID', 'Label']]
    datast2 = dataset[column_select]
    x_dataset_or = datast2.values



    dataset2 = pd.read_csv(data_path +"/1995_0_data_set_7861792_ry39hp_to_validate.csv")
    y_list = list(dataset2["Label"])
    column_select = [c for c in dataset.columns if c not in ['Patient ID', 'Label', 'Acquired Date']]
    dataset2 = dataset2[column_select]
    x_dataset = dataset2.values
    x_dataset_or = list(x_dataset_or)
    x_dataset_or.extend(x_dataset)
    y_list_or.extend(y_list)


    return np.array(x_dataset_or), np.array(y_list_or)

def load_dataset2(data_path = "../data/D-new2.csv"):

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

def load_dataset_psi_ms(data_path = "../data/PSI_MS_Raw_Urine_Frederico.csv"):

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




if __name__ == '__main__':
    cmdline_parser = argparse.ArgumentParser('main logic used for optimizing rf feature number')

    cmdline_parser.add_argument('-d', '--dataset',
                                default="",
                                help='dataset',
                                required = True,
                                type=str)
                                
    args, unknowns = cmdline_parser.parse_known_args()
    
    if args.dataset == "PSI-MS":
        X,y = load_dataset_psi_ms()
    elif args.dataset == "FI-TWIM-MS":
        X,y = load_dataset2()
    elif args.dataset == "breast-cancer":
        X,y = read_breast_cancer()
    else:
        X,y = load_custom_dataset(args.dataset)
        
    X = normalize(X)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    splits, importances_list = get_kfold_splits(X, y)
    average_prediction_test = []

    input_sizes = list(range(40,500,20))
    for i, (train_index, test_index) in enumerate(splits):


        X_train = X[train_index]
        X_test = X[test_index]
    
        y_train = y[train_index]
        y_test = y[test_index]

    
        prob_list_test, best_sizes_list, lables_test, prediction_test, performance_dict  = [], [], [], [], []
        splits_sub, importances_list_sub = get_kfold_splits(X_train, y_train)
    
        for input_size in input_sizes:
            prob_list, lables, average_prediction, prediction   = [], [], [], []


            for i_sub, (train_index_sub, test_index_sub) in enumerate(splits_sub):
    
                X_train_sub = X_train[train_index_sub]
                X_val = X_train[test_index_sub]
        
                y_train_sub = y_train[train_index_sub]
                y_val = y_train[test_index_sub]

                clf = TabPFNClassifier(ignore_pretraining_limits=True)
    
                improtance_indece = np.argsort(importances_list_sub[i_sub])[-input_size:]
                tensor_x_train = torch.Tensor(np.array(X_train_sub)[:, improtance_indece]) 

                clf.fit(tensor_x_train, y_train_sub)
    
                tensor_x_test = torch.Tensor(np.array(X_val)[:, improtance_indece]) 
                predictions = clf.predict(tensor_x_test)
                score = accuracy_score(y_val, predictions)
                prediction.append(predictions)

                average_prediction.append(score)
                lables.extend(y_val)
    
                prob = clf.predict_proba(tensor_x_test)
                prob_list.extend(prob)
                
            roc= roc_auc_score(lables, np.array(prob_list)[:, 1])
            performance_dict.append(roc)

        
        ###### choose best model #####        
        arg_max = np.argmax(performance_dict)
        best_sizes = input_sizes[arg_max]
        best_sizes_list.append(best_sizes)
        clf = TabPFNClassifier(ignore_pretraining_limits=True)
        improtance_indece = np.argsort(importances_list[i])[-best_sizes:]
        tensor_x_train = torch.Tensor(np.array(X_train)[:, improtance_indece]) 
        clf.fit(tensor_x_train, y_train)
        
        tensor_x_test = torch.Tensor(np.array(X_test)[:, improtance_indece]) 
        predictions = clf.predict(tensor_x_test)
        score = accuracy_score(y_test, predictions)
        prediction_test.append(predictions)
    
        lables_test.extend(y_test)
        prob = clf.predict_proba(tensor_x_test)
        prob_list_test.extend(prob)
        f1 = f1_score(lables_test, predictions, average = 'weighted')
        roc= roc_auc_score(lables_test, np.array(prob_list_test)[:, 1])
        average_prediction_test.append([score, roc, f1, best_sizes])

    print("##### RESULTS #####")
    print(average_prediction_test)

