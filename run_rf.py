import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas as pd
from tabpfn import TabPFNClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.model_selection import LeaveOneOut
import random
from sklearn.ensemble import RandomForestClassifier
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


np.random.seed(32)
random.seed(32)
torch.manual_seed(32)   
torch.cuda.manual_seed(32)        
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False                        


def get_loo_splits(X, y):

    loo = LeaveOneOut()
    loo.get_n_splits(X)
    
    splits = [(train_index, test_index) for train_index, test_index in loo.split(X)]
    
    importances_list = []
    
    for (train_idx, test_idx) in splits: 
    
        rf = RandomForestClassifier(n_estimators = 1000)
        rf.fit(np.array(X)[train_idx], np.array(y)[train_idx])
        importances = rf.feature_importances_
        importances_list.append(importances)
    

    return splits, importances_list


def run_rf_tabpfn(X, y, input_size):



    splits, importances_list = get_loo_splits(X, y)

    loo = LeaveOneOut()
    loo.get_n_splits(X)

    size_dict = {}
    precision_dict = {}
    recall_dict = {}
    f1_dict = {}
    roc = {}
    
    average_prediction = []    
    label_list = []
    prediction_list = []
    proba_list = []
    
    for enum, (train_index, test_index) in enumerate(splits):

        print(f"  Train: index={train_index}")
        print(f"  Test:  index={test_index}")

    
        X_train = X[train_index]
        y_train = y[train_index]
    
        X_test = X[test_index]
        y_test = y[test_index]

        improtance_indece = np.argsort(importances_list[enum])[-input_size:]
            
        tensor_x_train = torch.Tensor(np.array(X_train)[:, improtance_indece]) # transform to torch tensor
        tensor_y_train = torch.Tensor(y_train).float()
    
    
        tensor_x_test = torch.Tensor(np.array(X_test)[:, improtance_indece]) # transform to torch tensor
        tensor_y_test = torch.Tensor(y_test).float()


        # Initialize a classifier
        clf = TabPFNClassifier()
        clf.fit(tensor_x_train, tensor_y_train)

        # Predict labels
        predictions = clf.predict(tensor_x_test)
        prob = clf.predict_proba(tensor_x_test)
        print("Accuracy", accuracy_score(tensor_y_test, predictions))
        
    
        score = accuracy_score(tensor_y_test, predictions)
        

        average_prediction.append(score)
        prediction_list.extend(predictions)
        label_list.extend(tensor_y_test.cpu())
        proba_list.append(prob[0])
        
        
    size_dict[input_size] = np.mean(average_prediction)
    
    f1_dict[input_size] = f1_score(label_list, prediction_list, average="weighted")
    precision_dict[input_size] = precision_score(label_list, prediction_list, average="weighted")
    recall_dict[input_size] = recall_score(label_list, prediction_list, average="weighted")
    roc[input_size] = roc_auc_score(label_list, np.array(proba_list)[:,1])
    
    print("F1-Loss")
    print(f1_dict)
    
    print("precision")
    print(precision)
    
    print("recall")
    print(recall)
    
    print("ROC")
    print(roc)
    
