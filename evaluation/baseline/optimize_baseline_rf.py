import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier, RandomForestClassifier
import torch
import random
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from models.models import Conv1DClassifier, AlternatingLinearDropoutMLP, DeepMLP
import argparse
os.environ["PYTHONHASHSEED"] = "32" 
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  

random.seed(32)
np.random.seed(32)
torch.manual_seed(32)
torch.cuda.manual_seed_all(32)   
torch.use_deterministic_algorithms(True)  
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_float32_matmul_precision("high")    


class MyDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data

        self.targets = targets

        
    def __getitem__(self, index):

        x = self.data[index]
        y = self.targets[index]

        
        return x, y
    
    def __len__(self):
        return len(self.data)
        

    
def return_model(model_class, input_size = None):
    
    if model_class == "cnn":
        model = Conv1DClassifier(in_channels=1,num_classes=1,conv_channels=[32, 64, 128],
                kernel_sizes=[2, 2, 2], strides=[1, 1, 1], pool_every=1, pool_kernel_size=2,activation="relu",
                use_batchnorm=True, dropout_p=0.1, global_pool="avg", head_hidden_dims=[64])
    
    elif model_class == "mlp":
        model = DeepMLP(input_dim=input_size, hidden_dims=[256, 256, 128],output_dim=1,
            activation="gelu", dropout_p=0.2, use_batchnorm=True, last_activation=None)
    
    elif model_class == "mlp-surv":
        model = AlternatingLinearDropoutMLP(layer_dims=[input_size, 512, 256, 1],
            dropout_p=[0.3, 0.3, 0.0], activation="relu", last_activation=None, use_batchnorm=False)
    

    return model 


        
def test(model, test_loader, device):
    model.eval()  # Set model to evaluation mode
    correct_predictions = 0
    total_predictions = 0
    predicted_list = []
    labels_list = []
    with torch.no_grad():  # Disable gradient calculation
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device).float()
            labels_list.extend(labels)
            outputs = model(inputs)
            predicted = (outputs > 0.5).float()
            correct_predictions += (predicted == labels.unsqueeze(1)).sum().item()
            total_predictions += labels.size(0)
            predicted_list.extend(predicted)
    accuracy = correct_predictions / total_predictions * 100
    #print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy, predicted_list, labels_list
    
    

def train(model_classification, train_loader,tensor_x_test, tensor_y_test, criterion_classification,optimizer_classifier, device, num_epochs=10):

    max_acc = 0
    epoch_accuracy_full = 0
    
    accuracy_list = []
    train_acc_list = []
    reconstruction_loss = []
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_classifier, T_0 = 1000)
    
    for epoch in range(num_epochs):
        model_classification.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        total_loss = 0
        
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device).float()
            optimizer_classifier.zero_grad() 

            
            classification = model_classification(inputs.to(device))  
            loss_classification = criterion_classification(classification, labels.unsqueeze(1)) 
            
            
            predicted = (classification > 0.5).float()
            correct_predictions += (predicted == labels.unsqueeze(1)).sum().item()
            total_predictions += labels.size(0)
            total_loss += loss_classification.item()
            #total_loss = loss
            
            
            loss_classification.backward()  
            optimizer_classifier.step()
            

            scheduler.step()

        epoch_accuracy = correct_predictions / total_predictions * 100
        epoch_loss = running_loss / len(train_loader)
        train_acc_list.append(epoch_accuracy)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.9f}")
        
        model_classification.eval()
        
        my_dataset = MyDataset(tensor_x_test,tensor_y_test) # create your datset
        my_dataloader_test = DataLoader(my_dataset, batch_size=32, shuffle = False) # create your dataloader
        
        prob = model_classification(tensor_x_test.to(device))
        accuracy, predicted_list, labels_list = test(model_classification, my_dataloader_test, device)
        accuracy_list.append(accuracy)
        

    return accuracy, accuracy_list, train_acc_list, reconstruction_loss, model_classification, prob




def get_kfold_splits(X, y):


    loo = StratifiedKFold(n_splits=10, shuffle=True, random_state=32)
    #loo.get_n_splits(X)
    
    splits = [(train_index, test_index) for train_index, test_index in loo.split(X, y)]
    
    importances_list = []
    
    for (train_idx, test_idx) in splits: 
    
        rf = RandomForestClassifier(n_estimators = 1000)
        rf.fit(np.array(X)[train_idx], np.array(y)[train_idx])
        importances = rf.feature_importances_
        importances_list.append(importances)
    

    return splits, importances_list
    
    
def read_breast_cancer():

    
    dataset = pd.read_csv("../../data/1995_0_data_set_7861791_ry39wp_model_construction.csv")
    y_list = list(dataset["Label"])
    y_list_or = y_list
    column_select = [c for c in dataset.columns if c not in ['Patient ID', 'Label']]
    datast2 = dataset[column_select]
    x_dataset_or = datast2.values



    dataset2 = pd.read_csv("../../data/1995_0_data_set_7861792_ry39hp_to_validate.csv")
    y_list = list(dataset2["Label"])
    column_select = [c for c in dataset.columns if c not in ['Patient ID', 'Label', 'Acquired Date']]
    dataset2 = dataset2[column_select]
    x_dataset = dataset2.values
    x_dataset_or = list(x_dataset_or)
    x_dataset_or.extend(x_dataset)
    y_list_or.extend(y_list)


    return np.array(x_dataset_or), np.array(y_list_or)

def load_dataset2(data_path = "../../data/D-new2.csv"):

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

def load_dataset_psi_ms(data_path = "../../data/PSI_MS_Raw_Urine_Frederico.csv"):

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
    cmdline_parser = argparse.ArgumentParser('main logic used for training')

    cmdline_parser.add_argument('-d', '--dataset',
                                default="",
                                help='dataset',
                                required = True,
                                type=str)
    cmdline_parser.add_argument('-m', '--model_type',
                                default="cnn",
                                help='type of baseline model',
                                required = True,
                                type=str)
                                
                                
                                
    args, unknowns = cmdline_parser.parse_known_args()
    
    if args.dataset == "PSI-MS":
        X,y = load_dataset_psi_ms()
        batch_size = 31
    elif args.dataset == "FI-TWIM-MS":
        X,y = load_dataset2()
        batch_size = 32
    elif args.dataset == "breast-cancer":
        X,y = read_breast_cancer()
        batch_size = 32
    else:
        X,y = load_custom_dataset(args.dataset)
        batch_size = 32
        
    X = normalize(X)

    
    assert args.model_type in ["cnn", "mlp", "mlp-surv"]
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



    splits, importances_list = get_kfold_splits(X, y)
    average_prediction_test = [] 
    input_sizes = list(range(40,500,20))
    
    
    for i, (train_index, test_index) in enumerate(splits):


        X_train = X[train_index]
        X_test = X[test_index]
    
        y_train = y[train_index]
        y_test = y[test_index]

        prob_list_test, lables_test, best_sizes_list, prediction_test = [], [], [], []

    
        splits_sub, importances_list_sub = get_kfold_splits(X_train, y_train)
        performance_dict = []
        for input_size in input_sizes:
            prob_list, lables, average_prediction,prediction  = [], [],[], []

            for i_sub, (train_index_sub, test_index_sub) in enumerate(splits_sub):
    
                X_train_sub = X_train[train_index_sub]
                X_val = X_train[test_index_sub]
        
                y_train_sub = y_train[train_index_sub]
                y_val = y_train[test_index_sub]
            
                tensor_x_train = torch.Tensor(np.array(X_train_sub)) 
                tensor_y_train = torch.Tensor(y_train_sub).float()
                tensor_x_test = torch.Tensor(np.array(X_val)) 
                tensor_y_test = torch.Tensor(y_val).float()
            
            
                model = return_model(args.model_type, input_size = input_size)

                improtance_indece = np.argsort(importances_list_sub[i_sub])[-input_size:]
                tensor_x_train = torch.Tensor(np.array(X_train_sub)[:, improtance_indece])
                optimizer_classifier = torch.optim.Adam(model.parameters(), lr=0.0001) 
                criterion_classification = nn.BCEWithLogitsLoss()
                tensor_x_test = torch.Tensor(np.array(X_val)[:, improtance_indece])
            
                class_sample_count = np.bincount(y_train_sub)  
                class_weights = 1. / class_sample_count  
                sample_weights = np.array([class_weights[int(label)] for label in y_train_sub])
                sample_weights = torch.Tensor(sample_weights)
                sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weights, len(sample_weights))
            
                my_dataset = MyDataset(tensor_x_train,tensor_y_train) 
                my_dataloader = DataLoader(my_dataset, batch_size=batch_size, shuffle = True) 
            
                accuracy, accuracy_list, train_acc_list, reconstruction_loss, model, prob = train(model, my_dataloader,tensor_x_test, tensor_y_test, criterion_classification,optimizer_classifier, device, num_epochs=100)


                tensor_x_test = torch.Tensor(np.array(X_val)[:, improtance_indece]) # transform to torch tensor
                predictions = model(tensor_x_test.to(device))

                average_prediction.append(accuracy)
                lables.extend(y_val)
                prob_list.extend(prob.cpu().detach().numpy())


            roc= roc_auc_score(lables, np.array(prob_list))
            performance_dict.append(roc)

        
        
        arg_max = np.argmax(performance_dict)

        ###### choose best model #####
        best_sizes = input_sizes[arg_max]
        best_sizes_list.append(best_sizes)
    
        improtance_indece = np.argsort(importances_list[i])[-best_sizes:]
        tensor_x_train = torch.Tensor(np.array(X_train)[:, improtance_indece])
        tensor_x_test = torch.Tensor(np.array(X_test)[:, improtance_indece])
        tensor_y_train = torch.Tensor(y_train).float()
        tensor_y_test = torch.Tensor(y_test).float()
    
    
        model = return_model(args.model_type, input_size = best_sizes)
                
        optimizer_classification = torch.optim.Adam(model.parameters(), lr=0.0001) 
        criterion_classification = nn.BCEWithLogitsLoss()
        
        
        my_dataset = MyDataset(tensor_x_train,tensor_y_train) 
        my_dataloader = DataLoader(my_dataset, batch_size=batch_size, shuffle = False, sampler = sampler) 
    
        accuracy, accuracy_list, train_acc_list, reconstruction_loss, model, prob = train(model, my_dataloader,tensor_x_test, tensor_y_test, criterion_classification, optimizer_classification, device, num_epochs=100)

    
        lables_test.extend(y_test)
        prob_list_test.extend(prob.cpu().detach().numpy())
        f1 = f1_score(lables_test, [1 if a >= 0.5 else 0 for a in prob_list_test], average = 'weighted')
        roc= roc_auc_score(lables_test, np.array(prob_list_test))
        average_prediction_test.append([accuracy, roc, f1, best_sizes])


    print("##### RESULTS #####")
    print(average_prediction_test)


