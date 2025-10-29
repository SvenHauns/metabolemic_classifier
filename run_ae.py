import pandas as pd
from tabpfn import TabPFNClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
import random
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from sklearn.metrics import precision_score, recall_score, f1_score

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

class LSTMBinaryClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMBinaryClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)  # Output is a single value for binary classification
        self.sigmoid = nn.Sigmoid()  # Optional, for binary output
        self.bn = nn.BatchNorm1d(input_size)
    
    def forward(self, x):

        lstm_out, (hn, cn) = self.lstm(self.bn(x))

        out = self.fc(torch.relu(lstm_out))  # Use the last hidden state

        return self.sigmoid(out)  # Sigmoid for probability output
        
        
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
    
    
class AutoEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AutoEncoder, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # Output is a single value for binary classification
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # Output is a single value for binary classification
        self.fc3 = nn.Linear(hidden_size, input_size)  # Output is a single value for binary classification
        
    def forward(self, x):

        out1 = torch.nn.functional.relu(self.fc1(x))  # Use the last hidden state
        #out = torch.nn.functional.relu(self.fc2(out))  # Use the last hidden state
        out = self.fc3(out1)  # Use the last hidden state
        
        return out, out1

def train_ae_conditional(model, model_classification, train_loader,tensor_x_test, tensor_y_test, criterion, criterion_classification, optimizer,optimizer_classifier, device, num_epochs=10, class_discount =  0.0001, batch_size = 32, test = True):

    max_acc = 0
    epoch_accuracy_full = 0
    
    accuracy_list = []
    train_acc_list = []
    reconstruction_loss = []
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = 1000)
    
    for epoch in range(num_epochs):
        model.train()
        model_classification.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device).float()
            
            optimizer.zero_grad() 
            optimizer_classifier.zero_grad() 

            outputs, compression = model(inputs)  
            loss = criterion(outputs, inputs) 
                        
            
            classification = model_classification(compression)  
            loss_classification = criterion_classification(classification, labels.unsqueeze(1)) 
            predicted = (classification > 0.5).float()
            correct_predictions += (predicted == labels.unsqueeze(1)).sum().item()
            total_predictions += labels.size(0)
            total_loss = loss + loss_classification * class_discount
            
            
            total_loss.backward()  
            optimizer.step()  
            optimizer_classifier.step()
            
            running_loss += total_loss.item()
            scheduler.step()

        epoch_accuracy = correct_predictions / total_predictions * 100
        epoch_loss = running_loss / len(train_loader)
        train_acc_list.append(epoch_accuracy)
        reconstruction_loss.append(loss.item())


        if test == True:
            model.eval()
        
            _, projection_test = model(tensor_x_test.to(device))     
            my_dataset = MyDataset(projection_test,tensor_y_test) # create your datset
            my_dataloader_test = DataLoader(my_dataset, batch_size=batch_size, shuffle = False) # create your dataloader
        
            accuracy = test(model_classification, my_dataloader_test, device)
            accuracy_list.append(accuracy)

    return accuracy, accuracy_list, train_acc_list, reconstruction_loss, model
    
    
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

    
def get_kfold_splits(X, y):


    loo = StratifiedKFold(n_splits=10, shuffle=True, random_state = 32)
    #loo.get_n_splits(X)
    
    splits = [(train_index, test_index) for train_index, test_index in loo.split(X, y)]
    
    importances_list = []
    

    return splits, importances_list

def train_predict_ae(X_train,y_train, X_test, hidden_size, class_discount, batch_size):

    tensor_x_train = torch.Tensor(np.array(X_train))
    tensor_y_train = torch.Tensor(y_train).float()
        
    tensor_x_test = torch.Tensor(np.array(X_test))


    autoencoder = AutoEncoder(input_size, hidden_size).to(device)
    lstm_classifier = LSTMBinaryClassifier(hidden_size, hidden_size, num_layers=1).to(device)
    optimizer_ae = torch.optim.Adam(autoencoder.parameters(), lr=0.001) 
    optimizer_classification = torch.optim.Adam(lstm_classifier.parameters(), lr=0.001) 
    criterion = nn.MSELoss()
    criterion_classification = nn.BCEWithLogitsLoss()
        
        
    my_dataset = MyDataset(tensor_x_train,tensor_y_train) 
    my_dataloader = DataLoader(my_dataset, batch_size=batch_size, shuffle = True) 
        
    accuracy, accuracy_list, train_acc_list, reconstruction_loss, model = train_ae_conditional(autoencoder, lstm_classifier, my_dataloader,tensor_x_test, None, criterion, criterion_classification, optimizer_ae,optimizer_classification, device, num_epochs=100, class_discount = class_discount, batch_size = batch_size, test = False)
        
    autoencoder.eval()
    with torch.no_grad():
        _,  tensor_x_train = autoencoder(tensor_x_train.to(device))
        _, tensor_x_test = autoencoder(tensor_x_test.to(device))

    # Initialize a classifier
    clf = TabPFNClassifier()
    clf.fit(tensor_x_train.cpu().detach(), tensor_y_train.cpu())
    predictions_prob = clf.predict_proba(tensor_x_test.cpu().detach())
    predictions = clf.predict(tensor_x_test.cpu().detach())
    
    return predictions, predictions_prob

def run_ae_tabpfn(X,y,hidden_size, class_discount, batch_size):
    
    splits, importances_list = get_kfold_splits(X, y)

    size_dict = {}
    prediction_dict = {}
    label_dict = {}
    proba_dict = {}

    f1_dict = {}
    precision_dict = {}
    recall_dict = {}
    roc = {}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size = np.shape(X)[1]


    average_prediction = []    
    prediction_list = []
    proba_list = []
    label_list = []
    for enum, (train_index, test_index) in enumerate(splits):


        X_train = X[train_index]
        y_train = y[train_index]
    
        X_test = X[test_index]
        y_test = y[test_index]

        tensor_x_train = torch.Tensor(np.array(X_train))
        tensor_y_train = torch.Tensor(y_train).float()
    
    
        tensor_x_test = torch.Tensor(np.array(X_test))
        tensor_y_test = torch.Tensor(y_test).float()
        
        
        autoencoder = AutoEncoder(input_size, hidden_size).to(device)
        lstm_classifier = LSTMBinaryClassifier(hidden_size, hidden_size, num_layers=1).to(device)
        optimizer_ae = torch.optim.Adam(autoencoder.parameters(), lr=0.001) 
        optimizer_classification = torch.optim.Adam(lstm_classifier.parameters(), lr=0.001) 
        criterion = nn.MSELoss()
        criterion_classification = nn.BCEWithLogitsLoss()
        
        
        my_dataset = MyDataset(tensor_x_train,tensor_y_train) 
        my_dataloader = DataLoader(my_dataset, batch_size=batch_size, shuffle = True) 
        
        accuracy, accuracy_list, train_acc_list, reconstruction_loss, model = train_ae_conditional(autoencoder, lstm_classifier, my_dataloader,tensor_x_test, tensor_y_test, criterion, criterion_classification, optimizer_ae,optimizer_classification, device, num_epochs=100, class_discount = class_discount, batch_size = batch_size)
        
        autoencoder.eval()
        with torch.no_grad():
            _,  tensor_x_train = autoencoder(tensor_x_train.to(device))
            _, tensor_x_test = autoencoder(tensor_x_test.to(device))

        # Initialize a classifier
        clf = TabPFNClassifier()
        clf.fit(tensor_x_train.cpu().detach(), tensor_y_train.cpu())
        predictions_prob = clf.predict_proba(tensor_x_test.cpu().detach())
        # Predict labels
        predictions = clf.predict(tensor_x_test.cpu().detach())
    
        score = accuracy_score(tensor_y_test.cpu(), predictions)
        average_prediction.append(score)
        prediction_list.extend(predictions)
        label_list.extend(tensor_y_test.cpu())
        proba_list.extend(predictions_prob)
        
        
    size_dict[hidden_size] = np.mean(average_prediction)
    
    
    prediction_dict[hidden_size] = prediction_list
    label_dict[hidden_size] = label_list
    proba_dict[hidden_size] = proba_list
    
    f1_dict[hidden_size] = f1_score(label_list, prediction_list, average="weighted")
    precision_dict[hidden_size] = precision_score(label_list, prediction_list, average="weighted")
    recall_dict[hidden_size] = recall_score(label_list, prediction_list, average="weighted")
    roc[hidden_size] = roc_auc_score(label_list, np.array(proba_list)[:,1])
    
    print("f1 loss")
    print(f1_dict)
    print("precision")
    print(precision_dict)
    print("recall")
    print(recall_dict)
    print("roc")
    print(roc)
    
    
    
