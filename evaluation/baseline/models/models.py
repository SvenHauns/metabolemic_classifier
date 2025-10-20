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
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
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
        
def get_activation(name = "relu") -> nn.Module:

    if callable(name):
        return name()
        
    name = str(name).lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU(inplace=True)
    if name == "mish":
        return nn.Mish()
    if name == "tanh":
        return nn.Tanh()
    if name == "leaky_relu":
        return nn.LeakyReLU(0.01, inplace=True)
    if name == "elu":
        return nn.ELU(inplace=True)
    if name in {"none", "identity", "id"}:
        return nn.Identity()
    raise ValueError(f"Unknown activation: {name}")

class DeepMLP(nn.Module):

    def __init__(
    self,
    input_dim: int,
    hidden_dims,
    output_dim: int,
    activation= "relu",
    dropout_p: float = 0.0,
    use_batchnorm: bool = False,
    last_activation= None,
    ) -> None:
    
        super().__init__()
        layers: List[nn.Module] = []
        in_f = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_f, h, bias=not use_batchnorm))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(get_activation(activation))
            if dropout_p > 0:
                layers.append(nn.Dropout(p=dropout_p))
            in_f = h
        layers.append(nn.Linear(in_f, output_dim))
        if last_activation is not None:
            layers.append(get_activation(last_activation))
        self.net = nn.Sequential(*layers)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))
        

class AlternatingLinearDropoutMLP(nn.Module):


    def __init__(
    self,
    layer_dims, # e.g. [in, h1, h2, ..., out]
    dropout_p = 0.5,
    activation= "relu",
    last_activation = None,
    use_batchnorm = False,
    ) -> None:
        super().__init__()
        assert len(layer_dims) >= 2, "layer_dims must include input and output"
        n_blocks = len(layer_dims) - 1


        if isinstance(dropout_p, (int, float)):
            dropouts = [float(dropout_p)] * (n_blocks - 1) # usually no dropout after final layer
        else:
            dropouts = list(dropout_p)
        assert len(dropouts) in {n_blocks - 1, n_blocks}, (
        "dropout_p must have len n_blocks-1 (no dropout after last) or n_blocks"
        )

        layers: List[nn.Module] = []
        for i in range(n_blocks):
            in_f, out_f = layer_dims[i], layer_dims[i + 1]
            is_last = (i == n_blocks - 1)


            layers.append(nn.Linear(in_f, out_f, bias=not use_batchnorm))
            if use_batchnorm and not is_last:
                layers.append(nn.BatchNorm1d(out_f))


            if not is_last:
                layers.append(get_activation(activation))
            elif last_activation is not None:
                layers.append(get_activation(last_activation))


            p = dropouts[i] if i < len(dropouts) else 0.0
            if p > 0:
                layers.append(nn.Dropout(p=p))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
        
        
class Conv1DClassifier(nn.Module):

    def __init__(
    self,
    in_channels,
    num_classes,
    conv_channels,
    kernel_sizes,
    strides= 1,
    paddings= None,
    dilations= 1,
    activation= "relu",
    use_batchnorm = True,
    dropout_p = 0.0,
    pool_every= 1,
    pool_kernel_size = 2,
    global_pool = "avg",
    head_hidden_dims = None
    ):
        super().__init__()
        assert len(conv_channels) == len(kernel_sizes), "conv_channels and kernel_sizes must match"


        if isinstance(strides, int):
            strides = [strides] * len(conv_channels)
        if isinstance(dilations, int):
            dilations = [dilations] * len(conv_channels)
        if paddings is None:
            paddings = [((k - 1) // 2) * d for k, d in zip(kernel_sizes, dilations)]


        blocks: List[nn.Module] = []
        c_in = in_channels
        act_name = activation


        for i, (c_out, k, s, p, d) in enumerate(zip(conv_channels, kernel_sizes, strides, paddings, dilations)):
            blocks.append(nn.Conv1d(c_in, c_out, kernel_size=k, stride=s, padding=p, dilation=d, bias=not use_batchnorm))
            if use_batchnorm:
                blocks.append(nn.BatchNorm1d(c_out))
            blocks.append(get_activation(act_name))
            if dropout_p and dropout_p > 0:
                blocks.append(nn.Dropout(p=dropout_p))
            if pool_every is not None and pool_every > 0 and ((i + 1) % pool_every == 0):
                blocks.append(nn.MaxPool1d(kernel_size=pool_kernel_size))
            c_in = c_out


        self.features = nn.Sequential(*blocks)


        if global_pool is None:
            self.pool = nn.Identity()
            self.flatten = nn.Flatten()
            head_in = c_in 
            self._head_needs_adaptive = True
        else:
            self.flatten = nn.Flatten()
        if global_pool.lower() == "avg":
            self.pool = nn.AdaptiveAvgPool1d(1)
        elif global_pool.lower() == "max":
            self.pool = nn.AdaptiveMaxPool1d(1)
        else:
            raise ValueError("global_pool must be one of {'avg','max',None}")
        
        head_in = c_in
        self._head_needs_adaptive = False


        head_layers: List[nn.Module] = []
        last = head_in
        if head_hidden_dims:
        # If we didn't pool to length 1, adaptively do so now to have a fixed dimension
            if self._head_needs_adaptive:
                head_layers.append(nn.AdaptiveAvgPool1d(1))
            for h in head_hidden_dims:
                head_layers.extend([
                nn.Flatten(),
                nn.Linear(last, h, bias=True),
                get_activation(act_name),
                nn.Dropout(p=dropout_p) if dropout_p > 0 else nn.Identity(),
                ])
                last = h

        head_layers.extend([
            nn.Flatten(),
            nn.Linear(last, num_classes, bias=True),
            ])
        self.head = nn.Sequential(*head_layers)




    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = torch.unsqueeze(x, 1)
        x = self.features(x)
        x = self.pool(x)
        x = self.head(x)
        return torch.sigmoid(x)
        
class AutoEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AutoEncoder, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  
        self.fc3 = nn.Linear(hidden_size, input_size)  
        
    def forward(self, x):

        out1 = torch.nn.functional.relu(self.fc1(x))  
        out = self.fc3(out1)  
        
        return out, out1


