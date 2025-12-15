
import numpy as np 
import torch 
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle


class SimpleDataset(Dataset) :
    def __init__(self, data_set, y, device) :
        self.data_set = [torch.tensor(data, dtype=torch.float32).to(device) for data in data_set]
        self.y = torch.tensor(y).squeeze().long().to(device)
        self.device = device
        
    def __len__(self) :
        return len(self.data_set[0])
    
    def __getitem__(self, i) :
        xs = [data[i] for data in self.data_set]
        y = self.y[i]
        return (xs, y)
    

class NumpyDataset(Dataset):
    """
    Numpy dataset class, converts numpy data to torch dataset.

    This can be used to convert any numpy data into a CompFS
    dataset.

    Args:
        X_data: numpy array of X_data
        y_data: Numpy array of y_data
        classification: Bool tells the class whether to save y values as longs or floats
    """

    def __init__(self, X_data, y_data, classification=True):

        self.x_bar = torch.tensor(np.mean(X_data, axis=0)).float()
        self.num_data = X_data.shape[0]
        self.data = []
        for x_sample, y_sample in zip(X_data, y_data):
            x = torch.from_numpy(x_sample).float()
            if classification:
                y = torch.tensor(y_sample).long()
            else:
                y = torch.tensor(y_sample).float()
            self.data.append((x, y))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.num_data

    def get_x_bar(self):
        try:
            return self.x_bar
        except AttributeError:
            x_bar = 0
            for sample in self.data:
                x_bar += sample[0]
            self.x_bar = x_bar / self.num_data
            return self.x_bar
    

def train_val_test_split(data:tuple, test_size=0.2, val_size=0.2, seed=0, scaling=False, ckpt=None) : 
    # data:Tuple(views, y), views:np.array(views[v]: v-th view np.array)

    n= len(data[1])

    X_set = data[0]  # Tuple (v1, v2, ... ,v)
    y = data[1]

    # Generate the indices for the train/test 
    # 
    # 
    # split 64/16/20 train/ val/ test
    train_indices, test_indices = train_test_split(np.arange(n), test_size=test_size, random_state=seed)
    train_indices, val_indices = train_test_split(train_indices, test_size=val_size, random_state=seed)
    
    # Use these indices to split each view
    
    tr_X_set = [X[train_indices] for X in X_set]
    va_X_set = [X[val_indices] for X in X_set]
    te_X_set = [X[test_indices] for X in X_set]

    tr_y, va_y, te_y = y[train_indices], y[val_indices], y[test_indices]

    # Scaling
    if scaling:
        print("Scaling the data")
        # Initialize a list to hold the scalers for each view
        if ckpt is None:
            scalers = []
        else:
            with open (ckpt, 'rb') as f:
                scalers = pickle.load(f)
            print("scaler loaded")
        
        # Scale the training data and save the scalers
        scaled_tr_X_set = []
        for i, view_data in enumerate(tr_X_set):
            if ckpt is None: 
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(view_data)
                scaled_tr_X_set.append(scaled_data)
                scalers.append(scaler)
            else:
                scaler = scalers[i]
                scaled_data = scaler.transform(view_data)
                scaled_tr_X_set.append(scaled_data)
                
        # Use the saved scalers to transform validation and test sets
        scaled_va_X_set = [scalers[i].transform(va_X_set[i]) for i in range(len(va_X_set))]
        scaled_te_X_set = [scalers[i].transform(te_X_set[i]) for i in range(len(te_X_set))]
    
        return scaled_tr_X_set, tr_y, scaled_va_X_set, va_y, scaled_te_X_set, te_y, scalers

    return tr_X_set, tr_y, va_X_set, va_y, te_X_set, te_y
        

def single_view_data_split(data):
    
    n = len(data[1])
    test_size, val_size = 0.2, 0.2
    train_indices, test_indices = train_test_split(np.arange(n), test_size=test_size, random_state=0)
    train_indices, val_indices = train_test_split(train_indices, test_size=val_size, random_state=0)

    # Use these indices to split each view
    X_train = data[0][train_indices]
    X_val = data[0][val_indices]
    X_test = data[0][test_indices]
    y_train, y_val, y_test = data[1][train_indices], data[1][val_indices], data[1][test_indices]
    
    #scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler
