import torch
from torch.utils.data import Dataset

# -----------------------------
# PyTorch Dataset Wrapper
# -----------------------------
class SimpleDataset(Dataset):
    """
    X_set = [view1, view2, ..., viewN]   (numpy arrays)
    y     = labels (numpy)
    """
    
    def __init__(self, data_set, y, device):
        self.data_set = [torch.tensor(v, dtype=torch.float32).to(device) for v in data_set]
        self.y = torch.tensor(y).long().to(device)
        self.device = device

    def __len__(self):
        return len(self.data_set[0])

    def __getitem__(self, i):
        xs = [v[i] for v in self.data_set]
        return xs, self.y[i]


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