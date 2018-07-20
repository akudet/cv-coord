import numpy as np
import torch.utils.data as data


def get_loader(batch_size, n_pairs):
    dataset = CoordDataset(n_pairs)
    return data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)


class CoordDataset(data.Dataset):

    def __init__(self, n_pairs):
        self.x = np.random.random((n_pairs, 4))
        self.y = ((self.x[:, :2] - self.x[:, 2:]) < 0).astype(np.float64)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.shape[0]
