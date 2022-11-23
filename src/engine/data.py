

class Dataset:

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __getitem__(self, idx):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()


class TableDataset(Dataset):

    def __init__(self, x, y, **kwargs):
        super().__init__(**kwargs)
        self.data_x = x
        self.data_y = y

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, ind):
        return self.data_x[:, ind], self.data_y[:, ind]


class DataLoader:

    def __init__(self, dataset, batch_size, shuffle):
        self.dataset = dataset
        self.bs = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        pass

    def __next__(self):
        pass
