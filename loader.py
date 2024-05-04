import random


class CustomDataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=True):
        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.dataset)
        for item in self.dataset:
            yield item["X"], item["y"]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
