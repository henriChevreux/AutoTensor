import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

class FashionMNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir="./data", batch_size=128, num_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Data augmentation and normalization for training - keep images grayscale
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))  # FashionMNIST normalization values
        ])
        
        # Just normalization for validation/test
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))  # FashionMNIST normalization values
        ])
        
        # Update dimensions to reflect grayscale images
        self.dims = (1, 28, 28)  # 1 channel, 28x28 pixels
        self.num_classes = 10
        self.class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    def prepare_data(self):
        # download
        datasets.FashionMNIST(self.data_dir, train=True, download=True)
        datasets.FashionMNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Assign train/val datasets
        if stage == 'fit' or stage is None:
            cifar_full = datasets.FashionMNIST(self.data_dir, train=True, transform=self.transform_train)
            self.cifar_train, self.cifar_val = random_split(cifar_full, [0.8, 0.2])

        # Assign test dataset
        if stage == 'test' or stage is None:
            self.cifar_test = datasets.FashionMNIST(self.data_dir, train=False, transform=self.transform_test)

    def train_dataloader(self):
        return DataLoader(self.cifar_train, batch_size=self.batch_size, shuffle=True, 
                          num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.cifar_val, batch_size=self.batch_size, 
                          num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.cifar_test, batch_size=self.batch_size, 
                          num_workers=self.num_workers, pin_memory=True) 