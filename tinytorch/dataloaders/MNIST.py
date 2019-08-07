import torch
import torchvision
from torchvision import datasets, transforms

__add__ = ['mnist_dataloader']



# This creates the transforms instance
transforms = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5], std=[0.5])])

def mnist_dataloader(batch_size=16):
    r"""
        The function return a tuple of train_dataloader and test_date_loader
        Args:
            batch_size: (optional) The specifies the batch size
        Example:
            train_data = mnist_dataloader(batch_size=32)[0]
            test_data = mnist_dataloader(batch_size=32)[1]
            train_data, test_data = mnist_dataloader(batch_size=32)
    """

    train_data = datasets.MNIST(root='./',transform=transforms, download=True)
    test_data = datasets.MNIST(root='./', transform=transforms, train=False, download=True)
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)
    return train_data_loader, test_data_loader