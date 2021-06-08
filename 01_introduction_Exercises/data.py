from torchvision import datasets, transforms

def mnist():
    """ Downloads and loads the MNIST training and test data """
    
    # Define a transform to normalize the data
    transform   = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize((0.5,), (0.5,)),])
    
    # Download and load the training and test dat
    train    = datasets.MNIST('~/.pytorch/MNIST_data/', download=True,
                              train=True, transform=transform)
    test     = datasets.MNIST('~/.pytorch/MNIST_data/', download=True,
                              train=False, transform=transform)
    
    return train, test
