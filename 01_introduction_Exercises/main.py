import sys
import argparse

import torch
from torch import nn
from torch import optim

from data import mnist
from model import Classifier

import matplotlib.pyplot as plt

class TrainOREvaluate(object):
    """
    A class used to train or evaluate a MNIST neural network  

    ...

    Methods
    -------
    train()
        Trains the network using MNIST training data
    evaluate()
        Evaluates the network using MNIST test data
    """
    
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>"
        )
        parser.add_argument("command",
                            help="Subcommand to run (either 'train' or 'evaluate')")
        args = parser.parse_args(sys.argv[1:2])
        
        # Check that this is a valid command to run
        if not hasattr(self, args.command):
            print('Unrecognized command')
            
            parser.print_help()
            exit(1)
            
        # Use dispatch pattern to invoke method with same name
        getattr(self, args.command)()
    
    def train(self):
        """ Trains the neural network using MNIST training data """
        
        print("Training a neural network using MNIST training data")
        parser = argparse.ArgumentParser(description='Training arguments')
        # Help: Should this listed in the help text as well?
        parser.add_argument('-lr', '--learning_rate', type=float, default=0.001,
                help='Learning rate for the PyTorch optimizer (default=0.001)')
        parser.add_argument('-e',  '--epochs',        type=int,   default=30,
                            help='Number of training epochs (default=30)')
        parser.add_argument('-sm', '--save_model_to', default='trained_model.pth',
            help="Location to store the trained model (default='trained_model.pth')")

        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # Create the network and define the loss function and optimizer
        model     = Classifier()
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        
        # Divide the training dataset into two parts: a training set and a validation set
        train_set, _ = mnist()
        batch_size   = 64
        train_n = int(0.7*len(train_set))
        val_n   = len(train_set) - train_n
        train_set, val_set = torch.utils.data.random_split(train_set, [train_n, val_n])
        train_loader = torch.utils.data.DataLoader(train_set,
                                                   batch_size=batch_size, shuffle=True)
        val_loader   = torch.utils.data.DataLoader(val_set,
                                                   batch_size=batch_size, shuffle=True)
        
        # Implement the training loop
        train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
        for e in range(args.epochs):
            train_loss    = 0
            train_correct = 0
            
            for images, labels in train_loader:
                # Set model to training mode and zero gradients since they accumulated
                model.train()
                optimizer.zero_grad()
                
                # Make a forward pass through the network to get the logits
                log_ps = model(images)
                ps     = torch.exp(log_ps)
                
                # Use the logits to calculate the loss
                loss = criterion(log_ps, labels)
                train_loss += loss.item()
                
                # Perform a backward pass through the network to calculate the gradients
                loss.backward()
                
                
                # Take a step with the optimizer to update the weights
                optimizer.step()
                
                # Keep track of how many are correctly classified
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                train_correct += equals.type(torch.FloatTensor).sum().item()        
            else:
                # Compute validattion loss and accuracy
                val_loss    = 0
                val_correct = 0
                
                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    model.eval()     # Sets the model to evaluation mode
                    for images, labels in val_loader:
                        # Forward pass and compute loss
                        log_ps    = model(images)
                        ps        = torch.exp(log_ps)
                        val_loss += criterion(log_ps, labels)
                        
                        # Keep track of how many are correctly classified
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        val_correct += equals.type(torch.FloatTensor).sum().item()
                
                # Store and print losses and accuracies
                train_losses.append(train_loss/len(train_loader))
                train_accuracies.append(train_correct/len(train_set))
                val_losses.append(val_loss/len(val_loader))
                val_accuracies.append(val_correct/len(val_set))
                
                print("Epoch: {}/{}.. ".format(e+1, args.epochs),
                      "Training Loss: {:.3f}.. ".format(train_losses[-1]),
                      "Training Accuracy: {:.3f}".format(train_accuracies[-1]),
                      "Validation Loss: {:.3f}.. ".format(val_losses[-1]),
                      "Validation Accuracy: {:.3f}".format(val_accuracies[-1]))

        # Save the trained network
        torch.save(model.state_dict(), args.save_model_to)
        
        # Plot the training loss curve
        f = plt.figure(figsize=(12, 8))
        plt.plot(train_losses, label='Training loss')
        plt.plot(val_losses,   label='Validation loss')
        plt.xlabel('Epoch number')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
        f.savefig('Training_Loss.pdf', bbox_inches='tight')
        
        # Plot the training accuracy curve
        f = plt.figure(figsize=(12, 8))
        plt.plot(train_accuracies, label='Training accuracy')
        plt.plot(val_accuracies,   label='Validation accuracy')
        plt.xlabel('Epoch number')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()
        f.savefig('Training_Accuracy.pdf', bbox_inches='tight')
        
    def evaluate(self):
        """ Evaluates the neural network using MNIST test data """
        
        print("Evaluating a neural network using MNIST test data")
        parser = argparse.ArgumentParser(description='Evaluation arguments')
        parser.add_argument('--load_model_from', default='')
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # Load the trained model
        model = Classifier()
        if args.load_model_from:
            # model = torch.load(args.load_model_from)
            state_dict = torch.load(args.load_model_from)
            model.load_state_dict(state_dict)
            
        
        # Download and load the test data
        _, test_set = mnist()
        batch_size  = 64
        test_loader  = torch.utils.data.DataLoader(test_set,
                                                   batch_size=batch_size, shuffle=True)
        
        # Evaluate test performance
        test_correct = 0
        
        # Turn off gradients for validation, saves memory and computations
        with torch.no_grad():
            model.eval()     # Sets the model to evaluation mode
            
            # Run through all the test points
            for images, labels in test_loader:
                # Forward pass
                log_ps    = model(images)
                ps        = torch.exp(log_ps)
        
                # Keep track of how many are correctly classified
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                test_correct += equals.type(torch.FloatTensor).sum().item()
            test_accuracy = test_correct/len(test_set)
        print("Test Accuracy: {:.3f}".format(test_accuracy))

if __name__ == '__main__':
    TrainOREvaluate()
    
    
    
    
    
    
    
    
    