import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F
import tensorflow as tf
import datetime


def starting_train(train_dataset, val_dataset, model, hyperparameters, n_eval):
    """
    Trains and evaluates a model.

    Args:
        train_dataset:   PyTorch dataset containing training data.
        val_dataset:     PyTorch dataset containing validation data.
        model:           PyTorch model to be trained.
        hyperparameters: Dictionary containing hyperparameters.
        n_eval:          Interval at which we evaluate our model.
    """

    # Get keyword arguments
    batch_size, epochs = hyperparameters["batch_size"], hyperparameters["epochs"]

    # Initialize dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True
    )
    # weights = 
    # Initalize optimizer (for gradient descent) and loss function
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()
    
    step = 0
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")
        
        # Loop over each batch in the dataset
        for batch in tqdm(train_loader):
            x, y = batch 
            # TODO: Forward propagate
            outputs = model(x)
            # TODO: Backpropagation and gradient descent
            loss = loss_fn(x, outputs)
            loss.backward() 
            optimizer.step()
            optimizer.zero_grad()
            
        # Periodically evaluate our model + log to Tensorboard
        if step % n_eval == 0:
            # TODO:
            # Compute training loss and accuracy.
            # Log the results to Tensorboard.
            print('Epoch: ', epoch, 'Loss: ', loss.item())
            model.eval()
            model.train()
            # TODO:
            # Compute validation loss and accuracy.
            # Log the results to Tensorboard. 
            # Don't forget to turn off gradient calculations!
            evaluate(val_loader, model, loss_fn)
            with torch.no_grad(): # IMPORTANT: turn off gradient computations
                outputs = model(val_loader)
                prediction = torch.argmax(outputs)
            step += 1
        print('Prediction: ' , prediction.item())

def forwardpropagate(self, x):
     x = self.flatten(x)
     x = self.fc1(x)
     x = F.relu(x)  
     x = self.fc2(x)
     x = F.relu(x)
     x = self.fc3(x)    
     return x

def compute_accuracy(outputs, labels):
    """
    Computes the accuracy of a model's predictions.

    Example input:
        outputs: [0.7, 0.9, 0.3, 0.2]
        labels:  [1, 1, 0, 1]

    Example output:
        0.75
    """

    n_correct = (torch.round(outputs) == labels).sum().item()
    n_total = len(outputs)
    return n_correct / n_total


def evaluate(val_loader, model, loss_fn):
    """
    Computes the loss and accuracy of a model on the validation dataset.

    TODO!
    
    """
    x,y = val_loader
    output = model(x)
    accuracy = compute_accuracy(output, y)
    loss = loss_fn(val_loader, output)

    return accuracy, loss
