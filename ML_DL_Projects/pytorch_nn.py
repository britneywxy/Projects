"""
Validate the NN implementation using PyTorch.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

class SingleLayerMLP(nn.Module):
    """constructing a single layer neural network with Pytorch"""
    def __init__(self, indim, outdim, hidden_layer=100):
        super(SingleLayerMLP, self).__init__()
        self.linear1 = nn.Linear(indim,hidden_layer)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_layer,outdim)

    def forward(self, x):
        """
        x shape (batch_size, indim)
        """
        x = self.linear1(x.float())
        x = self.relu(x)
        x = self.linear2(x)
        return x


class DS(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.length = len(X)
        self.X = X
        self.Y = Y

    def __getitem__(self, idx):
        x = self.X[idx, :]
        y = self.Y[idx]
        return (x, y)

    def __len__(self):
        return self.length


def validate(loader, is_train):
    """takes in a dataloader, then returns the model loss and accuracy on this loader"""
    loss_list_temp, acc_list_temp = [], []

    # setting model state
    model.eval()
    total_acc = 0
    total_sample = 0
    for x,y in loader:
        x = x.to(device=device)
        y = y.to(device=device)
        optimizer.zero_grad()

        # make predictions for this batch
        outputs = model(x)

        loss = loss_fun(outputs,y)

        if is_train:
            # adjust weights
            loss.backward()
            # adjust learning weights
            optimizer.step()

        classifications = torch.argmax(outputs, dim=1)
        correct_predictions = sum(classifications==y).item()
        total_acc += correct_predictions
        total_sample += len(y)

        loss_list_temp.append(loss.item())
        acc_list_temp.append(total_acc/total_sample)

    return np.mean(loss_list_temp), np.mean(acc_list_temp)

if __name__ == "__main__":
    """The dataset loaders were provided for you.
    You need to implement your own training process.
    You need plot the loss and accuracies during the training process and test process.
    """

    indim = 60
    outdim = 2
    hidden_dim = 100
    lr = 0.01
    batch_size = 64
    epochs = 500

    #dataset
    Xtrain = pd.read_csv("./data/X_train.csv")
    Ytrain = pd.read_csv("./data/y_train.csv")
    scaler = MinMaxScaler()
    Xtrain = pd.DataFrame(scaler.fit_transform(Xtrain), columns=Xtrain.columns).to_numpy()
    Ytrain = np.squeeze(Ytrain)
    m1, n1 = Xtrain.shape
    print(m1, n1)
    train_ds = DS(Xtrain, Ytrain)
    train_loader = DataLoader(train_ds, batch_size=batch_size)

    Xtest = pd.read_csv("./data/X_test.csv")
    Ytest = pd.read_csv("./data/y_test.csv").to_numpy()
    Xtest = pd.DataFrame(scaler.fit_transform(Xtest), columns=Xtest.columns).to_numpy()
    Ytest = np.squeeze(Ytest)
    m2, n2 = Xtest.shape
    print(m1, n2)
    test_ds = DS(Xtest, Ytest)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    #construct the model - use pytorch function
    model = SingleLayerMLP(indim, outdim, hidden_dim)
    loss_fun = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    train_loss_list, train_acc_list, test_loss_list, test_acc_list = [], [], [], []

    for epoch in range(epochs):
        train_loss, train_acc = validate(train_loader, True)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)

        test_loss, test_acc = validate(test_loader,False)
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)

        print("Epoch:", epoch)
        print("Train Loss:", train_loss, " Train accuracy:",train_acc)
        print("Test Loss:", test_loss, " Test accuracy:", test_acc)


    import pickle
    with open("metrics_hw_2.pkl", "wb") as f:
        pickle.dump((train_loss_list,train_acc_list,test_loss_list,test_acc_list), f)

