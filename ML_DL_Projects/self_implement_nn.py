"""
Implement a single layer neural network from scratch.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, List, Tuple, Dict


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

class Transform(object):
    """
    This is the base class. You do not need to change anything.
    Read the comments in this class carefully.
    """
    def __init__(self):
        """
        Initialize any parameters
        """
        pass

    def forward(self, x):
        """
        x should be passed as column vectors
        """
        pass

    def backward(self, grad_wrt_out):
        """
        Compute and save the gradients wrt the parameters for step()
        Return grad_wrt_x which will be the grad_wrt_out for previous Transform
        """
        pass

    def step(self):
        """
        Apply gradients to update the parameters
        """
        pass

    def zerograd(self):
        """
        This is used to Reset the gradients.
        Usually called before backward()
        """
        pass

class ReLU(Transform):
    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x):
        """
        x shape (indim, batch_size)
        return shape (indim, batch_size)
        """
        self.x = x
        return (torch.maximum(x,torch.tensor(0.0)))

    def backward(self, grad_wrt_out):
        """
        grad_wrt_out shape (outdim, batch_size)
        """
        grad_wrt = torch.where(self.x>0, grad_wrt_out, torch.tensor(0.0)) #condition, input, output
        return grad_wrt

class LinearMap(Transform):
    def __init__(self, indim, outdim, lr=0.01):
        """
        indim: input dimension
        outdim: output dimension
        lr: learning rate
        """
        super(LinearMap, self).__init__()
        self.weights = 0.01 *torch.rand((outdim, indim), dtype=torch.float64, requires_grad=True, device=device)
        self.bias = 0.01 * torch.rand((outdim, 1), dtype=torch.float64, requires_grad=True, device=device)
        self.lr = lr


    def forward(self, x):
        """
        x shape (indim, batch_size)
        return shape (outdim, batch_size)
        """
        self.x = x
        # weights * x + bias
        for_output = torch.matmul(self.weights, x) + self.bias
        return for_output

    def backward(self, grad_wrt_out):
        """
        grad_wrt_out shape (outdim, batch_size)
        return shape (indim, batch_size)
        """
        #compute grad_wrt_weights
        grad_wrt_weights = torch.matmul(grad_wrt_out,self.x.T)
        self.grad_wrt_weights = grad_wrt_weights

        #compute grad_wrt_bias
        grad_wrt_bias = torch.sum(grad_wrt_out,dim=1, keepdim=True) #.reshape(-1,1)
        self.grad_wrt_bias = grad_wrt_bias

        #compute & return grad_wrt_input
        grad_wrt_input = torch.matmul(self.weights.T, grad_wrt_out)
        return grad_wrt_input


    def step(self):
        """
        apply gradients calculated by backward() to update the parameters
        """
        self.weights = self.weights - self.lr * self.grad_wrt_weights
        self.bias = self.bias - self.lr * self.grad_wrt_bias

    def zerograd(self):
        self.grad_wrt_weights = None
        self.grad_wrt_bias = None

class SoftmaxCrossEntropyLoss(object):
    def forward(self, logits, labels):
        """
        logits are pre-softmax scores, labels are one-hot labels of given inputs
        logits and labels are in the shape of (num_classes, batch_size)
        returns loss as a scalar (i.e. mean value of the batch_size loss)
        """
        self.logits = logits
        self.labels = labels

        exp_logits = torch.exp(logits)
        softmax = exp_logits / torch.sum(exp_logits, axis = 0, keepdim = True)
        loss = -torch.sum(labels * torch.log(softmax)) / logits.shape[1]

        self.softmax = softmax
        self.loss = loss
        return loss


    def backward(self):
        """
        return grad_wrt_logits shape (num_classes, batch_size)
        (don't forget to divide by batch_size because your loss is a mean)
        """
        grad_wrt_logits = (self.softmax - self.labels) / self.logits.shape[1]
        return grad_wrt_logits


    def getAccu(self):
        """
        return accuracy here
        """
        preds = torch.argmax(self.softmax, dim=0)
        trues = torch.argmax(self.labels, dim=0)
        accuracy = torch.mean(torch.eq(preds,trues).float())
        return accuracy


class SingleLayerMLP(Transform):
    """constructing a single layer neural network with the previous functions"""
    def __init__(self, indim, outdim, hidden_layer=100, lr=0.01):
        super(SingleLayerMLP, self).__init__()
        self.linear1 = LinearMap(indim,hidden_layer,lr)
        self.relu = ReLU()
        self.linear2 = LinearMap(hidden_layer,outdim,lr)
        self.softmaxCrossEntropy = SoftmaxCrossEntropyLoss()


    def forward(self, x):
        """
        x shape (indim, batch_size)
        return the presoftmax logits shape(outdim, batch_size)
        """
        for_output = self.linear1.forward(x)
        relu_output = self.relu.forward(for_output)
        pre_logit = self.linear2.forward(relu_output)
        return pre_logit


    def backward(self, grad_wrt_out):
        """
        grad_wrt_out shape (outdim, batch_size)
        calculate the gradients wrt the parameters
        """
        grad_wrt_input = self.linear2.backward(grad_wrt_out)
        grad_wrt = self.relu.backward(grad_wrt_input)
        grad_wrt_params = self.linear1.backward(grad_wrt)
        return grad_wrt_params

    def step(self):
        """update model parameters"""
        self.linear1.step()
        self.linear2.step()

    def zerograd(self):
        self.linear1.zerograd()
        self.linear2.zerograd()

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

def labels2onehot(labels: np.ndarray):
    return np.array([[i==lab for i in range(2)] for lab in labels]).astype(int)

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
    Ytest = pd.read_csv("./data/y_test.csv")
    Xtest = pd.DataFrame(scaler.fit_transform(Xtest), columns=Xtest.columns).to_numpy()
    Ytest = np.squeeze(Ytest)
    m2, n2 = Xtest.shape
    print(m2, n2)
    test_ds = DS(Xtest, Ytest)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    #construct the model
    model = SingleLayerMLP(indim, outdim, hidden_dim, lr)
    loss_fun = SoftmaxCrossEntropyLoss()

    #construct the training process
    train_loss_list, train_acc_list, test_loss_list, test_acc_list = [], [], [], []
    for epoch in range(epochs): #epochs
        train_loss_list_temp, train_acc_list_temp, test_loss_list_temp, test_acc_list_temp = [], [], [], []

        for x, y in train_loader:
            x = torch.tensor(x).T # correct x size: (indim, batch_size)
            y = torch.tensor(labels2onehot(y)).T

            # Foward propagation
            pre_logit = model.forward(x)
            train_loss = loss_fun.forward(pre_logit,y)
            train_acc = loss_fun.getAccu()

            # Zero grad
            model.zerograd()

            # Backward propagation
            grad_wrt_logits = loss_fun.backward()
            model.backward(grad_wrt_logits)
            model.step()

            train_loss_list_temp.append(train_loss.item())
            train_acc_list_temp.append(train_acc.item())

        train_loss_list.append(np.mean(train_loss_list_temp))
        train_acc_list.append(np.mean(train_acc_list_temp))

        for x, y in test_loader:
            x = torch.tensor(x).T
            y = torch.tensor(labels2onehot(y)).T

            # Foward propagation
            pre_logit = model.forward(x)
            test_loss = loss_fun.forward(pre_logit,y)
            test_acc = loss_fun.getAccu()

            test_loss_list_temp.append(test_loss.item())
            test_acc_list_temp.append(test_acc.item())

        test_loss_list.append(np.mean(test_loss_list_temp))
        test_acc_list.append(np.mean(test_acc_list_temp))

        print("Epoch:", epoch)
        print("Train Loss:", train_loss_list[-1], " Train accuracy:",train_acc_list[-1])
        print("Test Loss:", test_loss_list[-1], " Test accuracy:", test_acc_list[-1])

    #save loss history, accuracy history -- plot
    import pickle
    with open("metrics_hw.pkl", "wb") as f:
        pickle.dump((train_loss_list,train_acc_list,test_loss_list,test_acc_list), f)