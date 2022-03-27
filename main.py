import math
import os

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import json
import numpy as np
from inspect import signature

from model import SymbolicNN, count_double_inputs
from functions import *
from regularization import L12Smooth
from utils import print_network

DEVICE = "cpu"
TRIALS = 2

class Dataset:
    def __init__(self, func, train_size, test_size, train_range_min, train_range_max, test_range_min, test_range_max) -> None:
        self.func = func
        self.train_size = train_size
        self.test_size = test_size
        self.train_range_min = train_range_min
        self.train_range_max = train_range_max
        self.test_range_min = test_range_min
        self.test_range_max = test_range_max

        self._x_train, self._y_train = self.generate_data(func, train_size, train_range_min, train_range_max)
        self._x_test, self._y_test = self.generate_data(func, test_size, test_range_min, test_range_max) 


    def generate_data(self, func, size, range_min, range_max):

        x_dim = len(signature(func).parameters)    

        x = (range_max - range_min) * torch.rand([size, x_dim]) + range_min
        y = torch.tensor([[func(*x_i)] for x_i in x])

        return x, y

    @property
    def x_train(self):
        return self._x_train

    @property
    def x_test(self):
        return self._x_test

    @property
    def y_train(self):
        return self._y_train

    @property
    def y_test(self):
        return self._y_test


def trainer(n_layers, funcs:list, func_name, var_names:list, dataset:Dataset, epochs:int, lr_rate:float=1e-2, reg_weight:float=5e-3):

    loss_list = []          # Total loss (MSE + regularization)
    init_sd_first = 0.1
    init_sd_last = 1.0
    init_sd_middle = 0.5

    x_dim = len(signature(dataset.func).parameters)  # Number of input arguments to the function
    width = len(funcs)
    n_double = count_double_inputs(funcs)


    for trial in range(TRIALS):
        print("Training on function " + func_name + " Trial " + str(trial+1) + " out of " + str(TRIALS))
        
        model = SymbolicNN(n_layers,funcs=funcs, initial_weights=[
                                  # kind of a hack for truncated normal
                                  torch.fmod(torch.normal(0, init_sd_first, size=(x_dim, width + n_double)), 2),
                                  torch.fmod(torch.normal(0, init_sd_middle, size=(width, width + n_double)), 2),
                                  torch.fmod(torch.normal(0, init_sd_middle, size=(width, width + n_double)), 2),
                                  torch.fmod(torch.normal(0, init_sd_last, size=(width, 1)), 2)
                              ]).to(DEVICE)
        criterion = nn.MSELoss()
        optimizer = optim.RMSprop(model.parameters(),
            lr=lr_rate * 10,
            alpha=0.9,  # smoothing constant
            eps=1e-10,
            momentum=0.0,
            centered=False
        )

        loss_val = np.nan
        while np.isnan(loss_val):
            x_dim = len(signature(dataset.func).parameters)  # Number of input arguments to the function
            lmbda = lambda epoch: 0.1
            scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)
            summary_step = 1000

            loss_list = []          # Total loss (MSE + regularization)
            error_test_list = []    # Test error

            for epoch in range(epochs):
                optimizer.zero_grad()  # zero the parameter gradients
                outputs = model(dataset.x_train)  # forward pass
                regularization = L12Smooth()
                mse_loss = criterion(outputs, dataset.y_train)

                reg_loss = regularization(model.get_weights_tensor())
                loss = mse_loss + reg_weight * reg_loss
                loss.backward()
                optimizer.step()

                if epoch % summary_step == 0:
                    error_val = mse_loss.item()
                    reg_val = reg_loss.item()
                    loss_val = loss.item()
                    loss_list.append(loss_val)

                    with torch.no_grad():  # test error
                        test_outputs = model(dataset.x_test)
                        test_loss = F.mse_loss(test_outputs, dataset.y_test)
                        error_test_val = test_loss.item()
                        error_test_list.append(error_test_val)

                    print("Epoch: %d\tTotal training loss: %f\tTest error: %f" % (epoch, loss_val, error_test_val))

                    if np.isnan(loss_val) or loss_val > 1000:  # If loss goes to NaN, restart training
                        break

                if epoch == 2000:
                    scheduler.step()  # lr /= 10

            scheduler.step()  # lr /= 10 again

            for epoch in range(epochs):
                optimizer.zero_grad()  # zero the parameter gradients
                outputs = model(dataset.x_train)
                regularization = L12Smooth()
                mse_loss = criterion(outputs, dataset.y_train)
                reg_loss = regularization(model.get_weights_tensor())
                loss = mse_loss + reg_weight * reg_loss
                loss.backward()
                optimizer.step()

                if epoch % summary_step == 0:
                    loss_val = loss.item()
                    loss_list.append(loss_val)

                    with torch.no_grad():  
                        test_outputs = model(dataset.x_test)
                        test_loss = F.mse_loss(test_outputs, dataset.y_test)
                        error_test_val = test_loss.item()
                        error_test_list.append(error_test_val)

                    print("Epoch: %d\tTotal training loss: %f\tTest error: %f" % (epoch, loss_val, error_test_val))

                    if np.isnan(loss_val) or loss_val > 1000:  # If loss goes to NaN, restart training
                        break

            with torch.no_grad():
                weights = model.get_weights()
                expr = print_network(weights, funcs, var_names[:x_dim])
                print('Expression: ',expr)

            # Save the model
            # check if the directory exists
            if not os.path.exists(f'./models/{func_name}/'):
                os.makedirs(f'./models/{func_name}/')

            torch.save(model.state_dict(), f'./models/{func_name}/' + func_name + '_trial_' +str(trial) + '.pth')
            # Write the expr to a json with other hyperparameters
            with open(f'./models/{func_name}/' + func_name + '_' + str(trial) + '.json', 'w') as f:
                json.dump({
                    'func': func_name,
                    'expr': str(expr),
                    'epochs': epochs,
                    'lr_rate': lr_rate,
                    'reg_weight': reg_weight,
                    'loss_list': loss_list,
                    'error_test_list': error_test_list,
                    'n_layers': n_layers,
                }, f)

def main(n_layers:int=2):
    
    dataset = Dataset(lambda x, y: np.exp(x+y), 256, 100, -1, 1, -2, 2)
    funcs = [
        *[Constant()] * 2,
        *[Identity()] * 4,
        *[Square()] * 4,
        *[Exp()] * 2,
        
        *[Sigmoid()] * 2,
        *[Product()] * 2,
    ]

    trainer(n_layers,funcs, 'exp^(x+y)', ['x', 'y', 'z'], dataset, 10001)
    
if __name__ == "__main__":
    main()

