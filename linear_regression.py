#!/usr/bin/env python
# coding: utf-8

# Creating a simple linear regression model using pytorch

# In[1]:


import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np


# Preparing and loading the data

# In[2]:


# create known params

weight = 0.7
bias = 0.3


# creating a range of nums

start = 0
end = 1
step = 0.02

X = torch.arange(start, end, step)
y = weight * X + bias

print(f"The first ten values of X are {X[:10]}")
print(f"The first ten values of y are {y[:10]}")
print(f"The length of X is {len(X)}")
print(f"The length of y is {len(y)}")


# Splitting the data into a training set and a testing set
# A validation set is not necessary in this case hence it will not be created
# Training set (80 % )
# Testing set (20 % )

# In[3]:


train_split = int(0.8 * len(X))
print(f"The total training samples are {train_split}")


# In[4]:


X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]


# Visualising our data

# In[5]:


def plot_predictions(train_data=X_train, train_labels=y_train, test_data=X_test, test_labels=y_test, predictions=None):
    """
    Plots the training and the testing data and compares predicitons
    """
    plt.figure(figsize=(10, 7))

    # training data is to be plotted in blue
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")

    # Testing data is to be plotted in red
    plt.scatter(test_data, test_labels, c="r", s=4, label="Testing data")

    # Checking if there are any predictions to plot them if they do exist
    if predictions is not None:
        plt.scatter(test_data, predictions, c="g", label="Predictions")

    plt.legend(prop={"size": 14})


# In[6]:


plot_predictions()


# We can clearly see a perfect regression line, this functio is only plotting the sets and not the predictions is because we have  not made any. We will now focus on actually creating the model

# In[7]:


class LinearRegression(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.randn(
            1, requires_grad=True, dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(
            1, requires_grad=True, dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        This function is responsible for the computation of the data in the model and how it will proceed
        """
        return self.weights * x + self.bias


# The above class uses two main models -> Gradient descent and backpropogation
# Checking the params of our subclass

# In[8]:


# Creating a random seed

torch.manual_seed(42)

model = LinearRegression()
list(model.parameters())


# In[9]:


model.state_dict()


# In[10]:


# Making Predictions to see how it predicts y_test

with torch.inference_mode():
    y_predictions = model(X_test)

print(y_predictions)
print(y_test)


# These values are completely different, the model did not predict the correct value. Plotting these values to compare

# In[11]:


plot_predictions(predictions=y_predictions)


# Training the model

# In[12]:


torch.manual_seed(42)

# Setting up loss function
loss_function = nn.L1Loss()

# Setting up an optimiser

optimiser = torch.optim.SGD(params=model.parameters(), lr=0.001)

# Creating the training loop

epochs = 10000

epoch_count = []
loss_vals = []
test_loss_vals = []


for epoch in range(epochs):
    model.train()

    # pass to forwrad function
    y_predictions = model(X_train)

    # calculate loss function
    loss = loss_function(y_predictions, y_train)
    # optimising based on loss function
    optimiser.zero_grad()

    # perform backpropogation on the loss wrt the params
    loss.backward()

    # step the optimiser
    optimiser.step()

    # testing the model
    model.eval()
    with torch.inference_mode():
        test_predictions = model(X_test)

        # calculate loss function

        test_loss = loss_function(test_predictions, y_test)
        epoch_count.append(epoch)
        loss_vals.append(loss)
        test_loss_vals.append(test_loss)
        print(f"Epoch: {epoch} | Loss: {loss} | Test Loss: {test_loss}")

        print(model.state_dict())


# In[13]:


with torch.inference_mode():
    final_predictions = model(X_test)

print(final_predictions)


# In[14]:


plot_predictions(predictions=final_predictions)


# The predicted are values are extremly accurate as seen from the given graph

# In[16]:


# Plotting the loss curve

plt.plot(epoch_count, np.array(torch.tensor(
    (loss_vals)).numpy()), label="Training Loss")
plt.plot(epoch_count, test_loss_vals, label="Testing Loss")
plt.title("Training and Testing Loss curves")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()


# In[17]:


# Saving the model
filepath = "../model/model.pth"
torch.save(model, filepath)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




