import torch
import torch.nn as nn

# Simple logistic regression model
class LogisticRegression(nn.Module):
    
    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    # forward pass we take linear part and then apply sigmoid
    # to get the answer b/w 0, 1
    def forward(self, x):
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted