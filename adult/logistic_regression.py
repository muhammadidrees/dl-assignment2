import torch
import torch.nn as nn

import sys
sys.path.append( './data/adult/' )
from adult_torch_model import *

torch.manual_seed(69)

train_dataset = AdultDataset()
test_dataset = AdultDataset(train=False)

batch_size = 1000

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

class LogisticRegression(nn.Module):
    
    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted

model = LogisticRegression(dataset.n_features)

# loss and optimizer
learning_rate = 0.0001
criterian = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# training loop
num_epoch = 1000
for epoch in range(num_epoch):
    for i, (X_train, y_train) in enumerate(train_loader):
         # forward pass
        y_predicted = model(X_train)

        # loss function
        loss = criterian(y_predicted, y_train)

        # backpass
        loss.backward()

        # updates
        optimizer.step()

        # zero gradients
        optimizer.zero_grad()

    if (epoch + 1) % 100 == 0:
        print(f'epoch: {epoch + 1}, loss = {loss.item():.4f}')

with torch.no_grad():
    for i, (X_test, y_test) in enumerate(test_loader):
        y_predicted = model(X_test)
        y_predicted_cls = y_predicted.round()
        acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
        print(f'Accuracy = {acc:.4f}')