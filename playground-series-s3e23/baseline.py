# %%
import pandas as pd
import torch
import torch.nn as nn
import torch.functional as f
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# %%
train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")
sample_submission = pd.read_csv("./data/sample_submission.csv")

train["is_generated"] = 1
test["is_generated"] = 1

# Drop column id
train.drop("id", axis=1, inplace=True)
test.drop("id", axis=1, inplace=True)


total = pd.concat([train, test], ignore_index=True)

print("The shape of the train data:", train.shape)
print("The shape of the test data:", test.shape)

# %%
target = "defects"
X = train.drop(target, axis=1)
Y = train[target]


device = "cuda"

train_set = torch.tensor(X.values, dtype=torch.float32).to(device)
train_target = torch.tensor(Y.values, dtype=torch.float32).to(device)
pred_set = torch.tensor(test.values, dtype=torch.float32).to(device)


# %%
class NN(nn.Module):
    def __init__(self, input_dim, hidden_dim1=64, hidden_dim2=32):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(hidden_dim2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.bn1(x)
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.bn2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return self.sigmoid(x)


input_dim = train_set.shape[1]

model = NN(input_dim).to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)


loss_history = []

epochs = 3000
for epoch in range(3000):
    optimizer.zero_grad()
    outputs = model(train_set).squeeze()

    loss = criterion(outputs, train_target)
    loss_history.append(loss.item())

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    loss.backward()
    optimizer.step()
train_steps = range(1, len(loss_history) + 1)

# %%
model.eval()

preds = model(pred_set)
# %%
pred_set
# %%
test.head()
# %%
test.columns

# %%
len(preds)
# %%
sample_submission = pd.read_csv("./data/sample_submission.csv")

sample_submission["defects"] = preds.detach().cpu()

sample_submission.to_csv("./submissions/1.csv", index=False)

# %%
dataset = Dataset()
