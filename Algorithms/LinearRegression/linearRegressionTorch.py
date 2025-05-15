import torch
X = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
y = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
model = torch.nn.Linear(1, 1)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
for _ in range(1000):
    optimizer.zero_grad()
    y_pred = model(X)
    loss = criterion(y_pred, y)
    loss.backward()
    optimizer.step()
with torch.no_grad():
    predictions = model(torch.tensor([[6.0], [7.0], [8.0]]))
    print(predictions)
