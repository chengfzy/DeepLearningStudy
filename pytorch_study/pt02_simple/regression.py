import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data, shape = (100, 1)
y = x.pow(2) + 0.3 * torch.rand(x.size())  # noisy y data, shape=(100,1)


# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x


net = Net(1, 10, 1)
print(net)
optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
loss_func = torch.nn.MSELoss()

plt.ion()  # interactive on

for t in range(100):
    prediction = net(x)
    loss = loss_func(prediction, y)

    optimizer.zero_grad()  # clear gradients for next train
    loss.backward()  # backpropagaton, compute gradients
    optimizer.step()  # apply gradients

    if t % 5 == 0:
        # plow ans show learning process
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-')
        plt.text(0.5, 0, 'Loss={0}'.format(loss.data.numpy()), fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()
