import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

# data
n_data = torch.ones(100, 2)
x0 = torch.normal(2 * n_data, 1)  # class0 x data, shape=(100,2)
y0 = torch.zeros(100)  # class0 y data, shape=(100,1)
x1 = torch.normal(-2 * n_data, 1)  # class1 x data, shape=(100,2)
y1 = torch.ones(100)  # class1 y data, shape=(100,1)
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # shape=(200,2)
y = torch.cat((y0, y1), ).type(torch.LongTensor)  # shape=(200,)


# plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
# plt.show()

# method 1
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x


net1 = Net(2, 10, 2)

# method 2
net2 = torch.nn.Sequential(
    torch.nn.Linear(2, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 2),
)

print(net1)
print(net2)

# optimizer = torch.optim.SGD(net1.parameters(), lr=0.02)
# loss_func = torch.nn.CrossEntropyLoss()
#
# plt.ion()  # interactive on
#
# for t in range(100):
#     out = net1(x)
#     loss = loss_func(out, y)
#
#     optimizer.zero_grad()  # clear gradients for next train
#     loss.backward()  # backpropagaton, compute gradients
#     optimizer.step()  # apply gradients
#
#     if t % 2 == 0:
#         # plow ans show learning process
#         plt.cla()
#         prediction = torch.max(out, 1)[1]
#         pred_y = prediction.data.numpy().squeeze()
#         target_y = y.data.numpy()
#         plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
#         accuary = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
#         plt.text(1.5, -4, 'Accuracy={0}'.format(accuary), fontdict={'size': 10, 'color': 'red'})
#         plt.pause(0.1)
#
# plt.ioff()
# plt.show()
