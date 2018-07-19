import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5*5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # max pooling over a (2, 2) windows
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # if the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print("net: ", net)

params = list(net.parameters())
print(len(params))
print(params[0].size())  # conve1's weight

input = torch.randn(1, 1, 32, 32)
out = net(input)
print('out: ', out)

# zero the gradient buffers of all parameters and backprops with random gradients
net.zero_grad()
out.backward(torch.randn(1, 10))

# Loss Function
output = net(input)
target = torch.arange(1, 11)  # a dummy target, for example
target = target.view(1, -1)  # make it the same shape as output
criterion = nn.MSELoss()
loss = criterion(output, target)
print("loss: ", loss)
print("MSELoss: ", loss.grad_fn)
print("Linear: ", loss.grad_fn.next_functions[0][0])
print("ReLU: ", loss.grad_fn.next_functions[0][0].next_functions[0][0])

# Backprop
net.zero_grad()  # zeroes the gradient buffers of all parameters
print('conv1.bias.grad before backward', net.conv1.bias.grad)
loss.backward()
print('conv1.bias.grad after backward', net.conv1.bias.grad)

# Update the Weights: weight = weight - learning_rate * gradient
learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)

# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)
# in your training loop
optimizer.zero_grad()  # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()  # does the update
