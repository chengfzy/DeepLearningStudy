import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.manifold import TSNE

# hyper parameters
EPOCH = 1
BATCH_SIZE = 50
LR = 0.001  # learning rate
DOWNLOAD_MNIST = True

train_data = torchvision.datasets.MNIST(
    root='../../data/mnist',
    train=True,
    transform=torchvision.transforms.ToTensor(),  # normalize to range [0.0, 1.0]
    download=DOWNLOAD_MNIST
)

# # plot one example
print('train data size = ', train_data.train_data.size())  # (60000, 28, 28)
print('train label size = ', train_data.train_labels.size())  # 60000
# plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
# plt.title('{0}'.format(train_data.train_labels[0]))
# plt.show()

# data loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

test_data = torchvision.datasets.MNIST(root='../../data/mnist', train=False)
test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:2000] / 255  # shape = (2000, 1, 28, 28)
test_y = test_data.test_labels[:2000]


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # input (1, 28, 28)
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),  # output (16, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # output (16, 14, 14)
        )
        self.conv2 = nn.Sequential(  # input (16, 14, 14)
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),  # output(32, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # output (32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)  # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # (batch, 32 * 7 * 7)
        output = self.out(x)
        return output, x


cnn = CNN()
print(cnn)

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()


# function for visualization
def plot_with_labels(lowDWeights, labels):
    plt.cla()
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    for x, y, s in zip(X, Y, labels):
        c = cm.rainbow(int(255 * s / 9))
        plt.text(x, y, s, backgroundcolor=c, fontsize=9)
    plt.xlim(X.min(), X.max())
    plt.ylim(Y.min(), Y.max())
    plt.title('Visualize last layer')
    plt.show()
    plt.pause(0.1)


plt.ion()
# training and testing
for epoch in range(EPOCH):
    for step, (bx, by) in enumerate(train_loader):
        output = cnn(bx)[0]
        loss = loss_func(output, by)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            test_output, last_layer = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.squeeze().numpy()
            accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
            print('Epoch: {0} | train_loss: {1} | test_acccuracy: {2}'.format(epoch, loss.data.numpy(), accuracy))

            # visualization of trained flatted layer (T-SNE)
            tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
            plot_only = 100
            low_dim_embs = tsne.fit_transform(last_layer.data.numpy()[:plot_only, :])
            labels = test_y.numpy()[:plot_only]
            plot_with_labels(low_dim_embs, labels)

plt.ioff()

# print 10 predictions from test data
test_output, _ = cnn(test_x[:10])
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print('prediction number: ', pred_y)
print('real number: ', test_y[:10].numpy())
