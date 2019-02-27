from __future__ import print_function
import torch
import numpy as np
from common.debug_info import *


# getting started
def getting_started():
    print(section('getting started'))

    # construction
    print(sub_section('construction'))
    xa1 = torch.empty(5, 3)  # uninitialized
    xa2 = torch.rand(5, 3)  # randomly initialized matrix
    xa3 = torch.zeros(5, 3, dtype=torch.long)  # filled zeros and of dtype long
    xa4 = torch.tensor([5.5, 3])  # directly from data
    xa5 = xa3.new_ones(5, 3, dtype=torch.double)  # new_* method take in sizes
    xa6 = torch.randn_like(xa3, dtype=torch.float)  # override dtype with same size

    # operations
    xb1 = torch.rand(5, 3)
    yb1 = torch.rand(5, 3)

    # operation: add
    print(sub_section('operations: add'))
    print('xb1 + yb1 = ', xb1 + yb1)
    print('xb1 + yb1 = ', torch.add(xb1, yb1))
    # with output argument
    rb1 = torch.empty(5, 3)
    torch.add(xb1, yb1, out=rb1)
    print('rb1 = ', rb1)
    # add in place
    yb1.add_(xb1)
    print('yb1 = ', yb1)
    # index
    print('xb1[:,1] = ', xb1[:, 1])

    # operation: resize
    print(sub_section('operations: resize'))
    xb2 = torch.randn(4, 4)
    yb2 = xb2.view(16)
    zb2 = xb2.view(-1, 8)
    print('xb2 = ', xb2)
    print('yb2 = ', yb2)
    print('zb2 = ', zb2)
    print(xb2.size(), yb2.size(), zb2.size())
    # if only one element, can use .item() to get the values as a python number
    xb3 = torch.randn(1)
    print('xb3 = ', xb3)
    print('xb3.item() = ', xb3.item())

    # numpy bridge
    print(sub_section('numpy bridge'))
    # torch => numpy
    xc1 = torch.ones(5)
    print('xc1 = ', xc1)
    yc1 = xc1.numpy()
    print('yc1 = ', yc1)
    # add, y will also changed
    xc1.add_(1)
    print('xc1 = ', xc1)
    print('yc1 = ', yc1)
    # numpy => torch
    xc2 = np.ones(5)
    yc2 = torch.from_numpy(xc2)
    np.add(xc2, 1, out=xc2)
    print('xc2 = ', xc2)
    print('yc2 = ', yc2)

    # CUDA tensors
    print(sub_section('CUDA tensors'))
    if torch.cuda.is_available():
        print('use CUDA')
        device = torch.device('cuda')


# autograd
def auto_grad():
    print(section('Auto Grad'))
    # grad function
    x = torch.ones(2, 2, requires_grad=True)
    y = x + 2
    z = y * y * 3
    out = z.mean()
    print('y = ', y)
    print('y.grad_fn = ', y.grad_fn)
    print('z = {0}, out = {1}'.format(z, out))

    # requires grad setting
    a = torch.randn(2, 2)
    a = (a * 3) / (a - 1)
    print('a.requires_grad = ', a.requires_grad)
    a.requires_grad_(True)
    print('a.requires_grad = ', a.requires_grad)
    b = (a * a).sum()
    print('b.grad_fn = ', b.grad_fn)

    # gradients
    out.backward()
    print('x.grad = ', x.grad)

    # do more crazy things with autograd
    x = torch.randn(3, requires_grad=True)
    y = x * 2
    while y.data.norm() < 1000:
        y = y * 2
    print('y = ', y)
    gradients = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
    y.backward(gradients)
    print('x.grad = ', x.grad)

    # stop autograd
    print(x.requires_grad)
    print((x ** 2).requires_grad)
    with torch.no_grad():
        print((x ** 2).requires_grad)


if __name__ == '__main__':
    print('pytorch version', torch.__version__)
    # getting_started()
    auto_grad()
