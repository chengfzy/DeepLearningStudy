from __future__ import print_function
import torch
import numpy as np
import util


# getting started
def getting_started():
    print(util.Section('Getting Started'))

    # construction
    print(util.SubSection('Construction'))
    xa1 = torch.empty(5, 3)  # uninitialized
    xa2 = torch.rand(5, 3)  # randomly initialized matrix
    xa3 = torch.zeros(5, 3, dtype=torch.long)  # filled zeros and of dtype long
    xa4 = torch.tensor([5.5, 3])  # directly from data
    xa5 = xa3.new_ones(5, 3, dtype=torch.double)  # new_* method take in sizes
    xa6 = torch.randn_like(xa5, dtype=torch.float)  # override dtype with same size
    print(f'x size = {xa6.size()}')

    # operations
    xb1 = torch.rand(5, 3)
    yb1 = torch.rand(5, 3)

    # operation: add
    print(util.SubSection('Operations: Add'))
    print(f'xb1 + yb1 = {xb1 + yb1}')
    print(f'xb1 + yb1 = {torch.add(xb1, yb1)}')
    # with output argument
    rb1 = torch.empty(5, 3)
    torch.add(xb1, yb1, out=rb1)
    print(f'rb1 = {rb1}')
    # add in place
    yb1.add_(xb1)
    print(f'yb1 = {yb1}')
    # index
    print(f'xb1[:,1] = {xb1[:, 1]}')

    # operation: resize
    print(util.SubSection('Operations: Resize'))
    xb2 = torch.randn(4, 4)
    yb2 = xb2.view(16)
    zb2 = xb2.view(-1, 8)
    print(f'xb2 = {xb2}')
    print(f'yb2 = {yb2}')
    print(f'zb2 = {zb2}')
    print(f'xb2.size = {xb2.size()}, yb2.size = {yb2.size()}, zb2.size = {zb2.size()}')
    # if only one element, can use .item() to get the values as a python number
    xb3 = torch.randn(1)
    print(f'xb3 = {xb3}')
    print(f'xb3.item() = {xb3.item()}')

    # numpy bridge, change one will change the other
    print(util.SubSection('NumPy Bridge'))
    # torch => numpy
    xc1 = torch.ones(5)
    print(f'xc1 = {xc1}')
    yc1 = xc1.numpy()
    print(f'yc1 = {yc1}')
    # add, y will also changed
    xc1.add_(1)
    print(f'xc1 = {xc1}')
    print(f'yc1 = {yc1}')
    # numpy => torch
    xc2 = np.ones(5)
    yc2 = torch.from_numpy(xc2)
    np.add(xc2, 1, out=xc2)
    print(f'xc2 = {xc2}')
    print(f'yc2 = {yc2}')

    # CUDA tensors
    print(util.SubSection('CUDA Tensors'))
    xd1 = torch.rand((3, 2))
    if torch.cuda.is_available():
        print('use CUDA')
        device = torch.device('cuda')
        yd1 = torch.ones_like(xd1, device=device)  # directly create a tensor on GPU
        xd2 = xd1.to(device)
        zd1 = xd2 + yd1
        print(f'zd1 = {zd1}')
        print(f'to CPU, zd1 = {zd1.to("cpu", torch.double)}')  # "to" can also change dtype together


# autograd
def auto_grad():
    print(util.Section('Auto Grad'))
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
    print((x**2).requires_grad)
    with torch.no_grad():
        print((x**2).requires_grad)


if __name__ == '__main__':
    print('pytorch version', torch.__version__)
    getting_started()
    auto_grad()
