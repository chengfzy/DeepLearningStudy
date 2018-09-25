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


if __name__ == '__main__':
    getting_started()
