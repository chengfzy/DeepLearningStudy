"""
Convert torch model to torch script via Tracing, and save to file
"""

import torch
import torchvision


# class MyModule(torch.nn.Module):
class MyModule(torch.jit.ScriptModule):  # for tracing
    def __init__(self, N, M):
        super(MyModule, self).__init__()
        self.weight = torch.nn.Parameter(torch.rand(N, M))

    @torch.jit.script_method  # for tracing
    def forward(self, input):
        # if input.sum() > 0:
        if bool(input.sum() > 0):
            output = self.weight.mv(input)
        else:
            output = self.weight + input
        return output


def main():
    # generate script module via tracing
    model = torchvision.models.resnet18()
    input = torch.rand(1, 3, 224, 224)
    traced_module = torch.jit.trace(model, input)

    # traced module can be evaluated identically to a regular pytorch module
    output = traced_module(torch.ones(1, 3, 224, 224))
    print(f'outputs: {output[0, :5]}')

    # save script module to file
    traced_module.save('../../temp/model01.pt')

    # user-defined model
    my_model = MyModule(2, 3)
    my_model.save('../../temp/model02.pt')


if __name__ == '__main__':
    main()
