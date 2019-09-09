import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

x = torch.Tensor([[1, 2], [3, 4]])
x = Variable(x, requires_grad=True)

m = nn.Linear(2, 1)
torch.nn.init.constant_(m.weight.data, 2)
torch.nn.init.constant_(m.bias,1)
print('m.weight',m.weight)
print('m.bias',m.bias)


out = m(x).sum()
print('out', out)
out.backward()
print('\n\nbbbbbbbbbbbbbackward')
print('weight_grad', m.weight.grad)
print('x.grad', x.grad)


optimizer = torch.optim.SGD(m.parameters(),  lr = 1)
optimizer.step()
print('ssssssssssstttttttttttttteeeeeeeeeppppppp')
print('m.weight', m.weight)
print('m.bias', m.bias)
print('x', x)
optimizer.zero_grad()
print('zerooooooooooooooooooooooooo')
print('x.grad22222', x.grad)
print('m.weight22222.grad', m.weight.grad)
print('\n\n********************')
print(m(x))

exit()