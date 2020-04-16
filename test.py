import torch


a = torch.ones((128,20,64))
b = torch.ones((128,64))
c = b.repeat(20,1,1)
print(c.size())
print(c.permute(1,0,2).size())



b = torch.ones(1)
f = b.repeat(2,5)
print(f.size())