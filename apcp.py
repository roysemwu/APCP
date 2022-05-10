import torch
import torch.nn as nn
from torch.autograd import Variable
import math


class GAP(torch.autograd.Function):
  @staticmethod
  def forward(ctx, input):
    ctx.save_for_backword(input)
    input = torch.mean(input, dim=0, keepdim=True)
    return input
  
  @staticmethod
  def backward(ctx, grad_output):
    input = ctx.saved_tensors
    grad_input = input[0].clone()
    for i in range(grad_input.shape[0]):
      grad_input[i, :, :, :] = grad_output.data / grad_output.shape[0]
    return Variable(grad_input)


class APCP(nn.Module):
  def __init__(self):
    super(APCP, self).__init__()
    self.sigmoid = nn.Sigmoid()
    self.epsilon = 1e-5
    # self.tanh = nn.Tanh()
    
  def forward(self, x, channel_index=None):
  
    x_hat = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
    e_ss = x_hat / (x_hat.sum() / (x.size[2]*x.size[3]-1) + self.epsilon)
    x = x * self.sigmoid(e_ss)
    
    x_aver = GAP.apply(x)
    val = (x_aver.pow(2).sum((2,3), keepdim=True) + self.epsilon).pow(0.5)
    val_hat = val / (val.pow(2).mean(dim=1, keepdim=True) + self.epsilon).pow(0.5)
    scale = self.sigmoid(torch.squeeze(val_hat))
    
    return x, scale
