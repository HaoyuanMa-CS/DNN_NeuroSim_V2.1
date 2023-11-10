import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
import math as m
from scipy.optimize import fsolve

def shift(x):
    #TODO: edge case, when x contains 0
    return 2.**torch.round(torch.log2(x))

def S(bits):
    return 2.**(bits-1)

def SR(x):
    r = torch.FloatTensor(*x.size()).uniform_()
    return torch.floor(x+r)

def C(x, bits):
    if bits > 15 or bits == 1:
        delta = 0
    else:
        delta = 1. / S(bits)
    upper = 1  - delta
    lower = -1 + delta
    return torch.clamp(x, lower, upper)

def QG(origin, x, bits_W=5, bits_G=5, lr=1, maxLevelLTP=32, maxLevelLTD=32):
    max_entry = x.abs().max()
    assert max_entry != 0, "QG blow"
    #if max_entry != 0:
    x /= shift(max_entry)
    gradient = lr * x
    # introduce non-linearity here
    numLevel = max(maxLevelLTP, maxLevelLTD)
    # apply delta pulse to old weight
    deltaPulse = torch.round((gradient)/2*numLevel)
    #ltpd = torch.where(torch.sign(deltaPulse)<0, 1.0, -1.0).float() #1.0 -- LTP, -1.0 -- LTD
    xPulse = InvNLS(origin, initial_guess=0.5)*numLevel
    #weightNew = torch.where(
    #    torch.sign(deltaPulse)<0, 
    #    NLS((xPulse-deltaPulse)/numLevel, ltpd=0, bias=torch.tensor(0)),
    #    NLS(((numLevel-xPulse)-deltaPulse)/numLevel, ltpd=0, bias=torch.tensor(0)))
    weightNew = NLS((xPulse - deltaPulse)/numLevel, torch.tensor(0))
    gradient = origin - C(weightNew, bits_W)
    norm = SR(gradient)  # normalize the gradient
    gradient = norm / S(bits_G)
    return gradient

def NLS(xPulse, bias):
    #a_P, c_P, d_P, g_P = 0.9909, 0.9281, 1.559, -2.477
    #a_D, c_D, d_D, g_D = 0.9882, 0.9468, 1.48, -2.363
    #a, c, d, g = (a_P + a_D) / 2, (c_P + c_D) / 2, (d_P + d_D) / 2, (g_P + g_D) / 2
    a, c, d, g = torch.tensor(0.98955), torch.tensor(0.93745), torch.tensor(1.5195), torch.tensor(-2.4200)
    #if ltpd < 0:
    #    xPulse = 1.0 - xPulse
    return a * (1 - torch.exp(-(xPulse / (d * torch.exp((g * xPulse)))) ** c)) - bias
    
def InvNLS(weights, initial_guess):
    #assert ltpd.shape == weights.shape, "ERR: Different shape between LTPD and weight arrays"
    #ltpd = torch.flatten(ltpd)
    initial_guess = torch.tensor(initial_guess)
    xPulse = []
    length = len(weights.flatten())
    for i in range(length):
        xPulse.append(fsolve(NLS, initial_guess, args=(weights.flatten()[i],)))
    return torch.tensor(np.array(xPulse).reshape(weights.shape))

class WAGERounding(Function):
    @staticmethod
    def forward(self, x, bits_A, bits_E, optional):
        self.optional = optional
        self.bits_E = bits_E
        self.save_for_backward(x)
        if bits_A == -1: ret = x
        else: ret = Q(x, bits_A)

        return ret

    @staticmethod
    def backward(self, grad_output):
        if self.bits_E == -1: return grad_output, None, None, None

        if self.needs_input_grad[0]:
            try:
                grad_input = QE(grad_output, self.bits_E)
            except AssertionError as e:
                print("="*80)
                print("Error backward:%s"%self.optional)
                print("-"*80)
                print(grad_output.max())
                print(grad_output.min())
                print("="*80)
                raise e
        else:
            grad_input = grad_output

        return grad_input, None, None, None

class WAGERounding_forward(Function):
    @staticmethod
    def forward(self, x, bits_A, bits_E, optional):
        self.optional = optional
        self.bits_E = bits_E
        self.save_for_backward(x)
        if bits_A == -1: ret = x
        else: ret = Q(x, bits_A)

        return ret

    @staticmethod
    def backward(self, grad_output):
        return grad_output, None, None, None

quantize_wage = WAGERounding.apply

class WAGEQuantizer(Module):
    def __init__(self, bits_A, bits_E, name="", writer=None):
        super(WAGEQuantizer, self).__init__()
        self.bits_A = bits_A
        self.bits_E = bits_E
        self.name = name
        self.writer = writer

    def forward(self, x):
        if self.bits_A != -1:
            x = C(x, self.bits_A) #  keeps the gradients
        #print(x.std())
        y = quantize_wage(x, self.bits_A, self.bits_E, self.name)
        if self.writer is not None:
            self.writer.add_histogram(
                    "activation-before/%s"%self.name, x.clone().cpu().data.numpy())
            self.writer.add_histogram(
                    "activation-after/%s"%self.name, y.clone().cpu().data.numpy())
        return y

def WAGEQuantizer_f(x, bits_A, bits_E, name=""):
        if bits_A != -1:
            x = C(x, bits_A) #  keeps the gradients
        y = quantize_wage(x, bits_A, bits_E, name)
        return y

if __name__ == "__main__":
    import time
    import numpy as np
    import json
    
    with open(r"Training_pytorch\utee\variables.json") as v:
        G_grad = json.load(v)
    print("="*80)
    print("Gradient")
    print("="*80)
    start_time = time.time()
    quant_data = QG(torch.tensor(G_grad['G']), torch.tensor(G_grad['grad'])).data.numpy()
    print("--- %s seconds ---" % (time.time() - start_time))
    print(quant_data)
