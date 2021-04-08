import torch
import torch.nn as nn
from time import time
import os
import sys
sys.path.append("..")
from utils import data_loader,utils

class Interaction(nn.Module):
    def __init__(self, s_padding, t_padding):
        super(Interaction, self).__init__()
        self.ms = torch.nn.ZeroPad2d((0,0,0,s_padding))
        self.mt = torch.nn.ZeroPad2d((0,0,0,t_padding))
       
    def forward(self, s, t):
        #s,t: B*L*E*N
        s_imaginary=s*0
        t_imaginary=t*0
        fs=torch.stack((s,s_imaginary),4)
        ft=torch.stack((t,t_imaginary),4)
        ft=self.mt(ft)
        fs=self.ms(fs)
        fs=torch.Tensor.fft(fs,1)
        ft=torch.Tensor.fft(ft,1)
        rr=torch.Tensor.mul(fs[:,:,:,:,0],ft[:,:,:,:,0])
        ii=torch.Tensor.mul(fs[:,:,:,:,1],ft[:,:,:,:,1])
        ri=torch.Tensor.mul(fs[:,:,:,:,0],ft[:,:,:,:,1])
        ir=torch.Tensor.mul(fs[:,:,:,:,1],ft[:,:,:,:,0])
        h=torch.stack((rr-ii,ri+ir),axis=4)
        h=torch.Tensor.ifft(h,1)
        h=h[:,:,:,:,0] #B*L*E*(Is+It-1)
        h=h.permute(0,1,3,2) #B*L*(Is+It-1)*E
        return h
        
device = torch.device("cuda:0")
#s=torch.ones(128,16,3,128).to(device)
#t=torch.ones(128,16,4,128).to(device)
'''
print(0)
for i in range(50):
    length = s.shape[-2] + t.shape[-2]
    s = torch.fft.fft(s, n=length,dim=2)
    t = torch.fft.fft(t, n=length,dim=2)
    h = s*t
    h = torch.fft.ifft(h,dim=2)
    h = h.float() #B*L*(Is+It-1)*E
print(1) 
'''

s=torch.ones(128,16,128,3).to(device)
t=torch.ones(128,16,128,4).to(device)

t0=time()
length = s.shape[-1] + t.shape[-1]
s = torch.fft.fft(s, n=length)
t = torch.fft.fft(t, n=length)
h = s*t
h = torch.fft.ifft(h)
h = h.permute(0,1,3,2).float() 
t1 = time()
print(t1-t0)

'''
t0 = time()
ms = torch.nn.ZeroPad2d((0,0,0,4))
mt = torch.nn.ZeroPad2d((0,0,0,3))
s_imaginary=s*0
t_imaginary=t*0
fs=torch.stack((s,s_imaginary),4)
ft=torch.stack((t,t_imaginary),4)

ft=mt(ft)
fs=ms(fs)
fs=torch.Tensor.fft(fs,1)
ft=torch.Tensor.fft(ft,1)
rr=torch.Tensor.mul(fs[:,:,:,:,0],ft[:,:,:,:,0])
ii=torch.Tensor.mul(fs[:,:,:,:,1],ft[:,:,:,:,1])
ri=torch.Tensor.mul(fs[:,:,:,:,0],ft[:,:,:,:,1])
ir=torch.Tensor.mul(fs[:,:,:,:,1],ft[:,:,:,:,0])
h=torch.stack((rr-ii,ri+ir),axis=4)
h=torch.Tensor.ifft(h,1)
h=h[:,:,:,:,0] #B*L*E*(Is+It-1)
h=h.permute(0,1,3,2) #B*L*(Is+It-1)*E
t1 = time()
print(t1-t0)
'''
'''
atn_heads=3
in_size=128
out_size=128

t0=time()
initializer = nn.init.xavier_uniform_
ws1= nn.Parameter(utils.glorot((atn_heads, in_size, out_size)))
ws12= nn.Parameter(utils.glorot((atn_heads, in_size, out_size)))
ws13= nn.Parameter(utils.glorot((atn_heads, in_size, out_size)))
ws14= nn.Parameter(utils.glorot((atn_heads, in_size, out_size)))
ws15= nn.Parameter(utils.glorot((atn_heads, in_size, out_size)))
ws11= nn.Parameter(utils.glorot((atn_heads, in_size, out_size)))
ws17= nn.Parameter(utils.glorot((atn_heads, in_size, out_size)))
ws176= nn.Parameter(utils.glorot((atn_heads, in_size, out_size)))
ws144= nn.Parameter(utils.glorot((atn_heads, in_size, out_size)))
ws134= nn.Parameter(utils.glorot((atn_heads, in_size, out_size)))
ws123= nn.Parameter(utils.glorot((atn_heads, in_size, out_size)))
t1 = time()
print(t1-t0)
'''

#user_user
            user_user_src=[]
            user_user_dst=[]
            with open(self._data + '/' + _data_list[0]) as fin:
                for line in fin.readlines():
                    _line = line.strip().split(',')
                    _user1, _user2= int(_line[0]), int(_line[1])
                    user_user_src.append(_user1)
                    user_user_dst.append(_user2)