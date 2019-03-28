import torch
from torchparse import parse_cfg
cfg = """
    [convs_module]
        [conv2d]
            out_channels=32
            kernel_size=3
            stride=1
            padding=valid
        [batchnorm2d]
        [elu]
        [maxpool2d]
            kernel_size=3
            stride=3

        REPEATx2
            [conv2d]
                out_channels=64
                kernel_size=3
                stride=1
                padding=valid
            [batchnorm2d]
            [elu]
            [maxpool2d]
                kernel_size=4
                stride=4
        END
        
    [moddims]
        transpose=[2,0,1]
        collapse=[1,2]

    [recur_module]
        [lstm]
            hidden_size = 64
            num_layers = 2

    [moddims]
        drop=[0]

    [dense_module]
        [batchnorm1d]
        [linear]
            out_features = 10
    """


model = parse_cfg(cfg, [3,200,300])
print(model)

# Will output
'''
ModuleDict(
  (convs): Sequential(
    (conv2d_0): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1))
    (batchnorm2d_0): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (elu_0): ELU(alpha=1.0)
    (maxpool2d_0): MaxPool2d(kernel_size=3, stride=3, padding=0, dilation=1, ceil_mode=False)
    (conv2d_1): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
    (batchnorm2d_1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (elu_1): ELU(alpha=1.0)
    (maxpool2d_1): MaxPool2d(kernel_size=4, stride=4, padding=0, dilation=1, ceil_mode=False)
    (conv2d_2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
    (batchnorm2d_2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (elu_2): ELU(alpha=1.0)
    (maxpool2d_2): MaxPool2d(kernel_size=4, stride=4, padding=0, dilation=1, ceil_mode=False)
  )
  (recur): LSTM(192, 64, num_layers=2)
  (dense): Sequential(
    (batchnorm1d_0): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (linear_0): Linear(in_features=64, out_features=10, bias=True)
  )
)
'''

x = torch.randn(16, 3,200,300)
x = model['convs'](x)
x = x.permute(0,3,1,2)
    
batch, f = x.size()[:2]
x = x.view(batch, f, -1)
x = model['recur'](x)[0]

# many to one rnn
x = x[:,-1]
x = model['dense'](x)


print(x.shape)

'''
torch.Size([16, 10])
'''

