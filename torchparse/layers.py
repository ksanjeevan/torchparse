import torch
import torch.nn as nn

from .base_layers import *


class Unperturbed(Layer):
    """
    Handles layers with no effect where input_shape==output_shape.
    Thin wrapper around Layer.
    """
    layer = {
            'relu' : nn.ReLU,
            'elu' : nn.ELU,
            'leakyrelu':nn.LeakyReLU,
            'sigmoid':nn.Sigmoid,
            'dropout':nn.Dropout
        }

    def __init__(self, config, in_shape, name=''):
        super(Unperturbed, self).__init__(config, in_shape)
        self.name = name

    def get_out_shape(self):
        out_shape = self.in_shape.clone()
        return out_shape

    def get_module(self):
        return Unperturbed.layer[self.name](**self.config)


class BatchNorm(Layer):
    """
    Handle shape transformation and correct 
    class instantiation for BatchNorm layers.
    """
    layer = {
            'batchnorm1d':nn.BatchNorm1d,
            'batchnorm2d':nn.BatchNorm2d
        }

    def __init__(self, config, in_shape, name=''):
        super(BatchNorm, self).__init__(config, in_shape)
        self.name = name

    def get_out_shape(self):
        out_shape = self.in_shape.clone()
        return out_shape

    def _get_channel(self):
        value = None
        if self.name == 'batchnorm2d':
            value = self.in_shape[0].item()
        elif self.name == 'batchnorm1d':
            if self.in_shape.size(0) == 1:
                value = self.in_shape[-1].item()
            elif self.in_shape.size(0) == 2:
                value = self.in_shape[1].item()
            else:
                raise ValueError('Incorrect dim for batchnorm1d')
        else:
            raise ValueError('Pass an implemented type of batchnorm')
        return {'num_features':value}

    def get_module(self):
        channel = self._get_channel()
        return BatchNorm.layer[self.name](**{**self.config, **channel})

class ModDims(Layer):
    """
    Handle shape changes due to tensor manipulation.

    transpose -> transpose/permute dims
        e.g.: transpose = [2,0] -> tensor([3,20,30]) becomes tensor([30,20,3])
    collapse -> combine two dimensions into one
        e.g.: collapse = [1,2] -> tensor([3,20,30]) becomes tensor([3,600]) 
    drop -> drop dims
        e.g.: drop = [0] for many-to-one RNN -> tensor([20,30]) becomes tensor([30]) 

    Operations are applied in order: 
            Transpose -> Collapse -> Drop
    although if different order is needed it can be defined in the cfg file e.g.:
        [moddims]
            collapse=[1,2]
        [moddims]
            transpose=[0,2]
    """

    def __init__(self, config, in_shape):
        super(ModDims, self).__init__(config, in_shape)

    def _perm_dims(self, out_shape):
        if 'transpose' in self.config:
            out_shape = out_shape[self.config['transpose']]
        return out_shape

    def _coll_dims(self, out_shape):
        if 'collapse' in self.config:
            coll_idx = self.config['collapse']
            disap_idx = coll_idx[1:]
            final_idx = [i for i in range(out_shape.size(0)) if i not in disap_idx]
            out_shape[coll_idx[0]] = torch.prod(out_shape[coll_idx])
            out_shape = out_shape[final_idx]
        return out_shape 

    def _drop_dims(self, out_shape):
        if 'drop' in self.config:
            disap_idx = self.config['drop']
            final_idx = [i for i in range(out_shape.size(0)) if i not in disap_idx]
            out_shape = out_shape[final_idx]
        return out_shape 

    def get_out_shape(self):
        out_shape = self.in_shape.clone()
        out_shape = self._perm_dims(out_shape)
        out_shape = self._coll_dims(out_shape)
        out_shape = self._drop_dims(out_shape)
        return out_shape

    def get_module(self):
        return None


class Linear(Dense):
    """
    Handle shape transformation and correct 
    class instantiation for nn.Linear.
    """
    def __init__(self, config, in_shape):
        super(Linear, self).__init__(config, in_shape)
        self.changed_feat = 'out_features'
        self.config['in_features'] = self.in_shape[-1].item()

    def get_module(self):
        return nn.Linear(**self.config)


class Recurrent(Dense):
    """
    Handle shape transformation and correct 
    class instantiation for Recurrent layers.
    """
    layer = {
                'lstm' : nn.LSTM,
                'gru' : nn.GRU,
                'rnn' : nn.RNN
        }

    def __init__(self, config, in_shape, name=''):

        super(Recurrent, self).__init__(config, in_shape)
        assert self.in_shape.size(0) == 2, "in_shape must be of dim 2"
        self.changed_feat = 'hidden_size'
        self.config['input_size'] = self.in_shape[-1].item()

        self.name = name

    def get_module(self):
        return Recurrent.layer[self.name](**self.config)


class Conv2d(ConvolveSpatial):
    """
    Handle shape transformation and correct 
    class instantiation for nn.Conv2d
    (should this be general for conv1d,3d?)
    """
    def __init__(self, config, in_shape):
        super(Conv2d, self).__init__(config, in_shape)
        self.config['in_channels'] = self.in_shape[0].item()
    
    def _channel_transf(self):
        return torch.tensor([self.config['out_channels']])

    def get_module(self):
        return nn.Conv2d(**self.config)

class MaxPool2d(ConvolveSpatial):
    """
    Handle shape transformation and correct 
    class instantiation for nn.MaxPool2d
    (should this be general for adaptive, avgpool, 1d,3d?)
    """
    def __init__(self, config, in_shape):
        super(MaxPool2d, self).__init__(config, in_shape)
    
    def _channel_transf(self):
        return self.in_shape[:1]

    def get_module(self):
        return nn.MaxPool2d(**self.config)