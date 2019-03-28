import torch
import torch.nn as nn
from collections import OrderedDict
import configparser, inspect, io, os

from .utils import add_counters, format_repeats, defined_submodule
from . import layers

def get_implem_layers():

    ret = {}
    names = [m[0] for m in inspect.getmembers(layers, inspect.isclass) if m[1].__module__ == layers.__name__]
    for n in names:
        l = getattr(layers, n)
        if hasattr(l, 'layer'):
            for sub_n in l.layer.keys():
                ret[sub_n] = l
        else:
            ret[n.lower()] = l
    return ret

class CFGParser(object):

    def __init__(self, cfg_fname):
        self.layers = get_implem_layers()

        self.config = configparser.ConfigParser(strict=False,
            allow_no_value=True)

        self.config.read_string(self._preparse(cfg_fname))
    

    def _preparse(self, cfg_fname):

        names = self.layers.keys()
        counters = dict(zip(names, [0]*len(names)))

        if os.path.isfile(cfg_fname):
            f = open(cfg_fname, 'r')
        else:
            f = io.StringIO(cfg_fname)
        new_rows = format_repeats(f)

        if not defined_submodule(new_rows):
            new_rows = ['[main_module]'] + new_rows

        out = add_counters(counters, new_rows)

        return '\n'.join(out)


    def _get_layer(self, config, in_shape, name):

        layer = self.layers[name](config, in_shape)
        if hasattr(layer, 'name'):
            layer.name = name
        return layer

    def _flow(self, in_shape):

        ret = OrderedDict()
        for l in self.config.sections():
            try:
                name, num = l.split('_')
            except: 
                raise ValueError('Not yet implemented layer: %s'%l)

            if num == 'module':
                ret[name] = OrderedDict()    
            else:
                layer = self._get_layer(self.config[l], in_shape, name)

                in_shape = layer.get_out_shape()
                module = layer.get_module()
                if module is not None:
                    ret[next(reversed(ret))][l] = module
        return ret


    def get_modules(self, in_shape):

        model = self._flow(in_shape)
        ret = OrderedDict()
        for subm_name, subm_od in model.items():
            if len(subm_od) == 1:
                ret[subm_name] = subm_od[next(iter(subm_od))]
            else:
                ret[subm_name] = nn.Sequential(subm_od)

        return nn.ModuleDict(ret)
        

def parse_cfg(fname, shape):
    return CFGParser(fname).get_modules(torch.tensor(shape))

if __name__ == '__main__':

    pass








    