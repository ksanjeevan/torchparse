import torch
import torch.nn as nn
from collections import OrderedDict
import configparser, inspect, io, os

from .utils import add_counters, format_repeats, defined_submodule, safe_conversion
from . import layers

def get_implem_layers():
    """
    Get all the layer types. If the layer type is responsible for > 1 nn.Module then
    add an entry with for each appropiate name in the dictionary.
    e.g.: {'lstm':torchparse.layers.Recurrent, 'gru':torchparse.layers.Recurrent} 
    """
    ret = {}
    names = [m[0] for m in inspect.getmembers(layers, inspect.isclass) \
            if m[1].__module__ == layers.__name__]
    for n in names:
        l = getattr(layers, n)
        if hasattr(l, 'layer'):
            for sub_n in l.layer.keys():
                ret[sub_n] = l
        else:
            ret[n.lower()] = l
    return ret

class CFGParser(object):
    """
    Handles parsing .cfg and returning the defined model as an nn.ModuleDict.
    """

    def __init__(self, cfg_fname):
        self.layers = get_implem_layers()

        self.config = configparser.ConfigParser(strict=False,
                                                allow_no_value=True)

        self.config.read_string(self._preparse(cfg_fname))
    

    def _preparse(self, cfg_fname):
        """
        Format/preprocess cfg before starting to create modules.
        """

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
        """
        Get the defined torchparse.layers object depending
        on the name of the desired layer.
        """
        layer = self.layers[name](config, in_shape)
        if hasattr(layer, 'name'):
            layer.name = name
        return layer


    def _extract_input(self, in_shape):
        
        if self.config.has_section('input'):
            in_shape = safe_conversion(self.config['input']['shape'])
            self.config.remove_section('input')

        if in_shape is not None:
            in_shape = torch.tensor(in_shape)

        return in_shape


    def _flow(self, in_shape=None):
        """
        Given a input shape in_shape, *sequantially* go through
        the cfg file keeping track of the intermediate shapes and
        storing each layer in the appropiate submodule.
        """
        in_shape = self._extract_input(in_shape)
        
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


    def get_modules(self, in_shape=None):
        """
        Wrapper for _flow, prunes the submodules in case they are
        nn.Sequential of size 1.
        """
        model = self._flow(in_shape)
        ret = OrderedDict()
        for subm_name, subm_od in model.items():
            if len(subm_od) == 1:
                ret[subm_name] = subm_od[next(iter(subm_od))]
            else:
                ret[subm_name] = nn.Sequential(subm_od)

        return nn.ModuleDict(ret)
        

def parse_cfg(fname, in_shape=None):
    """
    Get the defined model given an input shape.
    Arguments:
        fname (str): either path to .cfg file or raw cfg string.
        shape (list, tuple, torch.tensor): shape (without batch)
            of the input data to the model.
    """
    return CFGParser(fname).get_modules(in_shape)