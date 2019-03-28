import torch, ast


def padding_type(spatial, config):
    """
    Apply type padding in a convolution operation.
    Arguments:
        spatial (tensor): spatial dimensions.
        config (dict): module parameters.
            config['padding'] == 'same' -> padding should be such 
                that spatial dimensions remainthe same
            config['padding'] == 'valid' -> padding is 0
    """
    ret = None
    if 'padding' not in config:
        return 0
    elif isinstance(config['padding'], (tuple, list)):
        ret = torch.tensor(config['padding'])
    elif config['padding'] == 'same':

        k = torch.tensor(config['kernel_size'])
        s = torch.tensor(config['stride'])

        ret = (spatial*(s-1)-1+k)//2

    elif config['padding'] == 'valid':
        ret = torch.zeros(spatial.shape).long()
    else:
        raise ValueError('Pad type is invalid')
    return tuple(ret.numpy())

def safe_conversion(value):
    """
    Safely convert parameter value from .cfg file
    """
    try:
        return ast.literal_eval(value)
    except ValueError:
        return value
    
def out_conv2d(spatial, config):
    """
    Calculate spatial output shape after convolution. 
    Arguments:
        spatial (tensor): spatial dimensions.
        config (dict): module parameters.
    """
    p, k, s = [config[k] 
            for k in ['padding', 'kernel_size', 'stride']]
    p2 = p if isinstance(p, int) else p[0] + p[1]

    return (spatial + p2 - k)//s + 1


def format_repeats(file):
    """
    Deal with the repeating blocks in the model
    by keeping track of the encompassed rows and
    multiplying them before appending.
    """
    ret = []
    while True:
        try:
            l = next(file).lstrip().replace('\n','')
        except StopIteration:
            break
        if l.lower().startswith('repeat'):
            times = int(l.split('x')[1])
            repeats = []
            while True:
                l = next(file).lstrip().replace('\n','')
                if l.lower() == 'end':
                    break
                repeats.append(l)
            ret += repeats*times
        else:
            if not l.startswith('#'):
                ret += [l]
    return ret


def add_counters(dic, arr):
    """
    Keep track of how many times a type of layer has appeard and
    append _counter to their name to maintain module name uniqueness.
    """
    ret = []
    for el in arr:
        name = el[1:-1]
        num = dic.get(name, None)
        if num is not None:
            ret.append('[%s_%s]'%(name, dic[name]))
            dic[name] += 1
        else:
            ret.append(el)
    return ret

def defined_submodule(arr):
    """
    Check if model uses submodules
    """
    return any([el.endswith('_module]') for el in arr])


def get_sample_cfg():
    return """
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
        permute=[2,0,1]
        collapse=[1,2]

    [recur_module]
        [lstm]
            hidden_size = 64
            num_layers = 2

    [moddims]
        permute=[1]

    [dense_module]
        [batchnorm1d]
        [linear]
            out_features = 10

    """


    