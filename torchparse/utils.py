import torch, ast


def padding_type(h, config):
    ret = None
    if 'padding' not in config:
        return 0
    elif isinstance(config['padding'], (tuple, list)):
        ret = torch.tensor(config['padding'])
    elif config['padding'] == 'same':

        k = torch.tensor(config['kernel_size'])
        s = torch.tensor(config['stride'])

        ret = (h*(s-1)-1+k)//2

    elif config['padding'] == 'valid':
        ret = torch.zeros(h.shape).long()
    else:
        raise ValueError('Pad type is invalid')
    return tuple(ret.numpy())

def safe_conversion(value):
    try:
        return ast.literal_eval(value)
    except ValueError:
        return value
    
def out_conv2d(spatial, config):
    p, k, s = [config[k] 
            for k in ['padding', 'kernel_size', 'stride']]
    p2 = p if isinstance(p, int) else p[0] + p[1]

    return (spatial + p2 - k)//s + 1


def format_repeats(file):
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


    