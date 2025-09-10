import sys
from functools import partial

import numpy as np

import torch
import torch.nn as nn
try:
    from spikingjelly.clock_driven import surrogate, neuron, functional
except:
    from spikingjelly.activation_based import surrogate, neuron, functional

from snn.multi_step_layers import *

# from .ops import CUSTOM_MODULES_MAPPING, MODULES_MAPPING
# from .utils import syops_to_string, params_to_string

# from timm.utils import *
# from timm.utils.metrics import *  # AverageMeter, accuracy


def add_syops_counting_methods(net_main_module):
    # adding additional methods to the existing module object,
    # this is done this way so that each function has access to self object
    net_main_module.start_syops_count = start_syops_count.__get__(net_main_module)
    net_main_module.stop_syops_count = stop_syops_count.__get__(net_main_module)
    net_main_module.reset_syops_count = reset_syops_count.__get__(net_main_module)
    net_main_module.compute_average_syops_cost = compute_average_syops_cost.__get__(
                                                    net_main_module)

    net_main_module.reset_syops_count()

    return net_main_module

def accumulate_syops(self):
    if is_supported_instance(self):
        return self.__syops__
    else:
        sum = np.array([0.0, 0.0, 0.0, 0.0])
        for m in self.children():
            sum += m.accumulate_syops()
        return sum

def print_model_with_syops(model, total_syops, total_params, syops_units='GMac',
                           param_units='M', precision=3, ost=sys.stdout):

    for i in range(3):
        if total_syops[i] < 1:
            total_syops[i] = 1
    if total_params < 1:
        total_params = 1

    def accumulate_params(self):
        if is_supported_instance(self):
            return self.__params__
        else:
            sum = 0
            for m in self.children():
                sum += m.accumulate_params()
            return sum

    def syops_repr(self):
        accumulated_params_num = self.accumulate_params()
        accumulated_syops_cost = self.accumulate_syops()
        accumulated_syops_cost[0] /= model.__batch_counter__
        accumulated_syops_cost[1] /= model.__batch_counter__
        accumulated_syops_cost[2] /= model.__batch_counter__
        accumulated_syops_cost[3] /= model.__times_counter__

        # store info for later analysis
        self.accumulated_params_num = accumulated_params_num
        self.accumulated_syops_cost = accumulated_syops_cost

        return ', '.join([self.original_extra_repr(),
                          params_to_string(accumulated_params_num,
                                           units=param_units, precision=precision),
                          '{:.3%} Params'.format(accumulated_params_num / total_params),
                          syops_to_string(accumulated_syops_cost[0],
                                          units=syops_units, precision=precision),
                          '{:.3%} oriMACs'.format(accumulated_syops_cost[0] / total_syops[0]),
                          syops_to_string(accumulated_syops_cost[1],
                                          units=syops_units, precision=precision),
                          '{:.3%} ACs'.format(accumulated_syops_cost[1] / total_syops[1]),
                          syops_to_string(accumulated_syops_cost[2],
                                          units=syops_units, precision=precision),
                          '{:.3%} MACs'.format(accumulated_syops_cost[2] / total_syops[2]),
                          '{:.3%} Spike Rate'.format(accumulated_syops_cost[3] / 100.),
                          'SpkStat: {}'.format(self.__spkhistc__)])  # print self.__spkhistc__
                          #self.original_extra_repr()])


    def syops_repr_empty(self):
        return ''

    def add_extra_repr(m):
        m.accumulate_syops = accumulate_syops.__get__(m)
        m.accumulate_params = accumulate_params.__get__(m)
        if is_supported_instance(m):
            syops_extra_repr = syops_repr.__get__(m)
        else:
            syops_extra_repr = syops_repr_empty.__get__(m)
        if m.extra_repr != syops_extra_repr:
            m.original_extra_repr = m.extra_repr
            m.extra_repr = syops_extra_repr
            assert m.extra_repr != m.original_extra_repr

    def del_extra_repr(m):
        if hasattr(m, 'original_extra_repr'):
            m.extra_repr = m.original_extra_repr
            del m.original_extra_repr
        if hasattr(m, 'accumulate_syops'):
            del m.accumulate_syops

    model.apply(add_extra_repr)
    print(repr(model), file=ost)
    model.apply(del_extra_repr)

def get_model_parameters_number(model):
    params_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params_num

def compute_average_syops_cost(self):
    """
    A method that will be available after add_syops_counting_methods() is called
    on a desired net object.

    Returns current mean syops consumption per image.

    """

    for m in self.modules():
        m.accumulate_syops = accumulate_syops.__get__(m)
        # print(m)

    syops_sum = self.accumulate_syops()
    syops_sum = np.array([item / self.__batch_counter__ for item in syops_sum])

    for m in self.modules():
        if hasattr(m, 'accumulate_syops'):
            del m.accumulate_syops

    params_sum = get_model_parameters_number(self)
    return syops_sum, params_sum

def get_syops_model(model,pre_name='',buf={}):
    for name, module in model._modules.items():
        if hasattr(module,"_modules"):
            get_syops_model(module, pre_name=pre_name+'.'+name, buf=buf)
        
        if module.__class__.__name__ == 'msConv': 
            # print(pre_name+'.'+name)
            # print(module.__syops__)
            buf[pre_name[1:]+'.'+name+'.msConv']=module.__syops__
        elif module.__class__.__name__ == 'msAvgPool2d': 
            # print(pre_name+'.'+name)
            # print(module.__syops__)
            buf[pre_name[1:]+'.'+name+'.msAvgPool2d']=module.__syops__
        elif module.__class__.__name__ == 'IFNode': 
            # print(pre_name+'.'+name)
            # print(module.__syops__)
            buf[pre_name[1:]+'.'+name+'.IFNode']=module.__syops__
        elif module.__class__.__name__ == 'Conv2d': 
            # print(pre_name+'.'+name)
            # print(module.__syops__)
            buf[pre_name[1:]+'.'+name+'.Conv2d']=module.__syops__    
        elif module.__class__.__name__ == 'LIFNode': 
            # print(pre_name+'.'+name)
            # print(module.__syops__)
            buf[pre_name[1:]+'.'+name+'.LIFNode']=module.__syops__
        elif module.__class__.__name__.lower() == 'relu': 
            buf[pre_name[1:]+'.'+name+'.relu']=module.__syops__
        elif module.__class__.__name__ == 'AvgPool2d': 
            buf[pre_name[1:]+'.'+name+'.AvgPool2d']=module.__syops__
        elif module.__class__.__name__ == 'Linear': 
            buf[pre_name[1:]+'.'+name+'.Linear']=module.__syops__
        elif module.__class__.__name__ == 'Linear': 
            buf[pre_name[1:]+'.'+name+'.Linear']=module.__syops__
        
        # elif module.__class__.__name__ == 'Conv': 
        #     # print(pre_name+'.'+name)
        #     # print(module.__syops__)
        #     buf[pre_name[1:]+'.'+name+'.Conv']=module.__syops__

    return None
        

def start_syops_count(self, **kwargs):
    """
    A method that will be available after add_syops_counting_methods() is called
    on a desired net object.

    Activates the computation of mean syops consumption per image.
    Call it before you run the network.

    """
    add_batch_counter_hook_function(self)

    seen_types = set()

    def add_syops_counter_hook_function(module, ost, verbose, ignore_list):
        if type(module) in ignore_list:
            seen_types.add(type(module))
            if is_supported_instance(module):
                module.__params__ = 0
        elif is_supported_instance(module):
            if hasattr(module, '__syops_handle__'):
                return
            if type(module) in CUSTOM_MODULES_MAPPING:
                handle = module.register_forward_hook(
                                        CUSTOM_MODULES_MAPPING[type(module)])
            else:
                handle = module.register_forward_hook(MODULES_MAPPING[type(module)])
            module.__syops_handle__ = handle
            seen_types.add(type(module))
        else:
            if verbose and not type(module) in (nn.Sequential, nn.ModuleList) and \
               not type(module) in seen_types:
                print('Warning: module ' + type(module).__name__ +
                      ' is treated as a zero-op.', file=ost)
            seen_types.add(type(module))

    self.apply(partial(add_syops_counter_hook_function, **kwargs))


def stop_syops_count(self):
    """
    A method that will be available after add_syops_counting_methods() is called
    on a desired net object.

    Stops computing the mean syops consumption per image.
    Call whenever you want to pause the computation.

    """
    remove_batch_counter_hook_function(self)
    self.apply(remove_syops_counter_hook_function)
    # self.apply(remove_syops_counter_variables)  # keep this for later analyses


def reset_syops_count(self):
    """
    A method that will be available after add_syops_counting_methods() is called
    on a desired net object.

    Resets statistics computed so far.

    """
    add_batch_counter_variables_or_reset(self)
    self.apply(add_syops_counter_variable_or_reset)


# ---- Internal functions
def batch_counter_hook(module, input, output):
    batch_size = 1
    if len(input) > 0:
        # Can have multiple inputs, getting the first one
        input = input[0]
        batch_size = len(input)
    else:
        pass
        print('Warning! No positional inputs found for a module,'
              ' assuming batch size is 1.')
    module.__batch_counter__ += batch_size
    module.__times_counter__ += 1


def add_batch_counter_variables_or_reset(module):

    module.__batch_counter__ = 0
    module.__times_counter__ = 0


def add_batch_counter_hook_function(module):
    if hasattr(module, '__batch_counter_handle__'):
        return

    handle = module.register_forward_hook(batch_counter_hook)
    module.__batch_counter_handle__ = handle


def remove_batch_counter_hook_function(module):
    if hasattr(module, '__batch_counter_handle__'):
        module.__batch_counter_handle__.remove()
        del module.__batch_counter_handle__


def add_syops_counter_variable_or_reset(module):
    if is_supported_instance(module):
        if hasattr(module, '__syops__') or hasattr(module, '__params__'):
            print('Warning: variables __syops__ or __params__ are already '
                  'defined for the module' + type(module).__name__ +
                  ' syops can affect your code!')
            module.__syops_backup_syops__ = module.__syops__
            module.__syops_backup_params__ = module.__params__
        module.__syops__ = np.array([0.0, 0.0, 0.0, 0.0])
        module.__params__ = get_model_parameters_number(module)
        # add __spkhistc__ for each module (by yult 2023.4.18)
        module.__spkhistc__ = None #np.zeros(20)  # assuming there are no more than 20 spikes for one neuron


def is_supported_instance(module):
    if type(module) in MODULES_MAPPING or type(module) in CUSTOM_MODULES_MAPPING:
        return True
    return False


def remove_syops_counter_hook_function(module):
    if is_supported_instance(module):
        if hasattr(module, '__syops_handle__'):
            module.__syops_handle__.remove()
            del module.__syops_handle__


def remove_syops_counter_variables(module):
    if is_supported_instance(module):
        if hasattr(module, '__syops__'):
            del module.__syops__
            if hasattr(module, '__syops_backup_syops__'):
                module.__syops__ = module.__syops_backup_syops__
        if hasattr(module, '__params__'):
            del module.__params__
            if hasattr(module, '__syops_backup_params__'):
                module.__params__ = module.__syops_backup_params__
        # remove module.__spkhistc__ after print
        if hasattr(module, '__spkhistc__'):
            del module.__spkhistc__


# utils *****************************************

def syops_to_string(syops, units=None, precision=2):
    if units is None:
        if syops // 10**9 > 0:
            return str(round(syops / 10.**9, precision)) + ' G Ops'
        elif syops // 10**6 > 0:
            return str(round(syops / 10.**6, precision)) + ' M Ops'
        elif syops // 10**3 > 0:
            return str(round(syops / 10.**3, precision)) + ' K Ops'
        else:
            return str(syops) + ' Ops'
    else:
        if units == 'G Ops':
            return str(round(syops / 10.**9, precision)) + ' ' + units
        elif units == 'M Ops':
            return str(round(syops / 10.**6, precision)) + ' ' + units
        elif units == 'K Ops':
            return str(round(syops / 10.**3, precision)) + ' ' + units
        else:
            return str(syops) + ' Ops'


def params_to_string(params_num, units=None, precision=2):
    if units is None:
        if params_num // 10 ** 6 > 0:
            return str(round(params_num / 10 ** 6, precision)) + ' M'
        elif params_num // 10 ** 3:
            return str(round(params_num / 10 ** 3, precision)) + ' k'
        else:
            return str(params_num)
    else:
        if units == 'M':
            return str(round(params_num / 10.**6, precision)) + ' ' + units
        elif units == 'K':
            return str(round(params_num / 10.**3, precision)) + ' ' + units
        else:
            return str(params_num)


'''
Copyright (C) 2022 Guangyao Chen - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 * You should have received a copy of the MIT license with
 * this file. If not visit https://opensource.org/licenses/MIT
'''

import torch
import numpy as np
import torch.nn as nn
try:
    from spikingjelly.clock_driven.neuron import MultiStepIFNode, MultiStepLIFNode, IFNode, LIFNode, MultiStepParametricLIFNode, ParametricLIFNode
except:
    from spikingjelly.activation_based.neuron import MultiStepIFNode, MultiStepLIFNode, IFNode, LIFNode, MultiStepParametricLIFNode, ParametricLIFNode


def spike_rate(inp):
    Nspks_max = 30  # 例如for spikformer-8-512, real Nspks_max is 17 (2*8+1=17)；若用此计算Spikformer的能耗，则对应论文Appendix G中计算1）.
    # Nspks_max = 1  # 只有真正全0-1矩阵才作为event-driven，计算AC，否则均计算为MAC；若用此计算Spikformer的能耗，则对应论文Appendix G中计算2）.
    num = inp.unique()

    if len(num) <= Nspks_max+1 and inp.max() <= Nspks_max and inp.min() >= 0:
        spkhistc = None

        spike = True
        spike_rate = (inp.sum() / inp.numel()).item()

    else:
        spkhistc = None

        spike = False
        spike_rate = 1

    return spike, spike_rate, spkhistc

def ms_spike_rate(inp):
    Nspks_max = 32
    num = inp.unique()

    if len(num) <= Nspks_max+1 and inp.max() <= Nspks_max and inp.min() >= 0:
        spkhistc = None
        spike = True
        spike_rate = ((inp>0).sum() / inp.numel()).item()
    else:
        spkhistc = None
        spike = False
        spike_rate = 1

    return spike, spike_rate, spkhistc


def empty_syops_counter_hook(module, input, output):
    module.__syops__ += np.array([0.0, 0.0, 0.0, 0.0])


def upsample_syops_counter_hook(module, input, output):
    output_size = output[0]
    batch_size = output_size.shape[0]
    output_elements_count = batch_size
    for val in output_size.shape[1:]:
        output_elements_count *= val
    module.__syops__[0] += int(output_elements_count)

    # spike, rate = spike_rate(output[0])
    spike, rate, _ = spike_rate(output)

    if spike:
        module.__syops__[1] += int(output_elements_count) * rate
    else:
        module.__syops__[2] += int(output_elements_count)

    module.__syops__[3] += rate * 100

def relu_syops_counter_hook(module, input, output):
    active_elements_count = output.numel()
    module.__syops__[0] += int(active_elements_count)

    # spike, rate = spike_rate(output[0])
    spike, rate, _ = spike_rate(output)

    if spike:
        module.__syops__[1] += int(active_elements_count) * rate
    else:
        module.__syops__[2] += int(active_elements_count)

    module.__syops__[3] += rate * 100

def IF_syops_counter_hook(module, input, output):
    active_elements_count = input[0].numel()
    module.__syops__[0] += int(active_elements_count)

    # spike, rate = spike_rate(output[0])
    spike, rate, spkhistc = spike_rate(output)
    module.__syops__[1] += int(active_elements_count)
    module.__syops__[3] += rate * 100
    module.__spkhistc__ = spkhistc

def LIF_syops_counter_hook(module, input, output):
    active_elements_count = input[0].numel()
    module.__syops__[0] += int(active_elements_count)

    spike, rate, spkhistc = spike_rate(output)
    module.__syops__[1] += int(active_elements_count)
    # module.__syops__[2] += int(active_elements_count)
    module.__syops__[3] += rate * 100
    module.__spkhistc__ = spkhistc

def linear_syops_counter_hook(module, input, output):
    input = input[0]
    spike, rate, spkhistc = spike_rate(input)
    # pytorch checks dimensions, so here we don't care much
    batch_size = input.shape[0]
    output_last_dim = output.shape[-1]
    # bias_syops = output_last_dim if module.bias is not None else 0
    bias_syops = output_last_dim*batch_size if module.bias is not None else 0
    module.__syops__[0] += int(np.prod(input.shape) * output_last_dim + bias_syops)
    if spike:
        module.__syops__[1] += int(np.prod(input.shape) * output_last_dim + bias_syops) * rate
    else:
        module.__syops__[2] += int(np.prod(input.shape) * output_last_dim + bias_syops)

    module.__syops__[3] += rate * 100
    module.__spkhistc__ = spkhistc


def pool_syops_counter_hook(module, input, output):
    input = input[0]  # input is tuple, input[0].shape = torch.Size([4, 192, 32, 32]) [TB, C, H, W]  # output.shape = torch.Size([4, 192, 16, 16])
    spike, rate, spkhistc = spike_rate(input)
    module.__syops__[0] += int(np.prod(input.shape))

    if spike:
        module.__syops__[1] += int(np.prod(input.shape)) * rate
    else:
        module.__syops__[2] += int(np.prod(input.shape))

    module.__syops__[3] += rate * 100
    module.__spkhistc__ = spkhistc

def bn_syops_counter_hook(module, input, output):
    input = input[0]  # input is tuple, input[0].shape = torch.Size([4, 48, 32, 32]) [TB, C, H, W]
    spike, rate, spkhistc = spike_rate(input)
    batch_syops = np.prod(input.shape)
    if module.affine:
        batch_syops *= 2
    module.__syops__[0] += int(batch_syops)

    if spike:
        module.__syops__[1] += int(batch_syops) * rate
    else:
        module.__syops__[2] += int(batch_syops)

    module.__syops__[3] += rate * 100
    module.__spkhistc__ = spkhistc


def conv_syops_counter_hook(conv_module, input, output):
    # Can have multiple inputs, getting the first one
    input = input[0]  # input is tuple, input[0].shape = torch.Size([4, 3, 32, 32]) [TB, C, H, W]
    spike, rate, spkhistc = spike_rate(input)

    batch_size = input.shape[0]
    output_dims = list(output.shape[2:])  # output.shape = torch.Size([4, 48, 32, 32])

    kernel_dims = list(conv_module.kernel_size)
    in_channels = conv_module.in_channels
    out_channels = conv_module.out_channels
    groups = conv_module.groups

    filters_per_channel = out_channels // groups
    conv_per_position_syops = int(np.prod(kernel_dims)) * \
        in_channels * filters_per_channel

    active_elements_count = batch_size * int(np.prod(output_dims))

    overall_conv_syops = conv_per_position_syops * active_elements_count

    bias_syops = 0

    if conv_module.bias is not None:

        bias_syops = out_channels * active_elements_count

    overall_syops = overall_conv_syops + bias_syops

    conv_module.__syops__[0] += int(overall_syops)

    if spike:
        conv_module.__syops__[1] += int(overall_syops) * rate
    else:
        conv_module.__syops__[2] += int(overall_syops)

    conv_module.__syops__[3] += rate * 100
    conv_module.__spkhistc__ = spkhistc


def rnn_syops(syops, rnn_module, w_ih, w_hh, input_size):
    # matrix matrix mult ih state and internal state
    syops += w_ih.shape[0]*w_ih.shape[1]
    # matrix matrix mult hh state and internal state
    syops += w_hh.shape[0]*w_hh.shape[1]
    if isinstance(rnn_module, (nn.RNN, nn.RNNCell)):
        # add both operations
        syops += rnn_module.hidden_size
    elif isinstance(rnn_module, (nn.GRU, nn.GRUCell)):
        # hadamard of r
        syops += rnn_module.hidden_size
        # adding operations from both states
        syops += rnn_module.hidden_size*3
        # last two hadamard product and add
        syops += rnn_module.hidden_size*3
    elif isinstance(rnn_module, (nn.LSTM, nn.LSTMCell)):
        # adding operations from both states
        syops += rnn_module.hidden_size*4
        # two hadamard product and add for C state
        syops += rnn_module.hidden_size + rnn_module.hidden_size + rnn_module.hidden_size
        # final hadamard
        syops += rnn_module.hidden_size + rnn_module.hidden_size + rnn_module.hidden_size
    return syops


def rnn_syops_counter_hook(rnn_module, input, output):
    """
    Takes into account batch goes at first position, contrary
    to pytorch common rule (but actually it doesn't matter).
    If sigmoid and tanh are hard, only a comparison syops should be accurate
    """
    syops = 0
    # input is a tuple containing a sequence to process and (optionally) hidden state
    inp = input[0]
    batch_size = inp.shape[0]
    seq_length = inp.shape[1]
    num_layers = rnn_module.num_layers

    for i in range(num_layers):
        w_ih = rnn_module.__getattr__('weight_ih_l' + str(i))
        w_hh = rnn_module.__getattr__('weight_hh_l' + str(i))
        if i == 0:
            input_size = rnn_module.input_size
        else:
            input_size = rnn_module.hidden_size
        syops = rnn_syops(syops, rnn_module, w_ih, w_hh, input_size)
        if rnn_module.bias:
            b_ih = rnn_module.__getattr__('bias_ih_l' + str(i))
            b_hh = rnn_module.__getattr__('bias_hh_l' + str(i))
            syops += b_ih.shape[0] + b_hh.shape[0]

    syops *= batch_size
    syops *= seq_length
    if rnn_module.bidirectional:
        syops *= 2
    rnn_module.__syops__[0] += int(syops)

def rnn_cell_syops_counter_hook(rnn_cell_module, input, output):
    syops = 0
    inp = input[0]
    batch_size = inp.shape[0]
    w_ih = rnn_cell_module.__getattr__('weight_ih')
    w_hh = rnn_cell_module.__getattr__('weight_hh')
    input_size = inp.shape[1]
    syops = rnn_syops(syops, rnn_cell_module, w_ih, w_hh, input_size)
    if rnn_cell_module.bias:
        b_ih = rnn_cell_module.__getattr__('bias_ih')
        b_hh = rnn_cell_module.__getattr__('bias_hh')
        syops += b_ih.shape[0] + b_hh.shape[0]

    syops *= batch_size
    rnn_cell_module.__syops__[0] += int(syops)


def multihead_attention_counter_hook(multihead_attention_module, input, output):
    syops = 0

    q, k, v = input

    batch_first = multihead_attention_module.batch_first \
        if hasattr(multihead_attention_module, 'batch_first') else False
    if batch_first:
        batch_size = q.shape[0]
        len_idx = 1
    else:
        batch_size = q.shape[1]
        len_idx = 0

    dim_idx = 2

    qdim = q.shape[dim_idx]
    kdim = k.shape[dim_idx]
    vdim = v.shape[dim_idx]

    qlen = q.shape[len_idx]
    klen = k.shape[len_idx]
    vlen = v.shape[len_idx]

    num_heads = multihead_attention_module.num_heads
    assert qdim == multihead_attention_module.embed_dim

    if multihead_attention_module.kdim is None:
        assert kdim == qdim
    if multihead_attention_module.vdim is None:
        assert vdim == qdim

    syops = 0

    # Q scaling
    syops += qlen * qdim

    # Initial projections
    syops += (
        (qlen * qdim * qdim)  # QW
        + (klen * kdim * kdim)  # KW
        + (vlen * vdim * vdim)  # VW
    )

    if multihead_attention_module.in_proj_bias is not None:
        syops += (qlen + klen + vlen) * qdim

    # attention heads: scale, matmul, softmax, matmul
    qk_head_dim = qdim // num_heads
    v_head_dim = vdim // num_heads

    head_syops = (
        (qlen * klen * qk_head_dim)  # QK^T
        + (qlen * klen)  # softmax
        + (qlen * klen * v_head_dim)  # AV
    )

    syops += num_heads * head_syops

    # final projection, bias is always enabled
    syops += qlen * vdim * (vdim + 1)

    syops *= batch_size
    multihead_attention_module.__syops__[0] += int(syops)

def ms_conv_syops_counter_hook(conv_module, input, output):
    # Can have multiple inputs, getting the first one
    input = input[0]  # input is tuple, input[0].shape = torch.Size([4, 3, 32, 32,T]) [B, C, H, W,T]
    spike, rate, spkhistc = ms_spike_rate(input)

    B,C,H,W,T =  input.shape
    TS=T
    if C ==3:
        TS = 1
    
    batch_size = input.shape[0]
    output_dims = list(output.shape[2:4])  # output.shape = torch.Size([4, 48, 32, 32])

    kernel_dims = list(conv_module.kernel_size)
    in_channels = conv_module.in_channels
    out_channels = conv_module.out_channels
    groups = conv_module.groups

    filters_per_channel = out_channels // groups
    conv_per_position_syops = int(np.prod(kernel_dims)) * \
        in_channels * filters_per_channel

    active_elements_count = batch_size * int(np.prod(output_dims))

    overall_conv_syops = conv_per_position_syops * active_elements_count * TS

    bias_syops = 0

    if conv_module.bias is not None:

        bias_syops = out_channels * active_elements_count * TS

    overall_syops = overall_conv_syops + bias_syops

    conv_module.__syops__[0] += int(overall_syops)

    if spike:
        conv_module.__syops__[1] += int(overall_syops) * rate
    else:
        conv_module.__syops__[2] += int(overall_syops)

    conv_module.__syops__[3] += rate * 100
    conv_module.__spkhistc__ = spkhistc

def ms_pool_syops_counter_hook(module, input, output):
    input = input[0]  # input is tuple, input[0].shape = torch.Size([4, 192, 32, 32]) [TB, C, H, W]  # output.shape = torch.Size([4, 192, 16, 16])
    spike, rate, spkhistc = ms_spike_rate(input)
    module.__syops__[0] += int(np.prod(input.shape))

    if spike:
        module.__syops__[1] += int(np.prod(input.shape)) * rate
    else:
        module.__syops__[2] += int(np.prod(input.shape))

    module.__syops__[3] += rate * 100
    module.__spkhistc__ = spkhistc

CUSTOM_MODULES_MAPPING = {}

MODULES_MAPPING = {
    # convolutions
    nn.Conv1d: conv_syops_counter_hook,
    nn.Conv2d: conv_syops_counter_hook,
    nn.Conv3d: conv_syops_counter_hook,
    # activations
    nn.ReLU: relu_syops_counter_hook,
    nn.PReLU: relu_syops_counter_hook,
    nn.ELU: relu_syops_counter_hook,
    nn.LeakyReLU: relu_syops_counter_hook,
    nn.ReLU6: relu_syops_counter_hook,
    # poolings
    nn.MaxPool1d: pool_syops_counter_hook,
    nn.AvgPool1d: pool_syops_counter_hook,
    nn.AvgPool2d: pool_syops_counter_hook,
    nn.MaxPool2d: pool_syops_counter_hook,
    nn.MaxPool3d: pool_syops_counter_hook,
    nn.AvgPool3d: pool_syops_counter_hook,
    nn.AdaptiveMaxPool1d: pool_syops_counter_hook,
    nn.AdaptiveAvgPool1d: pool_syops_counter_hook,
    nn.AdaptiveMaxPool2d: pool_syops_counter_hook,
    nn.AdaptiveAvgPool2d: pool_syops_counter_hook,
    nn.AdaptiveMaxPool3d: pool_syops_counter_hook,
    nn.AdaptiveAvgPool3d: pool_syops_counter_hook,
    # BNs
    nn.BatchNorm1d: bn_syops_counter_hook,
    nn.BatchNorm2d: bn_syops_counter_hook,
    nn.BatchNorm3d: bn_syops_counter_hook,

    # Neuron IF
    MultiStepIFNode: IF_syops_counter_hook,
    IFNode: IF_syops_counter_hook,
    # Neuron LIF
    MultiStepLIFNode: LIF_syops_counter_hook,
    LIFNode: LIF_syops_counter_hook,
    # Neuron PLIF
    MultiStepParametricLIFNode: LIF_syops_counter_hook,
    ParametricLIFNode: LIF_syops_counter_hook,

    nn.InstanceNorm1d: bn_syops_counter_hook,
    nn.InstanceNorm2d: bn_syops_counter_hook,
    nn.InstanceNorm3d: bn_syops_counter_hook,
    nn.GroupNorm: bn_syops_counter_hook,
    # FC
    nn.Linear: linear_syops_counter_hook,
    # Upscale
    nn.Upsample: upsample_syops_counter_hook,
    # Deconvolution
    nn.ConvTranspose1d: conv_syops_counter_hook,
    nn.ConvTranspose2d: conv_syops_counter_hook,
    nn.ConvTranspose3d: conv_syops_counter_hook,
    # RNN
    nn.RNN: rnn_syops_counter_hook,
    nn.GRU: rnn_syops_counter_hook,
    nn.LSTM: rnn_syops_counter_hook,
    nn.RNNCell: rnn_cell_syops_counter_hook,
    nn.LSTMCell: rnn_cell_syops_counter_hook,
    nn.GRUCell: rnn_cell_syops_counter_hook,
    nn.MultiheadAttention: multihead_attention_counter_hook,

    # ms layer
    msConv: ms_conv_syops_counter_hook,
    msAvgPool2d:ms_pool_syops_counter_hook
}

if hasattr(nn, 'GELU'):
    MODULES_MAPPING[nn.GELU] = relu_syops_counter_hook

