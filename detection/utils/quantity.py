import torch

def anasys_data(data):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    fig=plt.figure()
    ax=fig.add_subplot(111)

    W_max = torch.max(data)
    W_min = torch.min(data)
    S =int(torch.max(torch.abs(W_max), torch.abs(W_min))+1)

    data = data.cpu().detach().numpy()
    data =data.reshape(-1)
    

    y=np.zeros(int(S*20))
    x=np.arange(-S, S, 0.1)

    for d in data:
        i = max(min(int(d/0.1)+int(S*10),int(S*20)-1),0)
        y[i]+=1

    ax.bar(x, y, width=0.1,  color='#FF0000', edgecolor='k', zorder=2)
    ax.set_yscale("log")
    plt.savefig('ss.png')
    
    return 

def SymmetryQ_Para(weight_value, bias_value=None, thresh=1.0, Qmax = 127, Qmin = -128,**kwargs):
    # anasys_data(weight_value)
    out_channel, in_channel, ks1, ks2 = weight_value.shape
    W_max = torch.max(weight_value)
    W_min = torch.min(weight_value)
    B_max=Qmin
    B_min=Qmax
    if bias_value is not None:
        B_max = torch.max(bias_value)
        B_min = torch.min(bias_value)
    Rmax = torch.max(W_max, B_max)
    Rmin = torch.min(W_min, B_min)
    S = torch.max(torch.abs(Rmax), torch.abs(Rmin)) / Qmax

    Q_weight_para = torch.round(weight_value / S)
    Q_bias_para = torch.round(bias_value / S) if bias_value is not None else None

    # print('max_weight=', Q_weight_para.max(), 'min_weight=', Q_weight_para.min())
    # print('max_bias=', Q_bias_para.max(), 'min_bias=', Q_bias_para.min())

    vth = torch.round(thresh / (torch.round(S * 2 ** 16) / (2 ** 16)))
    op_num = in_channel*ks1*ks2 + 1   # 1 represents add bias operation
    # print('vth=', vth)

    outputs= {'weight':Q_weight_para,
            'bias'  :Q_bias_para,
            'vth'   :vth,
            'S'     :S,
            'op_num':op_num}
    
    v_init = kwargs.get('v_init', None)
    if v_init is not None:
        v_init = torch.round(v_init / (torch.round(S * 2 ** 16) / (2 ** 16)))
        outputs['v_init']=v_init

    return outputs

def quantify_net(model):
    for _ , module in model._modules.items():
        if hasattr(module, 'quantify'):
            module.quantify()
        if hasattr(module,"_modules"):
            quantify_net(module)
    return model