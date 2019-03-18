import torch
import math
import numpy as np
from torch.distributions.normal import Normal


# 计算 mu sigma 和 w
def cal_para(out):

    cgm_factor = 4
    r_u = 1.6
    r_s = 1.1
    r_w = 1 / 1.75

    temperature = 0.01

    out = out.permute(0, 2, 1).contiguous()
    out = out.view(out.shape[0], out.shape[1], -1, cgm_factor)

    a0 = out[:, :, :, 0]
    a1 = out[:, :, :, 1]
    a2 = out[:, :, :, 2]
    a3 = out[:, :, :, 3]

    xi = 2 * torch.sigmoid(a0) - 1
    omega = torch.exp(4 * torch.sigmoid(a1)) * 2 / 255
    alpha = 2 * torch.sigmoid(a2) - 1
    beta = 2 * torch.sigmoid(a3)

    sigmas = []
    for k in range(cgm_factor):
        sigma = omega * torch.exp(k * (torch.abs(alpha) * r_s - 1))
        sigma *= math.sqrt(temperature)
        sigmas.append(sigma)

    mus = []
    for k in range(cgm_factor):
        temp_sum = 0
        for i in range(k):
            temp_sum += sigmas[i] * r_u * alpha
        mu = xi + temp_sum
        mus.append(mu)

    ws = []
    for k in range(cgm_factor):
        temp_sum = 0
        for i in range(4):
            temp_sum += alpha.pow(2 * i) * beta.pow(i) * (r_w ** i)
        w = (alpha.pow(2 * k) * beta.pow(k) * (r_w ** k)) / temp_sum
        ws.append(w)

    _mus = 0
    for k in range(cgm_factor):
        _mus += ws[k]*mus[k]

    for k in range(cgm_factor):
        mus[k] = mus[k] + (_mus - mus[k])*(1 - temperature)


    return sigmas, mus, ws

#  x dim = (batch, output_channel, length)
#  l dim = (batch, output_channel * cgm_factor, length)


def CGM_loss(out, y):
    y = y.permute(0, 2, 1)

    sigmas, mus, ws = cal_para(out)

    #print(torch.mean(sigmas[0]))
    #  验证w之和是1
    sum = 0
    for k in range(4):
        tw = ws[k].view(-1)
        sum += tw

    # todo 增加tau 拉近分布之间距离

    #  alternative： torch.distributions.normal.Normal
    probs = 0
    for k in range(4):
        dist = Normal(mus[k], sigmas[k])
        log_prob = dist.log_prob(y)

        x = dist.sample()
        # prob = log_prob * log_prob
        probs += ws[k] * log_prob

    return -torch.mean(probs)


def sample_from_CGM(out):
    out = out.unsqueeze(1)
    out = out.unsqueeze(0)
    sigmas, mus, ws = cal_para(out)

    value = 0
    rand = torch.rand(ws[0].shape).cuda()

    for k in range(4):
        mask_btm = torch.zeros(ws[k].shape).cuda()
        for i in range(k):
            mask_btm += ws[i]
        mask = (rand < (ws[k] + mask_btm)) * (rand >= mask_btm)
        mask = mask.float()
        gaussian_dist = Normal(loc=mus[k], scale=sigmas[k])
        x = gaussian_dist.sample()
        value += mask * x

    #  value shape (batch, length, channel'60')
    return value

