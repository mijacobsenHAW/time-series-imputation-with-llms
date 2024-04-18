"""
    Source: https://github.com/thuml/Time-Series-Library/tree/main
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.conv_blocks_layers import InceptionBlockV1


def fft_for_period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self,
                 sequence_len: int,
                 prediction_len: int,
                 top_k: int,
                 model_dimension: int,
                 d_ff: int,
                 num_kernels: int,
                 device: str):
        super(TimesBlock, self).__init__()
        self.sequence_len = sequence_len
        self.prediction_len = prediction_len
        self.k = top_k
        self.device = device
        # parameter-efficient design
        self.conv = nn.Sequential(
            InceptionBlockV1(in_channels=model_dimension,
                             out_channels=d_ff,
                             num_kernels=num_kernels),
            nn.GELU(),
            InceptionBlockV1(in_channels=d_ff,
                             out_channels=model_dimension,
                             num_kernels=num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size()
        if self.device:
            # x = x.to('cpu')
            period_list, period_weight = fft_for_period(x, self.k)
            # x = x.to('mps')
        else:
            period_list, period_weight = fft_for_period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.sequence_len + self.prediction_len) % period != 0:
                length = (((self.sequence_len + self.prediction_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.sequence_len + self.prediction_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.sequence_len + self.prediction_len)
                out = x
            # reshape
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.sequence_len + self.prediction_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        #if self.device:
        #    period_weight = period_weight.to('mps')
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res
