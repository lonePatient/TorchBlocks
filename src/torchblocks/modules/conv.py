import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        """ Conv1D layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2)
            Basically works like a Linear layer but the weights are transposed
        """
        super().__init__()
        self.out_channels = out_channels
        w = torch.empty(in_channels, out_channels)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.out_channels,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x


class MaskedConv1d(nn.Conv1d):

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1,
                 groups=1, bias=True, causal=True):
        if causal:
            padding = (kernel_size - 1) * dilation
        else:
            padding = (kernel_size - 1) * dilation // 2
        super(MaskedConv1d, self).__init__(in_channels, out_channels, kernel_size,
                                           stride=1, padding=padding, dilation=dilation,
                                           groups=groups, bias=bias)

    def forward(self, inputs):
        output = super(MaskedConv1d, self).forward(inputs)
        return output[:, :, :inputs.size(2)]


class GatedConv1d(MaskedConv1d):

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1,
                 groups=1, bias=True, causal=True):
        super(GatedConv1d, self).__init__(in_channels, 2 * out_channels,
                                          kernel_size, dilation, groups, bias, causal)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        output = super(GatedConv1d, self).forward(inputs)
        mask, output = output.chunk(2, 1)
        mask = self.sigmoid(mask)

        return output * mask


class DilateConvLayer(nn.Module):
    def __init__(self, input_size, channels, dilation, dropout=0.1):
        super(DilateConvLayer, self).__init__()
        self.base = nn.Sequential(
            nn.Dropout2d(dropout),
            nn.Conv2d(input_size, channels, kernel_size=1),
            nn.GELU(),
        )

        self.convs = nn.ModuleList(
            [nn.Conv2d(channels, channels, kernel_size=3, groups=channels, dilation=d, padding=d) for d in dilation])

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.base(x)
        outputs = []
        for conv in self.convs:
            x = conv(x)
            x = F.gelu(x)
            outputs.append(x)
        outputs = torch.cat(outputs, dim=1)
        outputs = outputs.permute(0, 2, 3, 1).contiguous()
        return outputs


class ConvolutionCharEncoder(nn.Module):
    r"""
    char级别的卷积编码器.
    """

    def __init__(self, char_emb_size=50, feature_maps=(40, 30, 30), kernels=(1, 3, 5)):
        r"""
        :param int char_emb_size: char级别embedding的维度. Default: 50
            :例: 有26个字符, 每一个的embedding是一个50维的向量, 所以输入的向量维度为50.
        :param tuple feature_maps: 一个由int组成的tuple. tuple的长度是char级别卷积操作的数目, 第`i`个int表示第`i`个卷积操作的filter.
        :param tuple kernels: 一个由int组成的tuple. tuple的长度是char级别卷积操作的数目, 第`i`个int表示第`i`个卷积操作的卷积核.
        :param initial_method: 初始化参数的方式, 默认为`xavier normal`
        """
        super(ConvolutionCharEncoder, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(1, feature_maps[i], kernel_size=(char_emb_size, kernels[i]), bias=True,
                      padding=(0, kernels[i] // 2))
            for i in range(len(kernels))])

    def forward(self, x):
        r"""
        :param torch.Tensor x: ``[batch_size * sent_length, word_length, char_emb_size]`` 输入字符的embedding
        :return: torch.Tensor : 卷积计算的结果, 维度为[batch_size * sent_length, sum(feature_maps), 1]
        """
        x = x.contiguous().view(x.size(0), 1, x.size(1), x.size(2))
        # [batch_size*sent_length, channel, width, height]
        x = x.transpose(2, 3)
        # [batch_size*sent_length, channel, height, width]
        return self._convolute(x).unsqueeze(2)

    def _convolute(self, x):
        feats = []
        for conv in self.convs:
            y = conv(x)
            # [batch_size*sent_length, feature_maps[i], 1, width - kernels[i] + 1]
            y = torch.squeeze(y, 2)
            # [batch_size*sent_length, feature_maps[i], width - kernels[i] + 1]
            y = torch.tanh(y)
            y, __ = torch.max(y, 2)
            # [batch_size*sent_length, feature_maps[i]]
            feats.append(y)
        return torch.cat(feats, 1)  # [batch_size*sent_length, sum(feature_maps)]
