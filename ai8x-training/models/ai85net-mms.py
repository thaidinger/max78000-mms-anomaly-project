###################################################################################################
#
# MMS anomaly-grid network for MAX78000.
#
###################################################################################################
"""
Small MMS anomaly-grid network.
"""
from torch import nn
import torch

import ai8x


class AI85MMSAnomalyNet(nn.Module):
    """
    TinyGLASS-style anomaly-grid model.

    Input:  B x 3 x 128 x 128
    Output: B x 2 x 32 x 32
    """

    def __init__(
            self,
            num_classes=2,
            num_channels=3,
            dimensions=(128, 128),  # pylint: disable=unused-argument
            bias=True,
            auxiliary_head=False,
            **kwargs
    ):
        super().__init__()
        self.auxiliary_head = auxiliary_head

        # Downsample immediately so the first hidden tensor fits without streaming/FIFO.
        self.conv1 = ai8x.FusedMaxPoolConv2dBNReLU(num_channels, 8, 3, pool_size=2,
                                                   pool_stride=2, stride=1, padding=1,
                                                   bias=bias, batchnorm='NoAffine',
                                                   **kwargs)
        self.conv2 = ai8x.FusedMaxPoolConv2dBNReLU(8, 16, 3, pool_size=2, pool_stride=2,
                                                   stride=1, padding=1, bias=bias,
                                                   batchnorm='NoAffine', **kwargs)
        self.conv3 = ai8x.FusedConv2dBNReLU(16, 32, 3, stride=1, padding=1, bias=bias,
                                            batchnorm='NoAffine', **kwargs)
        self.conv4 = ai8x.FusedConv2dBNReLU(32, 32, 3, stride=1, padding=1, bias=bias,
                                            batchnorm='NoAffine', **kwargs)
        self.conv5 = ai8x.FusedConv2dBNReLU(32, 16, 3, stride=1, padding=1, bias=bias,
                                            batchnorm='NoAffine', **kwargs)
        self.head = ai8x.FusedConv2dBN(16, num_classes, 1, stride=1, padding=0, bias=bias,
                                       batchnorm='NoAffine', **kwargs)
        if self.auxiliary_head:
            self.aux_fc = nn.Linear(32, num_classes, bias=True)

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        x = self.conv1(x)  # 8 x 64 x 64
        x = self.conv2(x)  # 16 x 32 x 32
        x = self.conv3(x)  # 32 x 32 x 32
        x = self.conv4(x)  # 32 x 32 x 32
        x = self.conv5(x)  # 16 x 32 x 32
        map_logits = self.head(x)  # num_classes x 32 x 32
        if not self.auxiliary_head:
            return map_logits

        avg_features = x.mean(dim=(2, 3))
        max_features = x.amax(dim=(2, 3))
        image_logits = self.aux_fc(torch.cat((avg_features, max_features), dim=1))
        return map_logits, image_logits


def ai85mmsanomalynet(pretrained=False, **kwargs):
    """
    Constructs an MMS anomaly-grid model.
    """
    assert not pretrained
    return AI85MMSAnomalyNet(**kwargs)


def ai85mmsanomalynetaux(pretrained=False, **kwargs):
    """
    Constructs an MMS anomaly-grid model with a training-only image classification head.
    """
    assert not pretrained
    return AI85MMSAnomalyNet(auxiliary_head=True, **kwargs)


class AI85MMSAnomalyNetTiny(nn.Module):
    """
    Very small MMS anomaly-grid model.

    Input:  B x 3 x 128 x 128
    Output: B x 2 x 32 x 32
    """

    def __init__(
            self,
            channels,
            num_classes=2,
            num_channels=3,
            dimensions=(128, 128),  # pylint: disable=unused-argument
            bias=True,
            auxiliary_head=False,
            **kwargs
    ):
        super().__init__()
        self.auxiliary_head = auxiliary_head
        self.feature_names = []

        in_channels = num_channels
        for index, out_channels in enumerate(channels, start=1):
            name = f'conv{index}'
            if len(channels) == 1:
                layer = ai8x.FusedMaxPoolConv2dBNReLU(
                    in_channels, out_channels, 3, pool_size=4, pool_stride=4,
                    stride=1, padding=1, bias=bias, batchnorm='NoAffine', **kwargs
                )
            else:
                layer = _mms_conv_layer(in_channels, out_channels, bias,
                                        pooled=index in (1, 2), **kwargs)
            setattr(self, name, layer)
            self.feature_names.append(name)
            in_channels = out_channels

        self.head = ai8x.FusedConv2dBN(in_channels, num_classes, 1, stride=1, padding=0,
                                       bias=bias, batchnorm='NoAffine', **kwargs)
        if self.auxiliary_head:
            self.aux_fc = nn.Linear(2 * in_channels, num_classes, bias=True)

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        for name in self.feature_names:
            x = getattr(self, name)(x)

        map_logits = self.head(x)  # num_classes x 32 x 32
        if not self.auxiliary_head:
            return map_logits

        avg_features = x.mean(dim=(2, 3))
        max_features = x.amax(dim=(2, 3))
        image_logits = self.aux_fc(torch.cat((avg_features, max_features), dim=1))
        return map_logits, image_logits


def _mms_conv_layer(in_channels, out_channels, bias, pooled=False, **kwargs):
    if pooled:
        return ai8x.FusedMaxPoolConv2dBNReLU(in_channels, out_channels, 3, pool_size=2,
                                             pool_stride=2, stride=1, padding=1, bias=bias,
                                             batchnorm='NoAffine', **kwargs)
    return ai8x.FusedConv2dBNReLU(in_channels, out_channels, 3, stride=1, padding=1,
                                  bias=bias, batchnorm='NoAffine', **kwargs)


class AI85MMSAnomalyNetDeep(nn.Module):
    """
    Deeper MMS anomaly-grid model.

    Input:  B x 3 x 128 x 128
    Output: B x 2 x 32 x 32
    """

    def __init__(
            self,
            channels,
            num_classes=2,
            num_channels=3,
            dimensions=(128, 128),  # pylint: disable=unused-argument
            bias=True,
            auxiliary_head=False,
            residual=False,
            **kwargs
    ):
        super().__init__()
        self.auxiliary_head = auxiliary_head
        self.feature_names = []
        self.residual_names = set()

        in_channels = num_channels
        same_shape_run = 0
        for index, out_channels in enumerate(channels, start=1):
            name = f'conv{index}'
            pooled = index in (1, 2)
            # MMS NOTE: keep the same 128->64->32 downsampling as the deployable small model.
            setattr(self, name, _mms_conv_layer(in_channels, out_channels, bias,
                                                pooled=pooled, **kwargs))
            self.feature_names.append(name)

            if residual and not pooled and in_channels == out_channels:
                same_shape_run += 1
                if same_shape_run % 2 == 1:
                    resid_name = f'resid{index}'
                    setattr(self, resid_name, ai8x.Add())
                    self.residual_names.add(name)
            else:
                same_shape_run = 0
            in_channels = out_channels

        self.head = ai8x.FusedConv2dBN(in_channels, num_classes, 1, stride=1, padding=0,
                                       bias=bias, batchnorm='NoAffine', **kwargs)
        if self.auxiliary_head:
            self.aux_fc = nn.Linear(2 * in_channels, num_classes, bias=True)

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        for name in self.feature_names:
            if name in self.residual_names:
                x_res = x
                x = getattr(self, name)(x)
                x = getattr(self, f"resid{name[4:]}")(x, x_res)
            else:
                x = getattr(self, name)(x)

        map_logits = self.head(x)  # num_classes x 32 x 32
        if not self.auxiliary_head:
            return map_logits

        avg_features = x.mean(dim=(2, 3))
        max_features = x.amax(dim=(2, 3))
        image_logits = self.aux_fc(torch.cat((avg_features, max_features), dim=1))
        return map_logits, image_logits


MMS_15_CHANNELS = (8, 16, 24, 24, 32, 32, 32, 32, 32, 32, 32, 32, 24, 24, 16)
MMS_25_CHANNELS = (8, 16, 24, 24, 32, 32, 32, 32, 32, 32, 32, 32, 32,
                   32, 32, 32, 32, 32, 32, 32, 24, 24, 24, 24, 16)
MMS_1_CHANNELS = (8,)
MMS_3_CHANNELS = (8, 16, 16)
MMS_10_CHANNELS = (8, 16, 24, 24, 32, 32, 32, 24, 24, 16)


def ai85mmsanomalynet1(pretrained=False, **kwargs):
    """
    Constructs a 1-feature-layer MMS anomaly-grid model.
    """
    assert not pretrained
    return AI85MMSAnomalyNetTiny(MMS_1_CHANNELS, **kwargs)


def ai85mmsanomalynet1aux(pretrained=False, **kwargs):
    """
    Constructs a 1-feature-layer MMS model with a training-only image head.
    """
    assert not pretrained
    return AI85MMSAnomalyNetTiny(MMS_1_CHANNELS, auxiliary_head=True, **kwargs)


def ai85mmsanomalynet3(pretrained=False, **kwargs):
    """
    Constructs a 3-feature-layer MMS anomaly-grid model.
    """
    assert not pretrained
    return AI85MMSAnomalyNetTiny(MMS_3_CHANNELS, **kwargs)


def ai85mmsanomalynet3aux(pretrained=False, **kwargs):
    """
    Constructs a 3-feature-layer MMS model with a training-only image head.
    """
    assert not pretrained
    return AI85MMSAnomalyNetTiny(MMS_3_CHANNELS, auxiliary_head=True, **kwargs)


def ai85mmsanomalynet10(pretrained=False, **kwargs):
    """
    Constructs a 10-feature-layer MMS anomaly-grid model.
    """
    assert not pretrained
    return AI85MMSAnomalyNetDeep(MMS_10_CHANNELS, **kwargs)


def ai85mmsanomalynet10aux(pretrained=False, **kwargs):
    """
    Constructs a 10-feature-layer MMS model with a training-only image head.
    """
    assert not pretrained
    return AI85MMSAnomalyNetDeep(MMS_10_CHANNELS, auxiliary_head=True, **kwargs)


def ai85mmsanomalynet15(pretrained=False, **kwargs):
    """
    Constructs a 15-feature-layer MMS anomaly-grid model.
    """
    assert not pretrained
    return AI85MMSAnomalyNetDeep(MMS_15_CHANNELS, **kwargs)


def ai85mmsanomalynet15aux(pretrained=False, **kwargs):
    """
    Constructs a 15-feature-layer MMS model with a training-only image head.
    """
    assert not pretrained
    return AI85MMSAnomalyNetDeep(MMS_15_CHANNELS, auxiliary_head=True, **kwargs)


def ai85mmsanomalynet15res(pretrained=False, **kwargs):
    """
    Constructs a 15-feature-layer residual MMS anomaly-grid model.
    """
    assert not pretrained
    return AI85MMSAnomalyNetDeep(MMS_15_CHANNELS, residual=True, **kwargs)


def ai85mmsanomalynet15resaux(pretrained=False, **kwargs):
    """
    Constructs a 15-feature-layer residual MMS model with a training-only image head.
    """
    assert not pretrained
    return AI85MMSAnomalyNetDeep(MMS_15_CHANNELS, auxiliary_head=True, residual=True, **kwargs)


def ai85mmsanomalynet25(pretrained=False, **kwargs):
    """
    Constructs a 25-feature-layer MMS anomaly-grid model.
    """
    assert not pretrained
    return AI85MMSAnomalyNetDeep(MMS_25_CHANNELS, **kwargs)


def ai85mmsanomalynet25aux(pretrained=False, **kwargs):
    """
    Constructs a 25-feature-layer MMS model with a training-only image head.
    """
    assert not pretrained
    return AI85MMSAnomalyNetDeep(MMS_25_CHANNELS, auxiliary_head=True, **kwargs)


def ai85mmsanomalynet25res(pretrained=False, **kwargs):
    """
    Constructs a 25-feature-layer residual MMS anomaly-grid model.
    """
    assert not pretrained
    return AI85MMSAnomalyNetDeep(MMS_25_CHANNELS, residual=True, **kwargs)


def ai85mmsanomalynet25resaux(pretrained=False, **kwargs):
    """
    Constructs a 25-feature-layer residual MMS model with a training-only image head.
    """
    assert not pretrained
    return AI85MMSAnomalyNetDeep(MMS_25_CHANNELS, auxiliary_head=True, residual=True, **kwargs)


models = [
    {
        'name': 'ai85mmsanomalynet',
        'min_input': 1,
        'dim': 2,
    },
    {
        'name': 'ai85mmsanomalynetaux',
        'min_input': 1,
        'dim': 2,
    },
    {
        'name': 'ai85mmsanomalynet1',
        'min_input': 1,
        'dim': 2,
    },
    {
        'name': 'ai85mmsanomalynet1aux',
        'min_input': 1,
        'dim': 2,
    },
    {
        'name': 'ai85mmsanomalynet3',
        'min_input': 1,
        'dim': 2,
    },
    {
        'name': 'ai85mmsanomalynet3aux',
        'min_input': 1,
        'dim': 2,
    },
    {
        'name': 'ai85mmsanomalynet10',
        'min_input': 1,
        'dim': 2,
    },
    {
        'name': 'ai85mmsanomalynet10aux',
        'min_input': 1,
        'dim': 2,
    },
    {
        'name': 'ai85mmsanomalynet15',
        'min_input': 1,
        'dim': 2,
    },
    {
        'name': 'ai85mmsanomalynet15aux',
        'min_input': 1,
        'dim': 2,
    },
    {
        'name': 'ai85mmsanomalynet15res',
        'min_input': 1,
        'dim': 2,
    },
    {
        'name': 'ai85mmsanomalynet15resaux',
        'min_input': 1,
        'dim': 2,
    },
    {
        'name': 'ai85mmsanomalynet25',
        'min_input': 1,
        'dim': 2,
    },
    {
        'name': 'ai85mmsanomalynet25aux',
        'min_input': 1,
        'dim': 2,
    },
    {
        'name': 'ai85mmsanomalynet25res',
        'min_input': 1,
        'dim': 2,
    },
    {
        'name': 'ai85mmsanomalynet25resaux',
        'min_input': 1,
        'dim': 2,
    },
]
