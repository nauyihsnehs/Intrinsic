import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseModel(torch.nn.Module):
    def load(self, path):
        parameters = torch.load(path, map_location=torch.device('cpu'))
        if "optimizer" in parameters:
            parameters = parameters["model"]
        self.load_state_dict(parameters)


def _calc_same_pad(i, k, s, d):
    return max((-(i // -s) - 1) * s + (k - 1) * d + 1 - i, 0)


def conv2d_same(x, weight, bias=None, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1):
    ih, iw = x.size()[-2:]
    kh, kw = weight.size()[-2:]
    pad_h = _calc_same_pad(ih, kh, stride[0], dilation[0])
    pad_w = _calc_same_pad(iw, kw, stride[1], dilation[1])
    x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
    return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)


class Conv2dSame(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)

    def forward(self, x):
        return conv2d_same(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


def _make_resnet_backbone(resnet):
    pretrained = nn.Module()
    pretrained.layer1 = nn.Sequential(
        resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1
    )
    pretrained.layer2 = resnet.layer2
    pretrained.layer3 = resnet.layer3
    pretrained.layer4 = resnet.layer4
    return pretrained


def _make_pretrained_resnext101_wsl(use_pretrained, in_chan=3, group_width=8):
    resnet = torch.hub.load(
        "facebookresearch/WSL-Images",
        f"resnext101_32x{group_width}d_wsl",
        # pretrained=use_pretrained,
    )
    if in_chan != 3:
        resnet.conv1 = torch.nn.Conv2d(in_chan, 64, 7, 2, 3, bias=False)
    return _make_resnet_backbone(resnet)


def _make_efficientnet_backbone(effnet):
    pretrained = nn.Module()
    pretrained.layer1 = nn.Sequential(
        effnet.conv_stem, effnet.bn1, effnet.act1, *effnet.blocks[0:2]
    )
    pretrained.layer2 = nn.Sequential(*effnet.blocks[2:3])
    pretrained.layer3 = nn.Sequential(*effnet.blocks[3:5])
    pretrained.layer4 = nn.Sequential(*effnet.blocks[5:9])
    return pretrained


def _make_pretrained_efficientnet_lite3(use_pretrained, exportable=False, in_chan=3):
    efficientnet = torch.hub.load(
        "rwightman/gen-efficientnet-pytorch",
        "tf_efficientnet_lite3",
        pretrained=use_pretrained,
        exportable=exportable,
    )
    if in_chan != 3:
        efficientnet.conv_stem = Conv2dSame(in_chan, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
    return _make_efficientnet_backbone(efficientnet)


def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()
    out_shape1 = out_shape2 = out_shape3 = out_shape
    out_shape4 = out_shape

    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        out_shape4 = out_shape * 8

    scratch.layer1_rn = nn.Conv2d(in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer2_rn = nn.Conv2d(in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer3_rn = nn.Conv2d(in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer4_rn = nn.Conv2d(in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    return scratch


def _make_encoder(backbone, features, use_pretrained, groups=1, expand=False, exportable=True, in_chan=3, group_width=8):
    if backbone == "resnext101_wsl":
        pretrained = _make_pretrained_resnext101_wsl(use_pretrained, in_chan=in_chan, group_width=group_width)
        scratch = _make_scratch([256, 512, 1024, 2048], features, groups=groups, expand=expand)
    elif backbone == "efficientnet_lite3":
        pretrained = _make_pretrained_efficientnet_lite3(use_pretrained, exportable=exportable, in_chan=in_chan)
        scratch = _make_scratch([32, 48, 136, 384], features, groups=groups, expand=expand)
    else:
        raise ValueError(f"Backbone '{backbone}' not implemented")
    return pretrained, scratch


class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode, align_corners=False):
        super().__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)


class ResidualConvUnit(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        return out + x


class FeatureFusionBlock(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)

    def forward(self, *xs):
        output = xs[0]
        if len(xs) == 2:
            output += self.resConfUnit1(xs[1])
        output = self.resConfUnit2(output)
        output = nn.functional.interpolate(output, scale_factor=2, mode="bilinear", align_corners=True)
        return output


class ResidualConvUnit_custom(nn.Module):
    def __init__(self, features, activation, bn):
        super().__init__()
        self.bn = bn
        self.groups = 1
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)
        if self.bn:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)
        self.activation = activation
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        out = self.activation(x)
        out = self.conv1(out)
        if self.bn:
            out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        if self.bn:
            out = self.bn2(out)
        return self.skip_add.add(out, x)


class FeatureFusionBlock_custom(nn.Module):
    def __init__(self, features, activation, deconv=False, bn=False, expand=False, align_corners=True, size=None):
        super().__init__()
        self.deconv = deconv
        self.align_corners = align_corners
        self.expand = expand
        out_features = features // 2 if self.expand else features
        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)
        self.resConfUnit1 = ResidualConvUnit_custom(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit_custom(features, activation, bn)
        self.skip_add = nn.quantized.FloatFunctional()
        self.size = size

    def forward(self, *xs, size=None):
        output = xs[0]
        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)
        output = self.resConfUnit2(output)
        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}
        output = nn.functional.interpolate(output, **modifier, mode="bilinear", align_corners=self.align_corners)
        output = self.out_conv(output)
        return output


class MidasNet(BaseModel):
    def __init__(self, activation='sigmoid', pretrained=False, features=256, input_channels=3, output_channels=1, group_width=8, last_residual=False):
        super().__init__()
        self.out_chan = output_channels
        self.last_res = last_residual
        self.pretrained, self.scratch = _make_encoder(
            backbone="resnext101_wsl",
            features=features,
            use_pretrained=pretrained,
            in_chan=input_channels,
            group_width=group_width,
        )
        self.scratch.refinenet4 = FeatureFusionBlock(features)
        self.scratch.refinenet3 = FeatureFusionBlock(features)
        self.scratch.refinenet2 = FeatureFusionBlock(features)
        self.scratch.refinenet1 = FeatureFusionBlock(features)

        if activation == 'sigmoid':
            out_act = nn.Sigmoid()
        elif activation == 'relu':
            out_act = nn.ReLU()
        else:
            out_act = nn.Identity()

        res_dim = 128 + (input_channels if last_residual else 0)
        self.scratch.output_conv = nn.ModuleList([
            nn.Conv2d(features, 128, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(res_dim, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, output_channels, kernel_size=1, stride=1, padding=0),
            out_act,
        ])

    def forward(self, x):
        layer_1 = self.pretrained.layer1(x)
        layer_2 = self.pretrained.layer2(layer_1)
        layer_3 = self.pretrained.layer3(layer_2)
        layer_4 = self.pretrained.layer4(layer_3)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv[0](path_1)
        out = self.scratch.output_conv[1](out)
        if self.last_res:
            out = torch.cat((out, x), dim=1)
        out = self.scratch.output_conv[2](out)
        out = self.scratch.output_conv[3](out)
        out = self.scratch.output_conv[4](out)
        out = self.scratch.output_conv[5](out)
        return out


class MidasNet_small(BaseModel):
    def __init__(
        self,
        activation='sigmoid',
        pretrained=False,
        features=64,
        backbone="efficientnet_lite3",
        exportable=True,
        channels_last=False,
        align_corners=True,
        blocks={'expand': True},
        input_channels=3,
        output_channels=1,
        out_bias=0,
        last_residual=False,
    ):
        super().__init__()
        self.out_chan = output_channels
        self.channels_last = channels_last
        self.blocks = blocks
        self.backbone = backbone
        self.last_res = last_residual
        self.groups = 1

        features1 = features2 = features3 = features4 = features
        self.expand = False
        if self.blocks.get('expand'):
            self.expand = True
            features2 = features * 2
            features3 = features * 4
            features4 = features * 8

        self.pretrained, self.scratch = _make_encoder(self.backbone, features, pretrained, in_chan=input_channels, groups=self.groups, expand=self.expand, exportable=exportable)
        self.scratch.activation = nn.ReLU(False)

        self.scratch.refinenet4 = FeatureFusionBlock_custom(features4, self.scratch.activation, deconv=False, bn=False, expand=self.expand, align_corners=align_corners)
        self.scratch.refinenet3 = FeatureFusionBlock_custom(features3, self.scratch.activation, deconv=False, bn=False, expand=self.expand, align_corners=align_corners)
        self.scratch.refinenet2 = FeatureFusionBlock_custom(features2, self.scratch.activation, deconv=False, bn=False, expand=self.expand, align_corners=align_corners)
        self.scratch.refinenet1 = FeatureFusionBlock_custom(features1, self.scratch.activation, deconv=False, bn=False, align_corners=align_corners)

        if activation == 'sigmoid':
            output_act = nn.Sigmoid()
        elif activation == 'tanh':
            output_act = nn.Tanh()
        elif activation == 'relu':
            output_act = nn.ReLU()
        else:
            output_act = nn.Identity()

        res_dim = (features // 2) + (input_channels if last_residual else 0)
        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1, groups=self.groups),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(res_dim, 32, kernel_size=3, stride=1, padding=1),
            self.scratch.activation,
            nn.Conv2d(32, output_channels, kernel_size=1, stride=1, padding=0),
            output_act,
        )
        self.scratch.output_conv[-2].bias = torch.nn.Parameter(torch.ones(output_channels) * out_bias)

    def forward(self, x):
        if self.channels_last:
            x.contiguous(memory_format=torch.channels_last)

        layer_1 = self.pretrained.layer1(x)
        layer_2 = self.pretrained.layer2(layer_1)
        layer_3 = self.pretrained.layer3(layer_2)
        layer_4 = self.pretrained.layer4(layer_3)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv[0](path_1)
        out = self.scratch.output_conv[1](out)
        if self.last_res:
            out = torch.cat((out, x), dim=1)
        out = self.scratch.output_conv[2](out)
        out = self.scratch.output_conv[3](out)
        out = self.scratch.output_conv[4](out)
        out = self.scratch.output_conv[5](out)
        return out
