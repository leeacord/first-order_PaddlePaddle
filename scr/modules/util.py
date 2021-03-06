import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


def kp2gaussian(kp, spatial_size, kp_variance: np.ndarray) -> np.ndarray:
    """
    Transform a keypoint into gaussian like representation
    """
    mean = kp['value']
    coordinate_grid = make_coordinate_grid(spatial_size, mean.dtype)
    number_of_leading_dimensions = len(mean.shape) - 1
    shape = (1,) * number_of_leading_dimensions + tuple(coordinate_grid.shape)
    coordinate_grid = coordinate_grid.reshape(shape)
    repeats = tuple(mean.shape)[:number_of_leading_dimensions] + (1, 1, 1)
    coordinate_grid = coordinate_grid.tile(repeats)

    # Preprocess kp shape
    shape = tuple(mean.shape)[:number_of_leading_dimensions] + (1, 1, 2)
    mean = kp['value'].reshape(shape)
    mean_sub = (coordinate_grid - mean)
    kp_variance = paddle.to_tensor(np.array([kp_variance]).astype(np.float32))
    out = paddle.exp(-0.5 * (mean_sub ** 2).sum(-1) / kp_variance)
    return out


def make_coordinate_grid(spatial_size, type='float32'):
    """
    Create a meshgrid [-1,1] x [-1,1] of given spatial_size.
    """
    h, w = spatial_size
    x = paddle.arange(w, dtype='float32')
    y = paddle.arange(h, dtype='float32')

    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)

    yy = y.reshape((-1, 1)).tile((1, w))
    xx = x.reshape((1, -1)).tile((h, 1))

    meshed = paddle.concat([xx.unsqueeze(2), yy.unsqueeze(2)], 2)
    return meshed


class ResBlock2d(paddle.nn.Layer):
    """
    Res block, preserve spatial resolution.
    """
    def __init__(self, in_features, kernel_size, padding, **kwargs):
        super(ResBlock2d, self).__init__(**kwargs)
        self.conv1 = nn.Conv2D(in_features, in_features, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv2D(in_features, in_features, kernel_size=kernel_size, padding=padding)
        self.norm1 = nn.BatchNorm(num_channels=in_features)
        self.norm2 = nn.BatchNorm(num_channels=in_features)

    def forward(self, x):
        out = self.norm1(x)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = F.relu(out)
        out = self.conv2(out)
        out += x
        return out


class UpBlock2d(paddle.nn.Layer):
    """
    Upsampling block for use in decoder.
    """
    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(UpBlock2d, self).__init__()
        self.conv = nn.Conv2D(
            in_features,
            out_features,
            kernel_size=kernel_size,
            padding=padding,
            groups=groups)
        self.norm = nn.BatchNorm(num_channels=out_features)

    def forward(self, x):
        out = F.interpolate(x, scale_factor=2, mode='NEAREST', align_corners=False)
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out)
        return out


class DownBlock2d(paddle.nn.Layer):
    """
    Downsampling block for use in encoder.
    """
    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(DownBlock2d, self).__init__()
        self.conv = nn.Conv2D(in_features, out_features, kernel_size=kernel_size, padding=padding, groups=groups)
        self.norm = nn.BatchNorm(num_channels=out_features)
        self.pool = nn.AvgPool2D(kernel_size=(2, 2), stride=2)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        out = self.pool(out)
        return out


class SameBlock2d(paddle.nn.Layer):
    """
    Simple block, preserve spatial resolution.
    """
    def __init__(self, in_features, out_features, groups=1, kernel_size=3, padding=1):
        super(SameBlock2d, self).__init__()
        self.conv = nn.Conv2D(in_features, out_features, kernel_size=kernel_size, padding=padding, groups=groups)
        self.norm = nn.BatchNorm(out_features)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        return out


class Encoder(paddle.nn.Layer):
    """
    Hourglass Encoder
    """
    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Encoder, self).__init__()

        down_blocks = []
        for i in range(num_blocks):
            down_blocks.append(
                DownBlock2d(
                    in_features if i == 0 else min(max_features, block_expansion * (2 ** i)),
                    min(max_features, block_expansion * (2 ** (i + 1))),
                    kernel_size=3,
                    padding=1
                )
            )
        self.down_blocks = paddle.nn.LayerList(down_blocks)

    def forward(self, x):
        outs = [x]
        for down_block in self.down_blocks:
            outs.append(down_block(outs[-1]))
        return outs


class Decoder(paddle.nn.Layer):
    """
    Hourglass Decoder
    """
    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Decoder, self).__init__()
        up_blocks = []
        for i in range(num_blocks)[::-1]:
            in_filters = (1 if i == num_blocks - 1 else 2) * min(max_features, block_expansion * (2 ** (i + 1)))
            out_filters = min(max_features, block_expansion * (2 ** i))
            up_blocks.append(UpBlock2d(in_filters, out_filters, kernel_size=3, padding=1))

        self.up_blocks = paddle.nn.LayerList(up_blocks)
        self.out_filters = block_expansion + in_features

    def forward(self, x):
        out = x.pop()
        for up_block in self.up_blocks:
            out = up_block(out)
            skip = x.pop()
            # TODO: If the size of width or length is odd, out and skip cannot concat
            out = paddle.concat([out, skip], axis=1)
        return out


class Hourglass(paddle.nn.Layer):
    """
    Hourglass architecture.
    """
    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Hourglass, self).__init__()
        self.encoder = Encoder(block_expansion, in_features, num_blocks, max_features)
        self.decoder = Decoder(block_expansion, in_features, num_blocks, max_features)
        self.out_filters = self.decoder.out_filters

    def forward(self, x) -> paddle.fluid.core_avx.VarBase:
        return self.decoder(self.encoder(x))


class AntiAliasInterpolation2d(paddle.nn.Layer):
    """
    Band-limited downsampling, for better preservation of the input signal.
    """
    def __init__(self, channels, scale):
        super(AntiAliasInterpolation2d, self).__init__()
        sigma = (1 / scale - 1) / 2
        kernel_size = 2 * round(sigma * 4) + 1
        self.ka = kernel_size // 2
        self.kb = self.ka - 1 if kernel_size % 2 == 0 else self.ka

        kernel_size = [kernel_size, kernel_size]
        sigma = [sigma, sigma]
        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = paddle.meshgrid([paddle.arange(size, dtype='float32') for size in kernel_size])
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= paddle.exp(-(mgrid - mean) ** 2 / (2 * std ** 2 + 1e-9))

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / paddle.sum(kernel)
        # Reshape to depthwise convolutional weight
        kernel = kernel.reshape((1, 1, *kernel.shape))
        kernel = kernel.tile((channels, *((1,)*(len(kernel.shape)-1))))  # [1, 1, *kernel.shape] -> [channels, 1, *kernel.shape]
        self.kernel_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Assign(kernel), trainable=False)
        self.groups = channels
        self.scale = scale
        self.conv = nn.Conv2D(channels, channels, kernel_size=kernel.shape[-1], groups=self.groups,
                              weight_attr=self.kernel_attr,
                              bias_attr=False
                              )
        self.conv.weight.set_value(kernel)

    def forward(self, input):
        if self.scale == 1.0:
            return input
        out = F.pad(input, [self.ka, self.kb, self.ka, self.kb], mode='constant')
        out = self.conv(out)
        out = F.interpolate(out, scale_factor=self.scale, mode='NEAREST', align_corners=False)
        return out
