# copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle.fluid as fluid
import paddle.fluid.dygraph as dygraph

__all__ = ["VGGNet", "VGG11", "VGG13", "VGG16", "VGG19"]


class conv_block(dygraph.Layer):
    def __init__(self, input_channels, num_filter, groups, name=None, use_bias=False):
        super(conv_block, self).__init__()
        self._layers = []
        i = 0
        self.conv_in = dygraph.Conv2D(
            num_channels=input_channels,
            num_filters=num_filter,
            filter_size=3,
            stride=1,
            padding=1,
            act='relu',
            param_attr=fluid.param_attr.ParamAttr(
                name=name + str(i + 1) + "_weights"),
            bias_attr=False if not use_bias else fluid.param_attr.ParamAttr(
                name=name + str(i + 1) + "_bias"))
        if groups == 1:
            return
        for i in range(1, groups):
            _a = dygraph.Conv2D(
                num_channels=num_filter,
                num_filters=num_filter,
                filter_size=3,
                stride=1,
                padding=1,
                act='relu',
                param_attr=fluid.param_attr.ParamAttr(
                    name=name + str(i + 1) + "_weights"),
                bias_attr=False if not use_bias else fluid.param_attr.ParamAttr(
                    name=name + str(i + 1) + "_bias"))
            self._layers.append(_a)
        self.conv = dygraph.Sequential(self._layers)

    def forward(self, x):
        feat = self.conv_in(x)
        out = fluid.layers.pool2d(input=self.conv(feat), pool_size=2, pool_type='max', pool_stride=2)
        return out, feat


class VGGNet(dygraph.Layer):
    """
    Vgg19 network for perceptual loss. See Sec 3.3.
    """

    def __init__(self, layers=19, torch_version=False, requires_grad=False):
        """
        Args:
            layers: 
            torch_version: 是否使用pytorch的版本
            requires_grad: 
        """
        self.layers = layers
        vgg_spec = {
            11: ([1, 1, 2, 2, 2]),
            13: ([2, 2, 2, 2, 2]),
            16: ([2, 2, 3, 3, 3]),
            19: ([2, 2, 4, 4, 4])
        }
        assert layers in vgg_spec.keys(), \
            "supported layers are {} but input layer is {}".format(vgg_spec.keys(), layers)

        nums = vgg_spec[layers]
        self.conv1 = self.conv_block(3, 64, nums[0], name="conv1_", use_bias=True if torch_version else False)
        self.conv2 = self.conv_block(64, 128, nums[1], name="conv2_", use_bias=True if torch_version else False)
        self.conv3 = self.conv_block(128, 256, nums[2], name="conv3_", use_bias=True if torch_version else False)
        self.conv4 = self.conv_block(256, 512, nums[3], name="conv4_", use_bias=True if torch_version else False)
        self.conv5 = self.conv_block(512, 512, nums[4], name="conv5_", use_bias=True if torch_version else False)
        _a = fluid.ParamAttr(
            initializer=fluid.initializer.NumpyArrayInitializer(np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)),
            trainable=False)
        self.mean = self.create_parameter(shape=(1, 3, 1, 1), attr=_a, dtype="float32")
        _a = fluid.ParamAttr(
            initializer=fluid.initializer.NumpyArrayInitializer(np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)),
            trainable=False)
        self.std = self.create_parameter(shape=(1, 3, 1, 1), attr=_a, dtype="float32")
        if not requires_grad:
            for param in self.parameters():
                param.stop_gradient = True

    def forward(self, x):
        x = (x - self.mean) / self.std
        feat, feat_1 = self.conv1(x)
        feat, feat_2 = self.conv2(feat)
        feat, feat_3 = self.conv3(feat)
        feat, feat_4 = self.conv4(feat)
        _, feat_5 = self.conv5(feat)
        return [feat_1, feat_2, feat_3, feat_4, feat_5]


def VGG11():
    model = VGGNet(layers=11)
    return model


def VGG13():
    model = VGGNet(layers=13)
    return model


def VGG16():
    model = VGGNet(layers=16)
    return model


def VGG19():
    model = VGGNet(layers=19)
    return model
