from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout2d, Dropout, AvgPool2d, \
    MaxPool2d, AdaptiveAvgPool2d, Sequential, Module, Parameter
import torch.nn.functional as F
import torch
from collections import namedtuple
import math
import pdb
from models.cbam import CBAM
import torch.nn as nn


##################################  Original Arcface Model #############################################################

class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


class SEModule(Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(
            channels, channels // reduction, kernel_size=1, padding=0, bias=False)
        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(
            channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


# i = 0

class bottleneck_IR(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False), BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False), PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False), BatchNorm2d(depth))
        i = 0

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        # print(shortcut.shape)
        # print('---s---')
        res = self.res_layer(x)
        # print(res.shape)
        # print('---r---')
        # i = i + 50
        # print(i)
        # print('50')
        return res + shortcut


class bottleneck_IR_SE(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR_SE, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            BatchNorm2d(depth),
            SEModule(depth, 16)
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    '''A named tuple describing a ResNet block.'''
    # print('50')


def get_block(in_channel, depth, num_units, stride=2):
    return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1) for i in range(num_units - 1)]


def get_blocks(num_layers):
    if num_layers == 50:
        # Initialize blocks1, blocks2, blocks3 first
        blocks1 = [
            get_block(in_channel=64, depth=64, num_units=3),
        ]
        blocks2 = [
            get_block(in_channel=64, depth=128, num_units=4),
        ]
        blocks3 = [
            get_block(in_channel=128, depth=256, num_units=14),
        ]

    elif num_layers == 100:
        return [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    
    elif num_layers == 152:
        return [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=8),
            get_block(in_channel=128, depth=256, num_units=36),
            get_block(in_channel=256, depth=512, num_units=3)
        ]

    # Flatten blocks to avoid nested lists
    blocks1 = [b for sublist in blocks1 for b in sublist]
    blocks2 = [b for sublist in blocks2 for b in sublist]
    blocks3 = [b for sublist in blocks3 for b in sublist]

    return blocks1, blocks2, blocks3



class Backbone(Module):
    def __init__(self, num_layers, drop_ratio, mode='ir'):
        super(Backbone, self).__init__()
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks1, blocks2, blocks3 = get_blocks(num_layers)
        
        if mode == 'ir':
            unit_module = bottleneck_IR
        else:
            unit_module = bottleneck_IR_SE
        
        self.input_layer = nn.Sequential(nn.Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                         nn.BatchNorm2d(64),
                                         nn.PReLU(64))
        print("blocks1:", blocks1)
        print("blocks1 type:", type(blocks1))
        self.body1 = nn.Sequential(*[unit_module(b.in_channel, b.depth, b.stride) for b in blocks1])
        #self.cbam1 = CBAM(64)
        self.body2 = nn.Sequential(*[unit_module(b.in_channel, b.depth, b.stride) for b in blocks2])
        #self.cbam2 = CBAM(128)
        self.body3 = nn.Sequential(*[unit_module(b.in_channel, b.depth, b.stride) for b in blocks3])
        #self.cbam3 = CBAM(256)

    def forward(self, x):
        x = F.interpolate(x, size=112)
        x = self.input_layer(x)
        x1 = self.body1(x)
        #x1 = self.cbam1(x1)
        x2 = self.body2(x1)
        #x2 = self.cbam2(x2)
        x3 = self.body3(x2)
        #x3 = self.cbam3(x3)
        
        return x1, x2, x3

def load_pretrained_weights(model, checkpoint):
    import collections
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    model_dict = model.state_dict()
    new_state_dict = collections.OrderedDict()
    matched_layers, discarded_layers = [], []
    for i, (k, v) in enumerate(state_dict.items()):
        # print(i)

        # If the pretrained state_dict was saved as nn.DataParallel,
        # keys would contain "module.", which should be ignored.
        if k.startswith('module.'):
            k = k[7:]
        if k in model_dict and model_dict[k].size() == v.size():

            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            # print(k)
            discarded_layers.append(k)
    # new_state_dict.requires_grad = False
    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)
    print('load_weight', len(matched_layers))
    return model

# model = Backbone(50, 0.0, 'ir')
# ir_checkpoint = torch.load(r'C:\Users\86187\Desktop\project\mixfacial\models\pretrain\new_ir50.pth')
# print('hello')
# i1, i2, i3 = 0, 0, 0
# ir_checkpoint = torch.load(r'C:\Users\86187\Desktop\project\mixfacial\models\pretrain\ir50.pth', map_location=lambda storage, loc: storage)
# for (k1, v1), (k2, v2) in zip(model.state_dict().items(), ir_checkpoint.items()):
#     print(f'k1:{k1}, k2:{k2}')
#     model.state_dict()[k1] = v2

# torch.save(model.state_dict(), r'C:\Users\86187\Desktop\project\mixfacial\models\pretrain\new_ir50.pth')
#     print(k)
#     if k.startswith('body1'):
#         i1+=1
#     if k.startswith('body2'):
#         i2+=1
#     if k.startswith('body3'):
#         i3+=1
# print(f'i1:{i1}, i2:{i2}, i3:{i3}')

# print('-'*100)
# ir_checkpoint = torch.load(r'C:\Users\86187\Desktop\project\mixfacial\models\pretrain\ir50.pth', map_location=lambda storage, loc: storage)
# le = 0
# for k, v in ir_checkpoint.items():
#     # print(k)
#     if k.startswith('body'):
#         if le < i1:
#             le += 1
#             key = k.split('.')[0] + str(1) + k.split('.')[1:]
#             print(key)
# # ir_checkpoint = ir_checkpoint["model"]
# model = load_pretrained_weights(model, ir_checkpoint)
# img = torch.rand(size=(2,3,224,224))
# out1, out2, out3 = model(img)
# print(out1.shape, out2.shape, out3.shape)
