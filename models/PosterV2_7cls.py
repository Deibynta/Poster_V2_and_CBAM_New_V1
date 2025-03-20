import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from .mobilefacenet import MobileFaceNet
from .ir50 import Backbone
from .vit_model import VisionTransformer, PatchEmbed
from timm.models.layers import trunc_normal_, DropPath
from thop import profile
from models.cbam import CBAM


def load_pretrained_weights(model, checkpoint):
    import collections

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    model_dict = model.state_dict()
    new_state_dict = collections.OrderedDict()
    matched_layers, discarded_layers = [], []
    for k, v in state_dict.items():
        # If the pretrained state_dict was saved as nn.DataParallel,
        # keys would contain "module.", which should be ignored.
        if k.startswith("module."):
            k = k[7:]
        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)
    # new_state_dict.requires_grad = False
    model_dict.update(new_state_dict)

    model.load_state_dict(model_dict)
    print("load_weight", len(matched_layers))
    return model


def window_partition(x, window_size, h_w, w_w):
    """
    Args:
        x: (B, H, W, C)
        window_size: window size

    Returns:
        local window features (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, h_w, window_size, w_w, window_size, C)
    windows = (
        x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    )
    return windows


class window(nn.Module):
    def __init__(self, window_size, dim):
        super(window, self).__init__()
        self.window_size = window_size
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        B, H, W, C = x.shape
        x = self.norm(x)
        shortcut = x
        h_w = int(torch.div(H, self.window_size).item())
        w_w = int(torch.div(W, self.window_size).item())
        x_windows = window_partition(x, self.window_size, h_w, w_w)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        return x_windows, shortcut


class WindowAttentionGlobal(nn.Module):
    """
    Global window attention based on: "Hatamizadeh et al.,
    Global Context Vision Transformers <https://arxiv.org/abs/2206.09959>"
    """

    def __init__(
        self,
        dim,
        num_heads,
        window_size,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        """
        Args:
            dim: feature size dimension.
            num_heads: number of attention head.
            window_size: window size.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            attn_drop: attention dropout rate.
            proj_drop: output dropout rate.
        """

        super().__init__()
        window_size = (window_size, window_size)
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = torch.div(dim, num_heads)
        self.scale = qk_scale or head_dim**-0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        self.qkv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, q_global):
        # print(f'q_global.shape:{q_global.shape}')
        # print(f'x.shape:{x.shape}')
        B_, N, C = x.shape
        B = q_global.shape[0]
        head_dim = int(torch.div(C, self.num_heads).item())
        B_dim = int(torch.div(B_, B).item())
        kv = (
            self.qkv(x)
            .reshape(B_, N, 2, self.num_heads, head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv[0], kv[1]
        print(f"q_global before repeat: {q_global.shape}")
        q_global = q_global.repeat(1, B_dim, 1, 1, 1)
        print(f"q_global.shape before reshape: {q_global.shape}")
        print(f"Expected reshape: ({B_}, {self.num_heads}, {N}, {head_dim})")
        print(f"Total elements in q_global: {q_global.numel()}")
        print(f"Expected elements after reshape: {B_ * self.num_heads * N * head_dim}")
        if q_global.numel() != (B_ * self.num_heads * N * head_dim):
          print("⚠️ Reshape still mismatched! Using alternative approach.")
          q = q_global.view(B_, self.num_heads, 1, head_dim).expand(-1, -1, N, -1)  
        else:
          q = q_global.reshape(B_, self.num_heads, N, head_dim)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def _to_channel_last(x):
    """
    Args:
        x: (B, C, H, W)

    Returns:
        x: (B, H, W, C)
    """
    return x.permute(0, 2, 3, 1)


def _to_channel_first(x):
    return x.permute(0, 3, 1, 2)


def _to_query(x, N, num_heads, dim_head):
    B = x.shape[0]
    x = x.reshape(B, 1, N, num_heads, dim_head).permute(0, 1, 3, 2, 4)
    return x


class Mlp(nn.Module):
    """
    Multi-Layer Perceptron (MLP) block
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        """
        Args:
            in_features: input features dimension.
            hidden_features: hidden features dimension.
            out_features: output features dimension.
            act_layer: activation function.
            drop: dropout rate.
        """

        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_reverse(windows, window_size, H, W, h_w, w_w):
    """
    Args:
        windows: local window features (num_windows*B, window_size, window_size, C)
        window_size: Window size
        H: Height of image
        W: Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, h_w, w_w, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class feedforward(nn.Module):
    def __init__(
        self,
        dim,
        window_size,
        mlp_ratio=4.0,
        act_layer=nn.GELU,
        drop=0.0,
        drop_path=0.0,
        layer_scale=None,
    ):
        super(feedforward, self).__init__()
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.layer_scale = True
            self.gamma1 = nn.Parameter(
                layer_scale * torch.ones(dim), requires_grad=True
            )
            self.gamma2 = nn.Parameter(
                layer_scale * torch.ones(dim), requires_grad=True
            )
        else:
            self.gamma1 = 1.0
            self.gamma2 = 1.0
        self.window_size = window_size
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.norm = nn.LayerNorm(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, attn_windows, shortcut):
        B, H, W, C = shortcut.shape
        h_w = int(torch.div(H, self.window_size).item())
        w_w = int(torch.div(W, self.window_size).item())
        x = window_reverse(attn_windows, self.window_size, H, W, h_w, w_w)
        x = shortcut + self.drop_path(self.gamma1 * x)
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm(x)))
        return x


class pyramid_trans_expr2(nn.Module):
    def __init__(self, img_size=224, num_classes=7, window_size=[28, 14, 7], num_heads=[2, 4, 8], dims=[64, 128, 256], embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.num_heads = num_heads
        self.dim_head = [int(torch.div(dim, head).item()) for dim, head in zip(dims, num_heads)]
        self.num_classes = num_classes
        self.window_size = window_size
        self.N = [win * win for win in window_size]
        
        self.ir_back = Backbone(50, 0.0, 'ir')
        self.face_landback = MobileFaceNet([112, 112], 512)
        
        # Load pre-trained models
        mobilefacenet_path = os.path.join(os.getcwd(), "models/pretrain/mobilefacenet_cbam.pth")
        #ir50_path = os.path.join(os.getcwd(), "models/pretrain/ir50_cbam.pth")
        ir50_path = "/content/drive/MyDrive/Poster_V2_and_CBAM_New_V1-main/ir50_cbam.pth"
        
        self.face_landback.load_state_dict(torch.load(mobilefacenet_path, map_location=torch.device('cpu'))['state_dict'])
        checkpoint = torch.load(ir50_path, map_location=torch.device('cpu'))
        if "state_dict" in checkpoint:
             checkpoint = checkpoint["state_dict"]  # Extract actual state_dict
        self.ir_back.load_state_dict(checkpoint, strict=False)  # Use strict=False to ignore missing keys
        
        self.conv1 = nn.Conv2d(dims[0], dims[0], kernel_size=3, stride=2, padding=1)
        #self.cbam1 = CBAM(dims[0])
        self.conv2 = nn.Conv2d(dims[1], dims[1], kernel_size=3, stride=2, padding=1)
        #self.cbam2 = CBAM(dims[1])
        self.conv3 = nn.Conv2d(dims[2], dims[2], kernel_size=3, stride=2, padding=1)
        #self.cbam3 = CBAM(dims[2])
        
        self.attn1 = WindowAttentionGlobal(dim=dims[0], num_heads=num_heads[0], window_size=window_size[0])
        self.attn2 = WindowAttentionGlobal(dim=dims[1], num_heads=num_heads[1], window_size=window_size[1])
        self.attn3 = WindowAttentionGlobal(dim=dims[2], num_heads=num_heads[2], window_size=window_size[2])
        
        self.window1 = window(window_size=window_size[0], dim=dims[0])
        #self.cbam_window1 = CBAM(dims[0])
        self.window2 = window(window_size=window_size[1], dim=dims[1])
        #self.cbam_window2 = CBAM(dims[1])
        self.window3 = window(window_size=window_size[2], dim=dims[2])
        #self.cbam_window3 = CBAM(dims[2])
        
    def forward(self, x):
        x_face = F.interpolate(x, size=112)
        x_face1, x_face2, x_face3 = self.face_landback(x_face)
        x_ir1, x_ir2, x_ir3 = self.ir_back(x)
        '''
        #x_ir1, x_ir2, x_ir3 = self.cbam1(self.conv1(x_ir1)), self.cbam2(self.conv2(x_ir2)), self.cbam3(self.conv3(x_ir3))
        x_window1, shortcut1 = self.window1(x_ir1)
        x_window2, shortcut2 = self.window2(x_ir2)
        x_window3, shortcut3 = self.window3(x_ir3)
        
        o1, o2, o3 = self.attn1(x_window1, x_face1), self.attn2(x_window2, x_face2), self.attn3(x_window3, x_face3)
        #o1, o2, o3 = self.cbam_window1(o1), self.cbam_window2(o2), self.cbam_window3(o3)
        
        o1, o2, o3 = _to_channel_first(o1), _to_channel_first(o2), _to_channel_first(o3)
        
        o1, o2, o3 = self.embed_q(o1).flatten(2).transpose(1, 2), self.embed_k(o2).flatten(2).transpose(1, 2), self.embed_v(o3)
        o = torch.cat([o1, o2, o3], dim=1)
        
        out = self.VIT(o)'''

                x_face1, x_face2, x_face3 = (
            _to_channel_last(x_face1),
            _to_channel_last(x_face2),
            _to_channel_last(x_face3),
        )

        q1, q2, q3 = (
            _to_query(x_face1, self.N[0], self.num_heads[0], self.dim_head[0]),
            _to_query(x_face2, self.N[1], self.num_heads[1], self.dim_head[1]),
            _to_query(x_face3, self.N[2], self.num_heads[2], self.dim_head[2]),
        )

        x_ir1, x_ir2, x_ir3 = self.ir_back(x)

        x_ir1, x_ir2, x_ir3 = self.conv1(x_ir1), self.conv2(x_ir2), self.conv3(x_ir3)
        x_window1, shortcut1 = self.window1(x_ir1)
        x_window2, shortcut2 = self.window2(x_ir2)
        x_window3, shortcut3 = self.window3(x_ir3)

        o1, o2, o3 = (
            self.attn1(x_window1, q1),
            self.attn2(x_window2, q2),
            self.attn3(x_window3, q3),
        )

        o1, o2, o3 = (
            self.ffn1(o1, shortcut1),
            self.ffn2(o2, shortcut2),
            self.ffn3(o3, shortcut3),
        )

        o1, o2, o3 = _to_channel_first(o1), _to_channel_first(o2), _to_channel_first(o3)

        o1, o2, o3 = (
            self.embed_q(o1).flatten(2).transpose(1, 2),
            self.embed_k(o2).flatten(2).transpose(1, 2),
            self.embed_v(o3),
        )

        o = torch.cat([o1, o2, o3], dim=1)

        out = self.VIT(o)
        return out
    
def compute_param_flop():
    model = pyramid_trans_expr2()
    img = torch.rand(size=(1, 3, 224, 224))
    flops, params = profile(model, inputs=(img,))
    print(f"flops:{flops/1000**3}G,params:{params/1000**2}M")
