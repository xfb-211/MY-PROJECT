"""
Pseudo-Guided Attention Modules for Weakly Supervised RT-DETR
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
import math


class AdaptiveMaskGenerator(nn.Module):
    """自适应伪标签掩码生成器"""

    def __init__(self, scale_config: Dict[str, float] = None):
        super().__init__()
        self.scale_config = scale_config or {
            'S3': {'sigma': 1.5, 'dilation': 1, 'threshold': 0.5},  # 小目标
            'S4': {'sigma': 2.0, 'dilation': 1, 'threshold': 0.6},  # 中目标
            'S5': {'sigma': 3.0, 'dilation': 1, 'threshold': 0.7}  # 大目标
        }

    def forward(self, pseudo_labels: Dict[str, torch.Tensor],
                feature_size: Tuple[int, int],
                scale_name: str) -> torch.Tensor:
        """
        生成基于伪标签的自适应注意力掩码

        Args:
            pseudo_labels: 伪标签字典，包含'boxes'和'scores'
            feature_size: (H, W)特征图尺寸
            scale_name: 尺度名称 'S3'/'S4'/'S5'

        Returns:
            mask: [B, 1, H, W] 注意力掩码
        """
        B = pseudo_labels['boxes'].shape[0]
        H, W = feature_size
        device = pseudo_labels['boxes'].device

        # 初始化掩码
        mask = torch.zeros(B, 1, H, W, device=device)
        config = self.scale_config[scale_name]

        for b in range(B):
            boxes = pseudo_labels['boxes'][b]  # [N, 4]
            scores = pseudo_labels['scores'][b]  # [N]

            for box, score in zip(boxes, scores):
                if score < config['threshold']:
                    continue

                # 边界框归一化到特征图坐标
                x1, y1, x2, y2 = self._normalize_box(box, H, W)

                # 生成高斯掩码
                gaussian = self._create_gaussian_mask(H, W, x1, y1, x2, y2,
                                                      config['sigma'] * score)
                mask[b, 0] = torch.max(mask[b, 0], gaussian)

        return mask

    def _normalize_box(self, box: torch.Tensor, H: int, W: int) -> Tuple[float, float, float, float]:
        """将边界框归一化到特征图坐标"""
        x1 = max(0, min(W - 1, box[0] * W))
        y1 = max(0, min(H - 1, box[1] * H))
        x2 = max(0, min(W - 1, box[2] * W))
        y2 = max(0, min(H - 1, box[3] * H))
        return x1, y1, x2, y2

    def _create_gaussian_mask(self, H: int, W: int, x1: float, y1: float,
                              x2: float, y2: float, sigma: float) -> torch.Tensor:
        """创建二维高斯掩码"""
        # 计算边界框中心
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        # 创建坐标网格
        y = torch.arange(0, H, dtype=torch.float32, device=x1.device).view(-1, 1)
        x = torch.arange(0, W, dtype=torch.float32, device=x1.device).view(1, -1)

        # 二维高斯分布
        gaussian = torch.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))
        return gaussian


class PseudoAIFI(nn.Module):
    """伪标签引导的注意力特征交互模块"""

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"

        # 注意力层
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        # 掩码生成器
        self.mask_generator = AdaptiveMaskGenerator()

        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )

        # 层归一化
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, scale_info: Dict = None,
                pseudo_labels: Optional[Dict] = None) -> torch.Tensor:
        """
        Args:
            x: 输入特征 [B, N, C]
            scale_info: 尺度信息 {'name': str, 'stride': int}
            pseudo_labels: 伪标签信息
        """
        B, N, C = x.shape

        # 保存残差连接
        residual = x

        # 层归一化
        x = self.norm1(x)

        # 生成QKV
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力分数
        attn_scores = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)

        # 应用伪标签引导掩码
        if pseudo_labels is not None and scale_info is not None:
            # 重塑为空间特征以生成掩码
            H = int(math.sqrt(N)) if N > 0 else 1
            W = N // H
            spatial_mask = self.mask_generator(pseudo_labels, (H, W), scale_info['name'])

            # 将空间掩码转换为序列掩码
            seq_mask = spatial_mask.reshape(B, 1, N).unsqueeze(1)  # [B, 1, 1, N]

            # 应用到注意力分数
            attn_scores = attn_scores + seq_mask * 10.0  # 增强目标区域权重

        # 注意力权重
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 注意力输出
        attn_output = (attn_weights @ v).transpose(1, 2).reshape(B, N, C)
        attn_output = self.out_proj(attn_output)
        attn_output = self.dropout(attn_output)

        # 残差连接
        x = residual + attn_output

        # 前馈网络
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x

        return x


class WeakSupervisionWrapper(nn.Module):
    """弱监督包装器，用于在训练时注入伪标签信息"""

    def __init__(self, module: nn.Module, module_type: str = 'aifi'):
        super().__init__()
        self.module = module
        self.module_type = module_type
        self.requires_pseudo = True

    def forward(self, x: torch.Tensor, **kwargs):
        if self.module_type == 'aifi' and 'pseudo_labels' in kwargs:
            return self.module(x, **kwargs)
        else:
            # 如果没有伪标签，使用原始模块
            return self.module(x)
