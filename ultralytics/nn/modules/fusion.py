"""
Cross-scale Feature Fusion Modules for Weakly Supervised RT-DETR
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional
import hashlib
from collections import OrderedDict


class DynamicPseudoCCFM(nn.Module):
    """动态伪标签引导的跨尺度特征融合模块"""

    def __init__(self, in_channels_list: List[int], out_channels: int = 256,
                 use_cache: bool = True, cache_size: int = 100):
        super().__init__()
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels
        self.use_cache = use_cache

        # 特征投影层
        self.projections = nn.ModuleList([
            nn.Conv2d(ch, out_channels, 1, bias=False) for ch in in_channels_list
        ])

        # 权重预测网络（轻量级）
        self.weight_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(sum(in_channels_list), 128),
            nn.ReLU(),
            nn.Linear(128, len(in_channels_list))
        )

        # 缓存机制
        self.cache = OrderedDict()
        self.cache_size = cache_size

    def forward(self, features: List[torch.Tensor],
                pseudo_labels: Optional[Dict] = None) -> torch.Tensor:
        """
        Args:
            features: 多尺度特征列表
            pseudo_labels: 伪标签信息
        """
        B = features[0].size(0)
        device = features[0].device

        # 生成缓存键
        cache_key = self._generate_cache_key(features, pseudo_labels)

        # 检查缓存
        if self.use_cache and cache_key in self.cache:
            weights = self.cache[cache_key].to(device)
            # 更新缓存顺序
            self.cache.move_to_end(cache_key)
        else:
            # 动态计算权重
            weights = self._compute_weights(features, pseudo_labels)

            # 更新缓存
            if self.use_cache:
                self._update_cache(cache_key, weights)

        # 特征投影
        projected = [proj(feat) for feat, proj in zip(features, self.projections)]

        # 统一分辨率（以最小分辨率为目标）
        target_size = projected[-1].shape[-2:]

        # 加权融合
        fused = torch.zeros(B, self.out_channels, *target_size, device=device)

        for i, feat in enumerate(projected):
            # 调整分辨率
            if feat.shape[-2:] != target_size:
                feat_resized = F.interpolate(feat, size=target_size, mode='bilinear',
                                             align_corners=False)
            else:
                feat_resized = feat

            # 应用权重
            weight = weights[:, i].view(B, 1, 1, 1).expand_as(feat_resized)
            fused = fused + weight * feat_resized

        return fused

    def _compute_weights(self, features: List[torch.Tensor],
                         pseudo_labels: Optional[Dict]) -> torch.Tensor:
        """计算融合权重"""
        B = features[0].size(0)
        device = features[0].device

        # 如果没有伪标签，使用均匀权重
        if pseudo_labels is None:
            num_scales = len(features)
            return torch.ones(B, num_scales, device=device) / num_scales

        # 基于伪标签的权重计算
        base_weights = []
        for i, feat in enumerate(features):
            _, _, H, W = feat.shape
            coverage = self._calculate_coverage(pseudo_labels, H, W)

            # Sigmoid函数将覆盖率映射到权重
            weight = torch.sigmoid(coverage * 3 - 1.5)  # [B, 1]
            base_weights.append(weight)

        base_weights = torch.cat(base_weights, dim=1)  # [B, num_scales]

        # 基于特征的权重微调
        feature_weights = []
        for feat in features:
            pooled = F.adaptive_avg_pool2d(feat, 1).flatten(1)
            feature_weights.append(pooled)

        feature_concat = torch.cat(feature_weights, dim=1)
        refined_weights = self.weight_predictor(feature_concat)  # [B, num_scales]

        # 结合两种权重
        combined = base_weights + 0.5 * refined_weights
        final_weights = F.softmax(combined, dim=1)

        return final_weights

    def _calculate_coverage(self, pseudo_labels: Dict, H: int, W: int) -> torch.Tensor:
        """计算伪标签在特征图上的覆盖率"""
        B = pseudo_labels['boxes'].size(0)
        device = pseudo_labels['boxes'].device

        coverage = torch.zeros(B, 1, device=device)

        for b in range(B):
            boxes = pseudo_labels['boxes'][b]  # [N, 4]
            scores = pseudo_labels['scores'][b]  # [N]

            total_area = 0.0
            for box, score in zip(boxes, scores):
                if score < 0.3:
                    continue

                # 归一化坐标到特征图
                x1 = max(0, min(W, box[0] * W))
                y1 = max(0, min(H, box[1] * H))
                x2 = max(0, min(W, box[2] * W))
                y2 = max(0, min(H, box[3] * H))

                area = (x2 - x1) * (y2 - y1)
                total_area += area * score  # 置信度加权

            coverage[b] = total_area / (H * W)

        return coverage

    def _generate_cache_key(self, features: List[torch.Tensor],
                            pseudo_labels: Optional[Dict]) -> str:
        """生成缓存键"""
        key_parts = []

        # 特征统计
        for feat in features:
            stats = f"shape{tuple(feat.shape)}_mean{feat.mean():.3f}"
            key_parts.append(stats)

        # 伪标签统计
        if pseudo_labels is not None:
            boxes = pseudo_labels['boxes']
            scores = pseudo_labels['scores']

            if boxes.numel() > 0:
                box_stats = f"boxes{boxes.shape}_mean{boxes.mean():.3f}"
                score_stats = f"scores{scores.shape}_mean{scores.mean():.3f}"
                key_parts.extend([box_stats, score_stats])

        # 生成哈希
        key_str = "_".join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()

    def _update_cache(self, key: str, weights: torch.Tensor):
        """更新缓存"""
        if len(self.cache) >= self.cache_size:
            # 移除最久未使用的
            self.cache.popitem(last=False)

        self.cache[key] = weights.detach().cpu()
