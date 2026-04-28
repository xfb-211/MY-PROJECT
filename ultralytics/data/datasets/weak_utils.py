"""
Weak Supervision Utilities for Pseudo Label Management
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import torch.nn.functional as F


class AdaptivePseudoLabelManager:
    """自适应伪标签管理器"""

    def __init__(self, config: Dict = None):
        self.config = config or {
            'initial_threshold': 0.5,
            'update_interval': 5,
            'min_confidence_gain': 0.1,
            'max_labels_per_image': 30,
            'stability_window': 10
        }

        # 训练状态跟踪
        self.history = {
            'mAP': deque(maxlen=self.config['stability_window']),
            'loss': deque(maxlen=self.config['stability_window'])
        }

        # 伪标签存储
        self.pseudo_labels = {}

    def should_update(self, epoch: int, metrics: Dict) -> bool:
        """判断是否需要更新伪标签"""
        # 更新历史
        self.history['mAP'].append(metrics.get('mAP', 0.0))
        self.history['loss'].append(metrics.get('loss', 0.0))

        # 基础规则：固定间隔
        if epoch % self.config['update_interval'] == 0:
            return True

        # 自适应规则：性能下降
        if len(self.history['mAP']) >= 3:
            recent_mAP = list(self.history['mAP'])[-3:]
            if recent_mAP[-1] < recent_mAP[0] - 0.02:
                return True

        return False

    def update_pseudo_labels(self, model: nn.Module, dataloader,
                             current_labels: Optional[Dict] = None) -> Dict:
        """更新伪标签"""
        model.eval()
        device = next(model.parameters()).device

        new_pseudo_labels = {
            'boxes': [],
            'scores': [],
            'class_ids': []
        }

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                images = batch['img'].to(device)

                # 模型预测
                outputs = model(images)

                # 解析预测结果
                pred_boxes = outputs['pred_boxes']  # [B, N, 4]
                pred_scores = outputs['pred_scores']  # [B, N]
                pred_classes = outputs['pred_classes']  # [B, N]

                B = pred_boxes.size(0)

                for b in range(B):
                    # 过滤高置信度预测
                    mask = pred_scores[b] >= self.config['initial_threshold']

                    if mask.any():
                        boxes = pred_boxes[b][mask]
                        scores = pred_scores[b][mask]
                        classes = pred_classes[b][mask]

                        # 非极大值抑制
                        keep = self._nms(boxes, scores)

                        # 限制标签数量
                        if len(keep) > self.config['max_labels']:
                            keep = keep[:self.config['max_labels']]

                        new_pseudo_labels['boxes'].append(boxes[keep])
                        new_pseudo_labels['scores'].append(scores[keep])
                        new_pseudo_labels['class_ids'].append(classes[keep])
                    else:
                        # 如果没有预测，使用原标签或空标签
                        if current_labels and batch_idx * B + b < len(current_labels['boxes']):
                            idx = batch_idx * B + b
                            new_pseudo_labels['boxes'].append(current_labels['boxes'][idx])
                            new_pseudo_labels['scores'].append(current_labels['scores'][idx])
                            new_pseudo_labels['class_ids'].append(current_labels['class_ids'][idx])
                        else:
                            new_pseudo_labels['boxes'].append(torch.empty(0, 4, device=device))
                            new_pseudo_labels['scores'].append(torch.empty(0, device=device))
                            new_pseudo_labels['class_ids'].append(torch.empty(0, device=device))

        model.train()
        return new_pseudo_labels

    # def update_pseudo_labels(self, model: nn.Module, dataloader,
    #                          current_labels: Optional[Dict] = None) -> Dict:
    #     """更新伪标签（最终版：复用rtdetr-weak.yaml已有参数，适配VisDrone+小目标优化）"""
    #     model.eval()
    #     device = next(model.parameters()).device
    #
    #     new_pseudo_labels = {
    #         'boxes': [],
    #         'scores': [],
    #         'class_ids': []
    #     }
    #
    #     # ========== 核心修改：读取rtdetr-weak.yaml的层级参数（无新变量） ==========
    #     # 从weak_supervision.pseudo_label读取已有参数
    #     pseudo_cfg = self.config.get('weak_supervision', {}).get('pseudo_label', {})
    #     base_thresh = pseudo_cfg.get('initial_threshold', 0.5)  # 读取0.5的阈值
    #     max_labels = pseudo_cfg.get('max_labels_per_image', 30)  # 读取单图最大标签数30
    #     num_classes = self.config.get('nc', 10)  # 读取nc=10（类别数）
    #     max_labels_per_class = max_labels // num_classes  # 每类最大保留数（30//10=3）
    #     # ========== 参数读取结束 ==========
    #
    #     with torch.no_grad():
    #         for batch_idx, batch in enumerate(dataloader):
    #             images = batch['img'].to(device)
    #
    #             # 模型预测（复用原有推理逻辑）
    #             outputs = model(images)
    #             pred_boxes = outputs['pred_boxes']  # [B, N, 4]
    #             pred_scores = outputs['pred_scores']  # [B, N]
    #             pred_classes = outputs['pred_classes']  # [B, N]
    #
    #             B = pred_boxes.size(0)
    #
    #             for b in range(B):
    #                 # 第一步：基础置信度筛选（复用rtdetr-weak.yaml的0.5阈值）
    #                 conf_mask = pred_scores[b] >= base_thresh
    #
    #                 if conf_mask.any():
    #                     # 筛选后基础数据
    #                     boxes = pred_boxes[b][conf_mask]
    #                     scores = pred_scores[b][conf_mask]
    #                     classes = pred_classes[b][conf_mask]
    #
    #                     # 第二步：类别均衡采样（核心优化）
    #                     keep = []
    #                     # 遍历所有出现的类别，每个类别按置信度取前3个（30//10=3）
    #                     for cls in torch.unique(classes):
    #                         cls_mask = (classes == cls)
    #                         cls_indices = torch.where(cls_mask)[0]
    #                         if len(cls_indices) == 0:
    #                             continue
    #                         # 按置信度降序排序
    #                         cls_scores = scores[cls_mask]
    #                         sorted_indices = cls_indices[torch.argsort(cls_scores, descending=True)]
    #                         # 取每类最大保留数（3个）
    #                         topk_indices = sorted_indices[:max_labels_per_class]
    #                         keep.extend(topk_indices.tolist())
    #                     # 转换为tensor（复用device）
    #                     keep = torch.tensor(keep, device=device) if keep else torch.empty(0, dtype=torch.long,
    #                                                                                       device=device)
    #
    #                     # ========== 新增：小目标专属筛选（适配VisDrone，无新变量） ==========
    #                     if len(keep) > 0:
    #                         # 提取保留框的坐标和面积
    #                         keep_boxes = boxes[keep]
    #                         keep_scores = scores[keep]
    #                         keep_classes = classes[keep]
    #
    #                         # 计算框面积（VisDrone小目标定义：<32×32像素）
    #                         box_w = keep_boxes[:, 2] - keep_boxes[:, 0]
    #                         box_h = keep_boxes[:, 3] - keep_boxes[:, 1]
    #                         box_area = box_w * box_h
    #
    #                         # 划分小/中/大目标
    #                         small_obj_mask = box_area < 32 * 32  # 小目标
    #                         medium_obj_mask = (box_area >= 32 * 32) & (box_area < 96 * 96)  # 中目标
    #                         large_obj_mask = box_area >= 96 * 96  # 大目标
    #
    #                         # 小目标优先保留：每类保留数×1.5，全局占比60%
    #                         final_keep = []
    #                         # 处理小目标
    #                         if small_obj_mask.any():
    #                             small_indices = torch.where(small_obj_mask)[0]
    #                             small_scores = keep_scores[small_indices]
    #                             # 小目标保留数：每类3个×1.5=4个（向下取整）
    #                             small_topk = min(len(small_indices), int(max_labels_per_class * 1.5))
    #                             final_keep.extend(small_indices[torch.argsort(small_scores, descending=True)[:small_topk]].tolist())
    #                         # 处理中目标
    #                         if medium_obj_mask.any():
    #                             medium_indices = torch.where(medium_obj_mask)[0]
    #                             medium_scores = keep_scores[medium_indices]
    #                             medium_topk = min(len(medium_indices), max_labels_per_class)
    #                             final_keep.extend(medium_indices[torch.argsort(medium_scores, descending=True)[:medium_topk]].tolist())
    #                         # 处理大目标
    #                         if large_obj_mask.any():
    #                             large_indices = torch.where(large_obj_mask)[0]
    #                             large_scores = keep_scores[large_indices]
    #                             large_topk = min(len(large_indices), max_labels_per_class)
    #                             final_keep.extend(large_indices[torch.argsort(large_scores, descending=True)[:large_topk]].tolist())
    #
    #                         # 转换为tensor
    #                         final_keep = torch.tensor(final_keep, device=device) if final_keep else torch.empty(0, dtype=torch.long, device=device)
    #
    #                         # 全局数量限制：小目标占60%，中+大占40%
    #                         if len(final_keep) > max_labels:
    #                             # 拆分小/非小目标
    #                             small_final = final_keep[small_obj_mask[final_keep]] if small_obj_mask.any() else torch.empty(0, dtype=torch.long, device=device)
    #                             non_small_final = final_keep[~small_obj_mask[final_keep]] if (~small_obj_mask).any() else torch.empty(0, dtype=torch.long, device=device)
    #
    #                             # 小目标保留60%，非小保留40%
    #                             small_topk_global = min(len(small_final), int(max_labels * 0.5))
    #                             non_small_topk_global = min(len(non_small_final), max_labels - small_topk_global)
    #
    #                             # 按置信度排序取topk
    #                             small_keep = small_final[torch.argsort(keep_scores[small_final], descending=True)[:small_topk_global]] if len(small_final) > 0 else torch.empty(0, dtype=torch.long, device=device)
    #                             non_small_keep = non_small_final[torch.argsort(keep_scores[non_small_final], descending=True)[:non_small_topk_global]] if len(non_small_final) > 0 else torch.empty(0, dtype=torch.long, device=device)
    #
    #                             # 合并最终保留索引
    #                             keep = torch.cat([small_keep, non_small_keep]) if (len(small_keep) > 0 or len(non_small_keep) > 0) else torch.empty(0, dtype=torch.long, device=device)
    #                     # ========== 小目标筛选结束 ==========
    #
    #                     # 第三步：全局数量限制（兜底，复用max_labels_per_image=30）
    #                     if len(keep) > max_labels:
    #                         keep_scores = scores[keep]
    #                         topk_global = torch.argsort(keep_scores, descending=True)[:max_labels]
    #                         keep = keep[topk_global]
    #
    #                     # 第四步：NMS去重（轻量版，复用原有方法）
    #                     if len(keep) > 0:
    #                         keep_nms = self._nms(boxes[keep], scores[keep])
    #                         keep = keep[keep_nms]
    #
    #                     # 填充伪标签（复用原有逻辑）
    #                     new_pseudo_labels['boxes'].append(
    #                         boxes[keep] if len(keep) > 0 else torch.empty(0, 4, device=device))
    #                     new_pseudo_labels['scores'].append(
    #                         scores[keep] if len(keep) > 0 else torch.empty(0, device=device))
    #                     new_pseudo_labels['class_ids'].append(
    #                         classes[keep] if len(keep) > 0 else torch.empty(0, device=device))
    #                 else:
    #                     # 无有效伪标签时，复用当前标签兜底（原有逻辑）
    #                     if current_labels and batch_idx * B + b < len(current_labels['boxes']):
    #                         idx = batch_idx * B + b
    #                         new_pseudo_labels['boxes'].append(current_labels['boxes'][idx])
    #                         new_pseudo_labels['scores'].append(current_labels['scores'][idx])
    #                         new_pseudo_labels['class_ids'].append(current_labels['class_ids'][idx])
    #                     else:
    #                         new_pseudo_labels['boxes'].append(torch.empty(0, 4, device=device))
    #                         new_pseudo_labels['scores'].append(torch.empty(0, device=device))
    #                         new_pseudo_labels['class_ids'].append(torch.empty(0, device=device))
    #
    #     model.train()
    #     return new_pseudo_labels

    @staticmethod
    def _nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.5) -> torch.Tensor:
        """非极大值抑制"""
        if boxes.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=boxes.device)

        # 按置信度排序
        sorted_scores, indices = torch.sort(scores, descending=True)
        sorted_boxes = boxes[indices]

        # 计算IoU
        ious = box_iou(sorted_boxes, sorted_boxes)

        # NMS算法
        keep = []
        while indices.numel() > 0:
            i = indices[0]
            keep.append(i)

            if indices.numel() == 1:
                break

            # 计算与剩余框的IoU
            iou = ious[0, 1:]

            # 保留IoU低于阈值的框
            keep_mask = iou <= iou_threshold
            indices = indices[1:][keep_mask]
            ious = ious[1:][:, 1:][:, keep_mask]

        return torch.tensor(keep, device=boxes.device)


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """计算IoU矩阵"""
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou
