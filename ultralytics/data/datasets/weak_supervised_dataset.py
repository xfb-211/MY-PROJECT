"""
Weakly Supervised Dataset for RT-DETR
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Dict, List, Tuple, Optional
import os
import json
from pathlib import Path


class WeakSupervisedDataset(Dataset):
    """弱监督数据集，支持伪标签动态更新"""

    def __init__(self, root_dir: str, annotation_file: str,
                 pseudo_labels: Optional[Dict] = None,
                 transform=None):
        """
        Args:
            root_dir: 图像根目录
            annotation_file: 弱标注文件（仅类别标签）
            pseudo_labels: 伪标签字典
            transform: 数据增强
        """
        self.root_dir = Path(root_dir)
        self.transform = transform

        # 加载弱标注
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)

        # 初始化伪标签
        self.pseudo_labels = pseudo_labels or self._init_pseudo_labels()

        # 图像列表
        self.image_ids = list(self.annotations.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        # 加载图像
        img_path = self.root_dir / f"{image_id}.jpg"
        image = self._load_image(img_path)

        # 获取弱标签（类别）
        weak_labels = self.annotations[image_id]

        # 获取伪标签（边界框）
        pseudo_info = self.pseudo_labels.get(image_id, {
            'boxes': np.empty((0, 4), dtype=np.float32),
            'scores': np.empty((0,), dtype=np.float32),
            'class_ids': np.empty((0,), dtype=np.int64)
        })

        sample = {
            'img': image,
            'img_path': str(img_path),
            'weak_labels': weak_labels,
            'pseudo_boxes': pseudo_info['boxes'],
            'pseudo_scores': pseudo_info['scores'],
            'pseudo_classes': pseudo_info['class_ids']
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def update_pseudo_labels(self, new_pseudo_labels: Dict):
        """更新伪标签"""
        self.pseudo_labels.update(new_pseudo_labels)

    def _load_image(self, path: Path):
        """加载图像（简化版本）"""
        # 这里应该使用实际的图像加载逻辑
        # 例如: from PIL import Image
        # return Image.open(path).convert('RGB')
        return np.random.rand(640, 640, 3)  # 占位符

    def _init_pseudo_labels(self) -> Dict:
        """初始化伪标签"""
        pseudo_labels = {}
        for img_id in self.image_ids:
            pseudo_labels[img_id] = {
                'boxes': np.empty((0, 4), dtype=np.float32),
                'scores': np.empty((0,), dtype=np.float32),
                'class_ids': np.empty((0,), dtype=np.int64)
            }
        return pseudo_labels


# import os
# import json
# import torch
# import yaml
# import numpy as np
# from PIL import Image
# from ultralytics.data.dataset import YOLODataset
# from ultralytics.data.augment import Compose, v8_transforms
# from torchvision.models import MobileNet_V2_Weights
# from torchvision.models import mobilenet_v2
# from torchvision.models.feature_extraction import create_feature_extractor
#
#
# class WeakSupervisedDataset(YOLODataset):
#     def __init__(self, img_path, label_path, weak_label=True, pseudo_label_path=None, augment=False, **kwargs):
#         super().__init__(img_path, label_path, **kwargs)
#         self.weak_label = weak_label  # 是否为弱监督标签（仅类别）
#         self.pseudo_labels = self.load_pseudo_labels(pseudo_label_path) if pseudo_label_path else None
#         self.augment = augment
#         self.transform = Compose(v8_transforms(flipud=0.2, fliplr=0.5, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0))
#
#     def load_pseudo_labels(self, pseudo_label_path):
#         """加载伪标签（格式：{image_name: [[class_id, x1, y1, x2, y2, conf], ...]}）"""
#         with open(pseudo_label_path, 'r') as f:
#             return json.load(f)
#
#     def __getitem__(self, idx):
#         img_path = self.im_files[idx]
#         img = Image.open(img_path).convert('RGB')
#         img_name = os.path.basename(img_path)
#
#         # 1. 加载标签（弱监督标签→仅类别；伪标签→完整框+类别+置信度）
#         if self.pseudo_labels and img_name in self.pseudo_labels:
#             # 伪标签：class_id x1 y1 x2 y2 conf（已过滤低置信度）
#             labels = torch.tensor(self.pseudo_labels[img_name], dtype=torch.float32)
#         else:
#             # 弱监督标签：仅class_id，生成占位框（后续由模型优化）
#             cls_labels = super().load_labels(idx)[:, 0:1]  # 仅取类别
#             placeholder_boxes = torch.ones((len(cls_labels), 4)) * 0.5  # 占位框（0.5,0.5,0.5,0.5）
#             labels = torch.cat([cls_labels, placeholder_boxes, torch.ones((len(cls_labels), 1))],
#                                dim=1)  # 最后一列是置信度（初始1.0）
#
#         # 2. 数据增强（同步变换标签）
#         if self.augment:
#             img, labels = self.transform(img, labels)
#
#         # 3. 格式转换（适配Ultralytics输入）
#         img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
#         boxes = labels[:, 1:5]  # x1 y1 x2 y2（相对坐标）
#         cls = labels[:, 0].long()
#         conf = labels[:, 5] if labels.shape[1] >= 6 else torch.ones_like(cls).float()
#
#         return img, boxes, cls, conf
#
#
# # ------------------------------
# # 辅助函数：用MobileNetV2+MIL生成初始伪标签
# # ------------------------------
# class MILClassifier(torch.nn.Module):
#     """多实例学习（MIL）分类器，用于弱监督标签生成"""
#
#     def __init__(self, num_classes=10):
#         super().__init__()
#         # self.backbone = mobilenet_v2(pretrained=True)
#         self.backbone = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
#         # return_nodes与forward读取的键保持一致（选classifier.0，MobileNetV2的特征输出节点）
#         self.feature_extractor = create_feature_extractor(self.backbone, return_nodes=['classifier.0'])
#         self.classifier = torch.nn.Linear(1280, num_classes)  # MobileNetV2 avgpool输出1280维
#
#     def forward(self, imgs):
#         """imgs: (batch, 3, 640, 640)"""
#         # 修正：将['avgpool']改为['classifier.0']，匹配return_nodes的节点名
#         features = self.feature_extractor(imgs)['classifier.0'].squeeze(-1).squeeze(-1)  # (batch, 1280)
#         logits = self.classifier(features)  # (batch, num_classes)
#         return torch.sigmoid(logits)
#
#
# def generate_initial_pseudo_labels(data_yaml, output_path, conf_thresh=0.6):
#     """生成初始伪标签"""
#     data_yaml = os.path.abspath(data_yaml)
#     if not os.path.exists(data_yaml):
#         raise FileNotFoundError(f"yaml配置文件不存在，请检查路径：{data_yaml}")
#
#     with open(data_yaml, 'r', encoding='utf-8') as f:
#         dataset_info = yaml.safe_load(f)
#
#     # 拼接train目录完整路径，遍历目录下的图像文件
#     img_root = dataset_info['path']
#     train_dir = os.path.join(img_root, dataset_info['train'])  # 拼接train目录完整路径
#     # 过滤图像文件（支持常见图像后缀），生成有效图像路径列表
#     img_paths = [
#         os.path.join(train_dir, fname)
#         for fname in os.listdir(train_dir)
#         if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
#     ]
#
#     print(f"生成图像路径列表，共 {len(img_paths)} 张图像")
#     if img_paths:  # 避免空列表索引报错
#         print(f"示例路径：{img_paths[0]}")
#
#     # 后续逻辑保持不变
#     num_classes = dataset_info.get('nc', 10)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = MILClassifier(num_classes=dataset_info['nc']).to(device)
#     model.eval()
#
#     pseudo_labels = {}
#     with torch.no_grad():
#         for img_path in img_paths:
#             if not os.path.exists(img_path):
#                 print(f"警告：图像文件不存在，跳过：{img_path}")
#                 continue
#             img = Image.open(img_path).convert('RGB').resize((640, 640))
#             # 补充：修复tensor拼接语法错误（原代码.unsqueeze(0)前多了一个.）
#             img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
#             img_tensor = img_tensor.unsqueeze(0).to(device)
#             cls_conf = model(img_tensor).squeeze(0)
#
#             valid_cls = torch.where(cls_conf >= conf_thresh)[0]
#             if len(valid_cls) == 0:
#                 continue
#             pseudo_boxes = torch.tensor([[0.25, 0.25, 0.75, 0.75]] * len(valid_cls),
#                                         dtype=torch.float32,
#                                         device=device)  # 新增device参数
#             pseudo_labels[os.path.basename(img_path)] = torch.cat([
#                 valid_cls.unsqueeze(1).float(),
#                 pseudo_boxes,
#                 cls_conf[valid_cls].unsqueeze(1)
#             ], dim=1).tolist()
#
#     with open(output_path, 'w') as f:
#         json.dump(pseudo_labels, f)
#     print(f"初始伪标签已保存到：{output_path}")
