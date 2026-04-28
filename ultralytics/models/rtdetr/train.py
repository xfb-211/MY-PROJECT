# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from copy import copy

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils import RANK, colorstr

from .val import RTDETRDataset, RTDETRValidator


class RTDETRTrainer(DetectionTrainer):
    """Trainer class for the RT-DETR model developed by Baidu for real-time object detection.

    This class extends the DetectionTrainer class for YOLO to adapt to the specific features and architecture of
    RT-DETR. The model leverages Vision Transformers and has capabilities like IoU-aware query selection and adaptable
    inference speed.

    Attributes:
        loss_names (tuple): Names of the loss components used for training.
        data (dict): Dataset configuration containing class count and other parameters.
        args (dict): Training arguments and hyperparameters.
        save_dir (Path): Directory to save training results.
        test_loader (DataLoader): DataLoader for validation/testing data.

    Methods:
        get_model: Initialize and return an RT-DETR model for object detection tasks.
        build_dataset: Build and return an RT-DETR dataset for training or validation.
        get_validator: Return a DetectionValidator suitable for RT-DETR model validation.

    Examples:
        >>> from ultralytics.models.rtdetr.train import RTDETRTrainer
        >>> args = dict(model="rtdetr-l.yaml", data="coco8.yaml", imgsz=640, epochs=3)
        >>> trainer = RTDETRTrainer(overrides=args)
        >>> trainer.train()

    Notes:
        - F.grid_sample used in RT-DETR does not support the `deterministic=True` argument.
        - AMP training can lead to NaN outputs and may produce errors during bipartite graph matching.
    """

    def get_model(self, cfg: dict | None = None, weights: str | None = None, verbose: bool = True):
        """Initialize and return an RT-DETR model for object detection tasks.

        Args:
            cfg (dict, optional): Model configuration.
            weights (str, optional): Path to pre-trained model weights.
            verbose (bool): Verbose logging if True.

        Returns:
            (RTDETRDetectionModel): Initialized model.
        """
        model = RTDETRDetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def build_dataset(self, img_path: str, mode: str = "val", batch: int | None = None):
        """Build and return an RT-DETR dataset for training or validation.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): Dataset mode, either 'train' or 'val'.
            batch (int, optional): Batch size for rectangle training.

        Returns:
            (RTDETRDataset): Dataset object for the specific mode.
        """
        return RTDETRDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=self.args,
            rect=False,
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            prefix=colorstr(f"{mode}: "),
            classes=self.args.classes,
            data=self.data,
            fraction=self.args.fraction if mode == "train" else 1.0,
        )

    def get_validator(self):
        """Return a DetectionValidator suitable for RT-DETR model validation."""
        self.loss_names = "giou_loss", "cls_loss", "l1_loss"
        return RTDETRValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))

    # # 在RTDETRTrainer类中新增
    # def build_dataloader(self, mode="train"):
    #     if mode == "train":
    #         # 有标签数据加载器
    #         sup_loader = super().build_dataloader(mode)
    #         # 无标签数据加载器
    #         self.unlabeled_dataset = self.build_dataset(mode="unlabeled")
    #         self.unsup_loader = iter(self.build_dataloader(dataset=self.unlabeled_dataset, mode="unlabeled"))
    #         return sup_loader
    #     else:
    #         return super().build_dataloader(mode)
    #
    # # 重写训练步
    # def train_step(self, batch):
    #     # 加载无标签批次
    #     try:
    #         unsup_batch = next(self.unsup_loader)
    #     except StopIteration:
    #         self.unsup_loader = iter(self.build_dataloader(mode="unlabeled"))
    #         unsup_batch = next(self.unsup_loader)
    #
    #     # 调用半监督前向
    #     total_loss, loss_dict = self.model.semi_forward_step(
    #         x_sup=batch["img"],
    #         x_unsup=unsup_batch["img"],
    #         gt_boxes=batch["bboxes"],
    #         gt_labels=batch["cls"]
    #     )
    #
    #     # 反向传播
    #     self.scaler.scale(total_loss).backward()
    #     return loss_dict
    #
    # # 在 RTDETRTrainer 类中新增
    # def validate(self):
    #     """重写验证方法，在验证后更新动态阈值"""
    #     # 调用原生验证逻辑
    #     metrics = super().validate()
    #
    #     # 获取验证集 AP50（或 AP）
    #     current_ap = metrics.results_dict.get("metrics/mAP50(B)", 0.0)
    #
    #     # 更新伪标签生成器的动态阈值
    #     if hasattr(self.model, "pseudo_label_generator"):
    #         self.model.pseudo_label_generator.update_dynamic_thresh(current_ap)
    #
    #     return metrics

