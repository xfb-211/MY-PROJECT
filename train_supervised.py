import warnings
warnings.filterwarnings('ignore')
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'ultralytics'))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics.models.rtdetr import RTDETRTrainer
from ultralytics.utils import LOGGER, colorstr, DEFAULT_CFG, RANK
from ultralytics.cfg import get_cfg


class SupervisedRTDETRTrainer(RTDETRTrainer):
    """
    纯监督RT-DETR训练器，用于验证基线性能。
    继承RTDETRTrainer，不包含任何半监督逻辑。
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        LOGGER.info("[SUPERVISED] Using pure supervised training (baseline)")

    def setup_model(self):
        """仅初始化学生模型，不加载教师模型"""
        super().setup_model()
        LOGGER.info("[SUPERVISED] Model initialized (no teachers)")


def main():
    """
    纯监督训练主函数
    用于验证DTAB-SSOD实验的基线性能
    """
    model_cfg = "rtdetr-l.pt"
    data_cfg = "./datasets/coco_semi.yaml"
    project = "runs/supervised_baseline"
    name = "RTDETR-Supervised"
    epochs = 120
    batch = 8
    imgsz = 640
    device = "0"
    workers = 0

    LOGGER.info(colorstr("green", "=" * 60))
    LOGGER.info(colorstr("green", "SUPERVISED BASELINE TRAINING"))
    LOGGER.info(colorstr("green", "=" * 60))
    LOGGER.info(colorstr("green", f"Model: {model_cfg}"))
    LOGGER.info(colorstr("green", f"Data: {data_cfg}"))
    LOGGER.info(colorstr("green", f"Epochs: {epochs}"))
    LOGGER.info(colorstr("green", f"Batch size: {batch}"))
    LOGGER.info(colorstr("green", f"Image size: {imgsz}"))
    LOGGER.info(colorstr("green", "=" * 60))

    overrides = {
        "model": model_cfg,
        "data": data_cfg,
        "epochs": epochs,
        "batch": batch,
        "imgsz": imgsz,
        "device": device,
        "workers": workers,
        "project": project,
        "name": name,
        "amp": False,
        "patience": 100,
        "save": True,
        "val": True,
        "exist_ok": True,
        "lr0": 1e-5,
        "weight_decay": 0.0001,
        "mosaic": 1.0,
        "warmup_bias_lr": 0.0,
        "warmup_epochs": 1.0,
    }

    cfg = get_cfg(DEFAULT_CFG)
    trainer = SupervisedRTDETRTrainer(cfg=cfg, overrides=overrides)
    trainer.train()

    LOGGER.info(colorstr("green", "=" * 60))
    LOGGER.info(colorstr("green", "SUPERVISED TRAINING COMPLETED"))
    LOGGER.info(colorstr("green", f"Results saved to: {trainer.save_dir}"))
    LOGGER.info(colorstr("green", "=" * 60))


if __name__ == "__main__":
    main()