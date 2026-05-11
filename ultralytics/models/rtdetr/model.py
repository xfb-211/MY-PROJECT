# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Interface for Baidu's RT-DETR, a Vision Transformer-based real-time object detector.

RT-DETR offers real-time performance and high accuracy, excelling in accelerated backends like CUDA with TensorRT.
It features an efficient hybrid encoder and IoU-aware query selection for enhanced detection accuracy.

References:
    https://arxiv.org/pdf/2304.08069.pdf
"""
import torch
from ultralytics.engine.model import Model
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils.torch_utils import TORCH_1_11, select_device

from .predict import RTDETRPredictor
from .train import RTDETRTrainer
from .val import RTDETRValidator

# ====================== DTAB-SSOD核心模块（单教师版本） ======================
from ultralytics.nn.modules.block import SingleTeacherPseudoLabelGenerator, MeanTeacherEMA, ConsistencyLoss, SemiRTDETRLoss, DropBlock


class RTDETR_Semi(Model):
    """
    扩展DTAB-SSOD半监督能力的RT-DETR模型
    完全兼容原生RT-DETR的所有功能，新增半监督训练、双教师融合、伪标签学习能力
    适配VisDrone数据集与rtdetr-l.yaml配置
    """

    def __init__(self, model: str = "semi-rtdetr-l.yaml", semi_config: dict = None) -> None:
        """
        初始化半监督RT-DETR模型
        :param model: 模型配置文件/预训练权重路径，支持.yaml/.pt
        :param semi_config: 半监督超参数配置，为空则使用默认值
        """
        assert TORCH_1_11, "RTDETR requires torch>=1.11"
        super().__init__(model=model, task="detect")

        # 半监督默认配置，可通过semi-rtdetr-l.yaml的semi字段覆盖
        self.default_semi_config = {
            "num_classes": 80,
            "conf_thresh": 0.7,
            "iou_thresh": 0.5,
            "ema_decay": 0.9996,
            "fusion_weights": (0.5, 0.5),
            "sup_loss_weight": 1.0,
            "unsup_loss_weight": 0.5,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "burn_up_steps": 12000,
            "warmup_steps": 12000,
            "initial_unsup_weight": 0.4,
            "weight_increment": 0.05,
            "max_unsup_weight": 1.0,
            "steps_per_epoch": 1475,
            "teacher1_path": "runs/coco_pretrain_teacher/teacher1/weights/best.pt",
            "teacher2_path": "runs/coco_pretrain_teacher/teacher2/weights/best.pt"
        }

        # 合并用户配置
        self.semi_config = semi_config if semi_config is not None else self.default_semi_config
        if hasattr(self.model, "yaml") and "semi" in self.model.yaml:
            self.semi_config.update(self.model.yaml["semi"])

        # 打印配置
        print("\n" + "=" * 60)
        print("✅ 半监督配置合并完成，最终参数如下：")
        print(f"  Teacher1 路径: {self.semi_config.get('teacher1_path', '未设置')}")
        print(f"  Teacher2 路径: {self.semi_config.get('teacher2_path', '未设置')}")
        print(f"  伪标签置信度阈值: {self.semi_config.get('conf_thresh', '未设置')}")
        print(f"  双教师融合权重: {self.semi_config.get('fusion_weights', self.semi_config.get('teacher_fusion_weight', '未设置'))}")
        print(f"  预热步数 burn_up_steps: {self.semi_config.get('burn_up_steps', '未设置')}")
        print(f"  初始无监督权重: {self.semi_config.get('initial_unsup_weight', '未设置')}")
        print(f"  DropBlock 概率: {self.semi_config.get('dropblock_prob', '未设置')}")
        print(f"  配置来源: {'yaml文件覆盖' if (hasattr(self.model, 'yaml') and 'semi' in self.model.yaml) else '默认值/传入参数'}")
        print("=" * 60 + "\n")

        # 初始化步数计数器
        self.register_buffer("global_step", torch.zeros(1, dtype=torch.long))

        # 初始化DTAB-SSOD核心模块
        self._device = select_device(self.semi_config["device"])
        self._init_semi_modules()

    def _init_semi_modules(self):
        """
        初始化DTAB-SSOD核心模块：双教师、伪标签生成器、损失函数、DropBlock
        """
        import torch
        from ultralytics.utils import LOGGER

        teacher1_path = self.semi_config.get("teacher1_path")
        teacher2_path = self.semi_config.get("teacher2_path")
        device = self._device

        if teacher1_path is None or teacher2_path is None:
            raise ValueError(
                "必须在semi-rtdetr-l.yaml的semi字段或semi_config参数中提供 "
                "teacher1_path和teacher2_path"
            )

        # 加载教师模型
        LOGGER.info(f"[DTAB] 加载教师模型1: {teacher1_path}")
        ckpt1 = torch.load(teacher1_path, map_location=device, weights_only=False)
        self._modules['_teacher1'] = ckpt1['model'].float().eval() \
            if isinstance(ckpt1, dict) and 'model' in ckpt1 else ckpt1.float().eval()

        LOGGER.info(f"[DTAB] 加载教师模型2: {teacher2_path}")
        ckpt2 = torch.load(teacher2_path, map_location=device, weights_only=False)
        self._modules['_teacher2'] = ckpt2['model'].float().eval() \
            if isinstance(ckpt2, dict) and 'model' in ckpt2 else ckpt2.float().eval()

        # 冻结教师参数
        for p in self._modules['_teacher1'].parameters():
            p.requires_grad_(False)
        for p in self._modules['_teacher2'].parameters():
            p.requires_grad_(False)
        LOGGER.info("双教师模型加载完毕并已冻结")

        # 初始化双教师融合模块（兼容两种键名）
        fusion_w = self.semi_config.get(
            "teacher_fusion_weight",
            self.semi_config.get("fusion_weights", [0.5, 0.5])
        )
        self._modules['_dual_teacher_fusion'] = DualTeacherFusion(
            num_classes=self.semi_config["num_classes"],
            ema_decay=self.semi_config.get("ema_decay", 0.9996),
            fusion_weights=fusion_w,
            device=device
        )

        # 初始化伪标签生成器
        self._modules['_pseudo_label_generator'] = PseudoLabelGenerator(
            conf_thresh=self.semi_config.get("conf_thresh", 0.7),
            device=device
        )

        # 初始化半监督损失函数
        self._modules['_semi_loss_fn'] = SemiRTDETRLoss(
            num_classes=self.semi_config["num_classes"],
            device=device
        )

        # 初始化DropBlock（提供默认值）
        self.dropblock = DropBlock(
            block_size=self.semi_config.get("dropblock_size", 7),
            drop_prob=self.semi_config.get("dropblock_prob", 0.2)
        )

        self._pseudo_count = 0
        LOGGER.info("所有DTAB-SSOD模块初始化完成")

    # 🔥 修复5：DTAB-SSOD官方权重退火（修复逻辑错误）
    def _update_unsup_weight(self):
        """DTAB论文标准无监督权重退火"""
        self.global_step += 1
        current_step = self.global_step.item()
        burn_up_steps = self.semi_config["burn_up_steps"]
        steps_per_epoch = self.semi_config["steps_per_epoch"]

        if current_step < burn_up_steps:
            new_weight = 0.0
        else:
            # 标准计算：每10个epoch递增一次
            epochs_after_warmup = (current_step - burn_up_steps) // steps_per_epoch
            num_steps = epochs_after_warmup // 10
            new_weight = self.semi_config["initial_unsup_weight"] + num_steps * self.semi_config["weight_increment"]
            new_weight = min(new_weight, self.semi_config["max_unsup_weight"])

        # 正确设置损失权重
        self._modules['_semi_loss_fn'].set_unsup_weight(new_weight)
        if current_step % 100 == 0:
            epoch = current_step // steps_per_epoch
            print(f"[权重退火] Step: {current_step}, Epoch: {epoch}, Unsup Weight: {new_weight:.3f}")
        return new_weight

    @property
    def task_map(self) -> dict:
        return {
            "detect": {
                "predictor": RTDETRPredictor,
                "validator": RTDETRValidator,
                "trainer": RTDETRTrainer,
                "model": RTDETRDetectionModel,
            }
        }

    def semi_forward_step(self, x_sup, x_unsup, gt_boxes, gt_labels):
        """
        半监督前向传播一步
        :param x_sup: 有标签图像 [bs, 3, H, W]
        :param x_unsup: 无标签图像 [bs, 3, H, W]
        :param gt_boxes: 有标签真实框 [bs, max_gts, 4] (xywh)
        :param gt_labels: 有标签真实类别 [bs, max_gts]
        :return: (total_loss, loss_dict)
        """
        self._update_unsup_weight()
        teacher1 = self._modules['_teacher1']
        teacher2 = self._modules['_teacher2']
        dual_fusion = self._modules['_dual_teacher_fusion']
        pseudo_gen = self._modules['_pseudo_label_generator']
        semi_loss_fn = self._modules['_semi_loss_fn']

        # ========== 1. 监督分支 ==========
        pred_sup = self.model(x_sup)
        # RT-DETR训练模式输出：(dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta) (5个元素)
        # RT-DETR评估模式输出：(y_tensor, (dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta)) (2个元素)
        # 统一提取(dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta)部分
        if isinstance(pred_sup, tuple) and len(pred_sup) == 2:
            pred_sup = pred_sup[1]

        sup_loss, sup_dict = semi_loss_fn(
            self.model, pred_sup, gt_boxes=gt_boxes, gt_labels=gt_labels,
            is_supervised=True
        )
        # 应用监督损失权重
        sup_loss = sup_loss * self.semi_config.get("sup_loss_weight", 1.0)

        # ========== 2. 伪标签生成 ==========
        with torch.no_grad():
            x_unsup_flip = torch.flip(x_unsup, dims=[3])
            # 教师模型推理 - 直接调用forward
            teacher1.eval()
            teacher2.eval()
            t1_out = teacher1(x_unsup)
            t2_out = teacher2(x_unsup_flip)

            # 统一处理教师模型输出格式
            if isinstance(t1_out, tuple) and len(t1_out) == 2:
                t1_out = t1_out[1]
            if isinstance(t2_out, tuple) and len(t2_out) == 2:
                t2_out = t2_out[1]

            # 统一为5元组格式
            t1_out_full = (t1_out[0], t1_out[1], None, None, None)
            t2_out_full = (t2_out[0], t2_out[1], None, None, None)
            pseudo_boxes, pseudo_labels, pseudo_mask = pseudo_gen(
                t1_out_full, t2_out_full, flipped=True
            )

        # ========== 3. 无监督分支 ==========
        unsup_loss = torch.tensor(0.0, device=x_sup.device)
        unsup_dict = {}
        if pseudo_mask.sum() > 0:
            pred_unsup = self.model(x_unsup)
            if isinstance(pred_unsup, tuple) and len(pred_unsup) == 2:
                pred_unsup = pred_unsup[1]
            unsup_loss, unsup_dict = semi_loss_fn(
                self.model, pred_unsup,
                pseudo_boxes=pseudo_boxes, pseudo_labels=pseudo_labels,
                pseudo_mask=pseudo_mask, is_supervised=False
            )

        # ========== 4. 特征扰动分支 ==========
        fp_loss = torch.tensor(0.0, device=x_sup.device)
        fp_dict = {}
        if pseudo_mask.sum() > 0:
            with torch.no_grad():
                features = []
                y = []
                x_fp = x_unsup
                for m in self.model.model[:-1]:
                    if m.f != -1:
                        x_fp = y[m.f] if isinstance(m.f, int) else \
                            [x_fp if j == -1 else y[j] for j in m.f]
                    x_fp = m(x_fp)
                    y.append(x_fp if m.i in self.model.save else None)
                head_inputs = [y[j] for j in self.model.model[-1].f]
                head_inputs = [self.dropblock(f) for f in head_inputs]

            head = self.model.model[-1]
            pred_fp = head(head_inputs)
            # 特征扰动分支输出是head直接输出的5元组，无需额外处理
            fp_loss, fp_dict = semi_loss_fn(
                self.model, pred_fp,
                pseudo_boxes=pseudo_boxes, pseudo_labels=pseudo_labels,
                pseudo_mask=pseudo_mask, is_supervised=False
            )

        # ========== 5. 总损失 ==========
        current_unsup_weight = semi_loss_fn.unsup_weight
        # 总损失 = 监督损失 + 无监督损失权重 * (无监督损失 + 特征扰动损失)/2
        total_loss = sup_loss + current_unsup_weight * (unsup_loss + fp_loss) / 2

        loss_dict = {**sup_dict, **unsup_dict, **{f"fp_{k}": v for k, v in fp_dict.items()}}
        loss_dict["total_loss"] = total_loss

        # 更新双教师模型
        dual_fusion.update_teachers(teacher1, teacher2, self.model)
        self._pseudo_count = pseudo_mask.sum().item()

        return total_loss, loss_dict


# ====================== 保留原生RTDETR类，保证向下兼容 ======================
class RTDETR(Model):
    """Interface for Baidu's RT-DETR model, a Vision Transformer-based real-time object detector.

    This model provides real-time performance with high accuracy. It supports efficient hybrid encoding, IoU-aware query
    selection, and adaptable inference speed.

    Attributes:
        model (str): Path to the pre-trained model.

    Methods:
        task_map: Return a task map for RT-DETR, associating tasks with corresponding Ultralytics classes.

    Examples:
        Initialize RT-DETR with a pre-trained model
        >>> from ultralytics import RTDETR
        >>> model = RTDETR("rtdetr-l.pt")
        >>> results = model("image.jpg")
    """

    def __init__(self, model: str = "rtdetr-l.pt") -> None:
        """Initialize the RT-DETR model with the given pre-trained model file.

        Args:
            model (str): Path to the pre-trained model. Supports .pt, .yaml, and .yml formats.
        """
        assert TORCH_1_11, "RTDETR requires torch>=1.11"
        super().__init__(model=model, task="detect")

    @property
    def task_map(self) -> dict:
        """Return a task map for RT-DETR, associating tasks with corresponding Ultralytics classes.

        Returns:
            (dict): A dictionary mapping task names to Ultralytics task classes for the RT-DETR model.
        """
        return {
            "detect": {
                "predictor": RTDETRPredictor,
                "validator": RTDETRValidator,
                "trainer": RTDETRTrainer,
                "model": RTDETRDetectionModel,
            }
        }