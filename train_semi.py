import warnings
warnings.filterwarnings('ignore')
import os
import sys
import torch
import time
import warnings
import torch.nn.functional as F
# 关键：先添加路径，再导入我们修改的模块
sys.path.append(os.path.join(os.path.dirname(__file__), 'ultralytics'))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics.models.rtdetr import RTDETRTrainer
from ultralytics.utils import LOGGER, colorstr, DEFAULT_CFG, RANK
from ultralytics.cfg import get_cfg
from ultralytics.utils.torch_utils import autocast, unwrap_model
import numpy as np


class SemiRTDETRTrainer(RTDETRTrainer):
    """
    Semi-supervised RT-DETR trainer based on DTAB-SSOD.
    Override _do_train to inject unsupervised branch into the training loop.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        self.unsup_loader = None
        self.unsup_iter = None
        self.semi_model = None

    def setup_model(self):
        """初始化学生模型 + 单教师半监督模块"""
        import copy

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        super().setup_model()

        semi_cfg = {}
        model_path = str(self.args.model) if self.args.model else ""

        if model_path.endswith(('.yaml', '.yml')) and os.path.isfile(model_path):
            self._load_semi_cfg_from_yaml(model_path, semi_cfg)
        elif model_path.endswith('.pt'):
            default_yaml_path = "./ultralytics/cfg/models/rt-detr/rtdetr-l.yaml"
            if os.path.isfile(default_yaml_path):
                self._load_semi_cfg_from_yaml(default_yaml_path, semi_cfg)

        if not semi_cfg:
            LOGGER.warning("[SemiRTDETR] No semi config found")
            self.teacher1 = None
            self.teacher_ema = None
            return

        LOGGER.info(f"[SemiRTDETR] semi_cfg: {semi_cfg}")

        from ultralytics.nn.modules.block import (
            SingleTeacherPseudoLabelGenerator,
            MeanTeacherEMA,
            SemiRTDETRLoss
        )

        device = self.device
        semi_cfg.setdefault("num_classes", self.data["nc"])
        semi_cfg.setdefault("device", "cuda" if device.type == "cuda" else "cpu")

        teacher1_path = semi_cfg.get("teacher1_path", "rtdetr-l.pt")

        try:
            if not os.path.exists(teacher1_path):
                LOGGER.warning(f"[SemiRTDETR] Teacher path {teacher1_path} not found, cloning student")
                self.teacher1 = copy.deepcopy(self.model)
            else:
                LOGGER.info(f"[SemiRTDETR] Loading teacher from: {teacher1_path}")
                ckpt1 = torch.load(teacher1_path, map_location=device, weights_only=False)
                self.teacher1 = ckpt1['model'].float().eval() \
                    if isinstance(ckpt1, dict) and 'model' in ckpt1 else ckpt1.float().eval()
                self.teacher1 = self.teacher1.to(device)

            for p in self.teacher1.parameters():
                p.requires_grad_(False)

        except Exception as e:
            LOGGER.warning(f"[SemiRTDETR] Error loading teacher, cloning student: {e}")
            self.teacher1 = copy.deepcopy(self.model)
            for p in self.teacher1.parameters():
                p.requires_grad_(False)

        self.pseudo_label_gen = SingleTeacherPseudoLabelGenerator(
            conf_thresh=semi_cfg.get("conf_thresh", 0.25),
            min_pseudo_conf=semi_cfg.get("min_pseudo_conf", 0.25),
            use_gmm_filtering=semi_cfg.get("use_gmm_filtering", True),
            covariance_type=semi_cfg.get("covariance_type", "full"),
            cost_alpha=semi_cfg.get("cost_alpha", 1.0),
            cost_beta=semi_cfg.get("cost_beta", 6.0),
            device=device
        ).to(device)

        self.teacher_ema = MeanTeacherEMA(
            teacher_model=self.teacher1,
            student_model=self.model,
            decay=semi_cfg.get("ema_decay", 0.9996),
            warm_up=semi_cfg.get("ema_warm_up", 2000),
            device=device
        )
        self.teacher_ema.eval()
        self.teacher1.eval()

        self.semi_loss_fn = SemiRTDETRLoss(
            num_classes=semi_cfg["num_classes"],
            device=device,
            warm_up_step=semi_cfg.get("warm_up_step", 60000)
        ).to(device)

        self.burn_up_steps = semi_cfg.get("burn_up_steps", 20000)
        self.initial_unsup_weight = semi_cfg.get("initial_unsup_weight", 0.5)
        self.weight_increment = semi_cfg.get("weight_increment", 0.0)
        self.max_unsup_weight = semi_cfg.get("max_unsup_weight", 2.0)
        self.steps_per_epoch = semi_cfg.get("steps_per_epoch", 1479)
        self.global_step = 0

        LOGGER.info("[SemiRTDETR] Semi-supervised modules initialized")
        LOGGER.info(f"[SemiRTDETR] Teacher params: {sum(p.numel() for p in self.teacher1.parameters())}")

    def _load_semi_cfg_from_yaml(self, yaml_path, semi_cfg):
        """从yaml文件加载半监督配置"""
        try:
            import yaml
            with open(yaml_path, 'r', encoding='utf-8') as f:
                full_yaml = yaml.safe_load(f)
            if full_yaml and 'semi' in full_yaml:
                semi_cfg.update(full_yaml['semi'])
        except Exception as e:
            LOGGER.warning(f"[DTAB] Failed to load YAML config from {yaml_path}: {e}")

    def get_dataloader(self, dataset_path, batch_size=16, rank=-1, mode="train"):
        """Build dataloader, also create unlabeled dataloader for training."""
        loader = super().get_dataloader(dataset_path, batch_size=batch_size, rank=rank, mode=mode)

        if mode == "train":
            unlabeled_path = self.data.get("unlabeled", "")
            LOGGER.info(f"[DTAB] unlabeled path: {unlabeled_path}")

            if unlabeled_path and str(unlabeled_path).strip():
                try:
                    unsup_dataset = self.build_dataset(str(unlabeled_path), mode="train", batch=batch_size)
                    LOGGER.info(f"[DTAB] Unlabeled dataset size: {len(unsup_dataset)}")

                    if len(unsup_dataset) > 0:
                        from ultralytics.data.build import InfiniteDataLoader

                        self.unsup_loader = InfiniteDataLoader(
                            dataset=unsup_dataset,
                            batch_size=max(batch_size // 2, 1),
                            shuffle=True,
                            num_workers=self.args.workers,
                            pin_memory=True,
                            collate_fn=getattr(unsup_dataset, "collate_fn", None),
                        )
                        self.unsup_iter = iter(self.unsup_loader)
                        LOGGER.info(f"[DTAB] Unlabeled dataloader built, size: {len(unsup_dataset)}")
                    else:
                        LOGGER.warning("[DTAB] Unlabeled dataset is empty, skipping unsupervised branch")
                        self.unsup_loader = None
                        self.unsup_iter = None

                except Exception as e:
                    LOGGER.warning(f"[DTAB] Unlabeled dataloader build failed: {e}")
                    import traceback
                    traceback.print_exc()
                    self.unsup_loader = None
                    self.unsup_iter = None
            else:
                LOGGER.warning("[DTAB] No unlabeled path configured, fallback to supervised-only")
                self.unsup_loader = None
                self.unsup_iter = None

        return loader

    def _update_unsup_weight(self):
        """无监督权重退火 - 更激进增长策略"""
        self.global_step += 1

        if self.global_step < self.burn_up_steps:
            new_weight = 0.0
        else:
            progress = min(
                (self.global_step - self.burn_up_steps) / (3 * self.steps_per_epoch),
                1.0
            )
            # 使用更激进的增长曲线
            new_weight = self.initial_unsup_weight + (1 - np.cos(progress * np.pi)) * 0.5 * (
                        self.max_unsup_weight - self.initial_unsup_weight)
            new_weight = min(new_weight, self.max_unsup_weight)

        self.semi_loss_fn.set_unsup_weight(new_weight)
        return new_weight

    def _get_unsup_batch(self):
        """Get one unlabeled batch, auto-loop."""
        if self.unsup_iter is None:
            return None
        try:
            return next(self.unsup_iter)
        except StopIteration:
            self.unsup_iter = iter(self.unsup_loader)
            return next(self.unsup_iter)

    def _teacher_forward(self, model, x):
        """Run backbone+neck+head forward, return (dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta).

        Eval mode returns (y_tensor, x_tuple) where y_tensor=[bs,300,4+nc].
        Training mode returns (dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta) directly.
        """
        y = []
        x_t = x
        for m in model.model[:-1]:
            if m.f != -1:
                x_t = y[m.f] if isinstance(m.f, int) else \
                    [x_t if j == -1 else y[j] for j in m.f]
            x_t = m(x_t)
            y.append(x_t if m.i in model.save else None)
        head_inputs = [y[j] for j in model.model[-1].f]
        head = model.model[-1]
        out = head(head_inputs)

        # Eval mode: out = (y_tensor[bs,300,4+nc], (dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta))
        # Must check: len==2, out[0] is 2D+ tensor, out[1] is tuple with >=4 elements
        if (isinstance(out, (list, tuple)) and len(out) == 2 and
                isinstance(out[0], torch.Tensor) and out[0].dim() >= 2 and
                isinstance(out[1], (list, tuple)) and len(out[1]) >= 4):
            return out[1]

        # Training mode: out is already (dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta)
        return out

    def _compute_unsup_loss(self):
        if not hasattr(self, 'semi_loss_fn') or self.semi_loss_fn is None:
            return torch.tensor(0.0, device=self.device)

        current_unsup_weight = self._update_unsup_weight()

        if current_unsup_weight <= 0 or self.unsup_iter is None:
            return torch.tensor(0.0, device=self.device)

        try:
            unsup_batch = self._get_unsup_batch()
        except StopIteration:
            return torch.tensor(0.0, device=self.device)

        if unsup_batch is None:
            return torch.tensor(0.0, device=self.device)

        unsup_batch = self.preprocess_batch(unsup_batch)
        x_unsup = unsup_batch["img"]
        x_weak = torch.flip(x_unsup, dims=[3])

        with torch.no_grad():
            # 使用 EMA 教师模型
            ema_teacher = self.teacher_ema.teacher if hasattr(self, 'teacher_ema') and self.teacher_ema else self.teacher1
            if ema_teacher.training:
                ema_teacher.eval()
            teacher_out = self._teacher_forward(ema_teacher, x_weak)
            pseudo_boxes, pseudo_labels, pseudo_mask = self.pseudo_label_gen(teacher_out)

            n_valid = pseudo_mask.sum().item() if pseudo_mask is not None else 0

            if n_valid == 0:
                return torch.tensor(0.0, device=self.device)

            pseudo_quality = self._compute_pseudo_quality(pseudo_mask, teacher_out)
            # 调整质量因子计算，降低阈值敏感性
            quality_factor = min(pseudo_quality / 0.5, 1.0)

        pseudo_targets = self.semi_loss_fn._build_pseudo_targets(
            pseudo_boxes, pseudo_labels, pseudo_mask
        )

        if pseudo_targets is None:
            return torch.tensor(0.0, device=self.device)

        pseudo_boxes_orig = pseudo_boxes.clone()
        if pseudo_boxes_orig.shape[-1] == 4:
            x1 = pseudo_boxes_orig[..., 0].clone()
            x2 = pseudo_boxes_orig[..., 2].clone()
            pseudo_boxes_orig[..., 0] = 1.0 - x2
            pseudo_boxes_orig[..., 2] = 1.0 - x1

        pred_unsup = self._teacher_forward(self.model, x_unsup)
        unsup_loss, _ = self.semi_loss_fn(
            self.model, pred_unsup,
            pseudo_boxes=pseudo_boxes_orig,
            pseudo_labels=pseudo_labels,
            pseudo_mask=pseudo_mask,
            is_supervised=False
        )

        consistency_loss = self._compute_query_consistency_loss(pred_unsup, teacher_out, pseudo_mask)

        # 确保unsup_loss是tensor且非负
        if not isinstance(unsup_loss, torch.Tensor):
            unsup_loss = torch.tensor(unsup_loss, device=self.device)
        unsup_loss = unsup_loss.clamp(min=0.0, max=10.0)

        if not isinstance(consistency_loss, torch.Tensor):
            consistency_loss = torch.tensor(consistency_loss, device=self.device)
        consistency_loss = consistency_loss.clamp(min=0.0, max=5.0)

        total_unsup_loss = unsup_loss + 0.2 * consistency_loss
        total_unsup_loss = total_unsup_loss.clamp(min=0.0, max=15.0)

        if hasattr(self, 'teacher_ema') and self.teacher_ema is not None:
            self.teacher_ema.update()

        if self.global_step % 400 == 0:
            debug_msg = (f"[Semi] step={self.global_step} "
                         f"unsup_weight={current_unsup_weight:.4f} "
                         f"quality_factor={quality_factor:.4f} "
                         f"n_valid_pseudo={n_valid} "
                         f"pseudo_quality={pseudo_quality:.4f} "
                         f"unsup_loss={unsup_loss.item():.4f} "
                         f"consistency_loss={consistency_loss.item():.4f}")
            sys.stderr.write(f"\r{debug_msg}\n")
            sys.stderr.flush()

        # 转换为标量并限制范围
        total_unsup_loss_scalar = total_unsup_loss.item() * current_unsup_weight * quality_factor
        total_unsup_loss_scalar = max(0.0, min(total_unsup_loss_scalar, 15.0))

        return torch.tensor(total_unsup_loss_scalar, device=self.device, requires_grad=True)

    def _compute_query_consistency_loss(self, student_out, teacher_out, pseudo_mask):
        """计算跨视图查询一致性损失"""
        try:
            dec_bboxes_s, dec_scores_s = student_out[0], student_out[1]
            dec_bboxes_t, dec_scores_t = teacher_out[0], teacher_out[1]

            student_queries = dec_bboxes_s[-1]
            teacher_queries = dec_bboxes_t[-1]

            batch_size = student_queries.shape[0]
            num_queries = student_queries.shape[1]

            valid_mask = pseudo_mask.flatten()
            if valid_mask.sum() == 0:
                valid_mask = torch.ones(num_queries, dtype=torch.bool, device=student_queries.device)

            loss = 0.0
            count = 0

            for b in range(batch_size):
                s_feat = student_queries[b][valid_mask[b % len(valid_mask)]]
                t_feat = teacher_queries[b][valid_mask[b % len(valid_mask)]]

                if len(s_feat) > 0 and len(t_feat) > 0:
                    min_len = min(len(s_feat), len(t_feat))
                    s_feat = s_feat[:min_len]
                    t_feat = t_feat[:min_len]

                    s_norm = F.normalize(s_feat, p=2, dim=-1)
                    t_norm = F.normalize(t_feat, p=2, dim=-1)

                    loss += (1 - (s_norm * t_norm).sum(dim=-1)).mean()
                    count += 1

            return loss / max(count, 1) if count > 0 else torch.tensor(0.0, device=self.device)

        except Exception:
            return torch.tensor(0.0, device=self.device)

    def _compute_pseudo_quality(self, pseudo_mask, teacher_out):
        """
        计算伪标签质量分数

        返回值：0-1之间，越高表示质量越好
        """
        if pseudo_mask is None or teacher_out is None:
            return 0.0

        dec_scores = teacher_out[1]
        scores = dec_scores[-1].sigmoid().max(dim=-1)[0]  # [bs, num_queries]

        # 只考虑有效伪标签的分数
        valid_scores = scores[pseudo_mask]

        if len(valid_scores) == 0:
            return 0.0

        # 质量分数 = 平均置信度
        quality = valid_scores.mean().item()

        return quality

    def _get_model_features(self, model, x):
        """获取模型中间层特征"""
        features = []
        unwrapped_model = unwrap_model(model)

        y = []
        x_t = x
        for m in unwrapped_model.model[:-1]:
            if m.f != -1:
                x_t = y[m.f] if isinstance(m.f, int) else \
                    [x_t if j == -1 else y[j] for j in m.f]
            x_t = m(x_t)
            y.append(x_t if m.i in unwrapped_model.save else None)

            if m.i in [3, 5, 7, 9]:
                features.append(x_t)

        return features if features else None

    def _do_train(self):
        """Override _do_train to inject unsupervised loss into the training loop."""
        from ultralytics.utils import TQDM
        import math

        if self.world_size > 1:
            self._setup_ddp()
        self._setup_train()

        nb = len(self.train_loader)
        nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1
        last_opt_step = -1
        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        self.run_callbacks("on_train_start")
        LOGGER.info(
            f"Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n"
            f"Using {self.train_loader.num_workers * (self.world_size or 1)} dataloader workers\n"
            f"Logging results to {colorstr('bold', self.save_dir)}\n"
            f"Starting training for " + (f"{self.args.time} hours..." if self.args.time else f"{self.epochs} epochs...")
        )
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
        epoch = self.start_epoch
        self.optimizer.zero_grad()
        while True:
            self.epoch = epoch
            self.run_callbacks("on_train_epoch_start")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.scheduler.step()

            self._model_train()
            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
            pbar = enumerate(self.train_loader)
            if epoch == (self.epochs - self.args.close_mosaic):
                self._close_dataloader_mosaic()
                self.train_loader.reset()

            if RANK in {-1, 0}:
                LOGGER.info(self.progress_string())
                pbar = TQDM(enumerate(self.train_loader), total=nb)
            self.tloss = None
            for i, batch in pbar:
                self.run_callbacks("on_train_batch_start")
                # Warmup
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]
                    self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()))
                    for j, x in enumerate(self.optimizer.param_groups):
                        x["lr"] = np.interp(
                            ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x["initial_lr"] * self.lf(epoch)]
                        )
                        if "momentum" in x:
                            x["momentum"] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])

                # Forward
                with autocast(self.amp):
                    batch = self.preprocess_batch(batch)

                    # 计算监督损失 (有标签数据)
                    if self.args.compile:
                        preds = self.model(batch["img"])
                        loss, self.loss_items = unwrap_model(self.model).loss(batch, preds)
                    else:
                        loss, self.loss_items = self.model(batch)

                    self.loss = loss.sum()

                    # 计算无监督损失 (无标签数据)
                    unsup_loss = torch.tensor(0.0, device=self.device)
                    if hasattr(self, 'semi_loss_fn') and self.semi_loss_fn is not None:
                        unsup_loss = self._compute_unsup_loss()

                    # 关键：监督损失 + 无监督损失 = 总损失
                    total_loss = self.loss + unsup_loss

                    if RANK != -1:
                        total_loss *= self.world_size
                    self.tloss = self.loss_items if self.tloss is None else (self.tloss * i + self.loss_items) / (i + 1)

                # Backward - 一次反向传播（包含监督 + 无监督）
                self.scaler.scale(total_loss).backward()
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni

                    # Timed stopping
                    if self.args.time:
                        self.stop = (time.time() - self.train_time_start) > (self.args.time * 3600)
                        if RANK != -1:
                            broadcast_list = [self.stop if RANK == 0 else None]
                            import torch.distributed as dist
                            dist.broadcast_object_list(broadcast_list, 0)
                            self.stop = broadcast_list[0]
                        if self.stop:
                            break

                # Log
                if RANK in {-1, 0}:
                    loss_length = self.tloss.shape[0] if len(self.tloss.shape) else 1
                    pbar.set_description(
                        ("%11s" * 2 + "%11.4g" * (2 + loss_length))
                        % (
                            f"{epoch + 1}/{self.epochs}",
                            f"{self._get_memory():.3g}G",
                            *(self.tloss if loss_length > 1 else torch.unsqueeze(self.tloss, 0)),
                            batch["cls"].shape[0],
                            batch["img"].shape[-1],
                        )
                    )
                    self.run_callbacks("on_batch_end")
                    if self.args.plots and ni in self.plot_idx:
                        self.plot_training_samples(batch, ni)

                self.run_callbacks("on_train_batch_end")

            self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}

            self.run_callbacks("on_train_epoch_end")
            if RANK in {-1, 0}:
                self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])

            # ====== DTAB-DEBUG: validate model immediately after loading ======
            if epoch == 0 and not hasattr(self, '_loading_validated'):
                self._loading_validated = True
                self.model.eval()
                _metrics, _fitness = self.validate()
                LOGGER.info(
                    f"[DTAB-DEBUG] Post-loading validation (no training): mAP50={_metrics['metrics/mAP50(B)']:.4f}")
                self._model_train()
            # ====== END DTAB-DEBUG ======

            # Validation
            final_epoch = epoch + 1 >= self.epochs
            if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                self._clear_memory(threshold=0.5)
                self.metrics, self.fitness = self.validate()

            # NaN recovery
            if self._handle_nan_recovery(epoch):
                continue

            self.nan_recovery_attempts = 0
            if RANK in {-1, 0}:
                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch
                if self.args.time:
                    self.stop |= (time.time() - self.train_time_start) > (self.args.time * 3600)

                # Save model
                if self.args.save or final_epoch:
                    self.save_model()
                    self.run_callbacks("on_model_save")

            # Scheduler
            t = time.time()
            self.epoch_time = t - self.epoch_time_start
            self.epoch_time_start = t
            if self.args.time:
                mean_epoch_time = (t - self.train_time_start) / (epoch - self.start_epoch + 1)
                self.epochs = self.args.epochs = math.ceil(self.args.time * 3600 / mean_epoch_time)
                self._setup_scheduler()
                self.scheduler.last_epoch = self.epoch
                self.stop |= epoch >= self.epochs
            self.run_callbacks("on_fit_epoch_end")
            self._clear_memory(0.5)

            # Early Stopping
            if RANK != -1:
                broadcast_list = [self.stop if RANK == 0 else None]
                import torch.distributed as dist
                dist.broadcast_object_list(broadcast_list, 0)
                self.stop = broadcast_list[0]
            if self.stop:
                break
            epoch += 1

        seconds = time.time() - self.train_time_start
        LOGGER.info(f"\n{epoch - self.start_epoch + 1} epochs completed in {seconds / 3600:.3f} hours.")
        self.final_eval()
        if RANK in {-1, 0}:
            if self.args.plots:
                self.plot_metrics()
            self.run_callbacks("on_train_end")
        self._clear_memory()
        from ultralytics.utils.torch_utils import unset_deterministic
        unset_deterministic()
        self.run_callbacks("teardown")


def main():
    """
    阶段2：半监督训练学生模型
    """
    model_cfg = "rtdetr-l.pt"
    data_cfg = "./datasets/coco_semi.yaml"
    project = "runs/student_training"
    name = "RTDETR-Student-Semi"
    epochs = 100
    batch = 4
    imgsz = 640
    device = "0"
    workers = 0

    # ====== 关键：使用阶段1训练的教师 ======
    teacher_path = "runs/teacher_training/RTDETR-Teacher/weights/best.pt"

    LOGGER.info(colorstr("green", "=" * 60))
    LOGGER.info(colorstr("green", "PHASE 2: SEMI-SUPERVISED STUDENT TRAINING"))
    LOGGER.info(colorstr("green", f"Teacher: {teacher_path}"))
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
        "patience": 50,
        "save": True,
        "val": True,
        "exist_ok": True,
        "lr0": 8e-6,
        "lrf": 0.01,
        "weight_decay": 0.0005,
        "mosaic": 1.0,
        "warmup_bias_lr": 0.0,
        "warmup_epochs": 1.0,
        "close_mosaic": 5,
    }

    cfg = get_cfg(DEFAULT_CFG)
    trainer = SemiRTDETRTrainer(cfg=cfg, overrides=overrides)
    trainer.train()


if __name__ == "__main__":
    main()
