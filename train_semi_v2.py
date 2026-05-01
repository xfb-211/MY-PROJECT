import warnings

warnings.filterwarnings('ignore')
import os
import sys
import torch
import time
import warnings

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
    支持通过 enable_semi 参数禁用半监督，验证纯监督基线。
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None, enable_semi=True):
        """
        Args:
            enable_semi: 是否启用半监督训练，设为False可验证纯监督基线
        """
        super().__init__(cfg, overrides, _callbacks)
        self.unsup_loader = None
        self.unsup_iter = None
        self.semi_model = None
        self.enable_semi = enable_semi

    def setup_model(self):
        """Initialize student model + semi-supervised modules."""
        super().setup_model()

        if not self.enable_semi:
            LOGGER.info("[DTAB] enable_semi=False, using pure supervised training (baseline)")
            return

        semi_cfg = {}
        model_path = str(self.args.model) if self.args.model else ""

        if model_path.endswith(('.yaml', '.yml')) and os.path.isfile(model_path):
            self._load_semi_cfg_from_yaml(model_path, semi_cfg)
        elif model_path.endswith('.pt'):
            default_yaml_path = "./ultralytics/cfg/models/rt-detr/rtdetr-l.yaml"
            if os.path.isfile(default_yaml_path):
                self._load_semi_cfg_from_yaml(default_yaml_path, semi_cfg)
            elif hasattr(self.model, 'model') and hasattr(self.model.model, 'yaml_file'):
                yaml_path = self.model.model.yaml_file
                if os.path.isfile(yaml_path):
                    self._load_semi_cfg_from_yaml(yaml_path, semi_cfg)

        if not semi_cfg:
            LOGGER.warning("[DTAB] No semi config found, fallback to supervised-only training")
            return

        LOGGER.info(f"[DTAB] semi_cfg: {semi_cfg}")

        from ultralytics.nn.modules.block import (
            EMAUpdater, DualTeacherFusion, PseudoLabelGenerator,
            DropBlock, SemiRTDETRLoss
        )

        device = self.device
        semi_cfg.setdefault("num_classes", self.data["nc"])
        semi_cfg.setdefault("device", "cuda" if device.type == "cuda" else "cpu")

        teacher1_path = semi_cfg.get("teacher1_path")
        teacher2_path = semi_cfg.get("teacher2_path")

        if teacher1_path and teacher2_path:
            LOGGER.info(f"[DTAB] Loading teacher1: {teacher1_path}")
            ckpt1 = torch.load(teacher1_path, map_location=device, weights_only=False)
            teacher1 = ckpt1['model'].float().eval() \
                if isinstance(ckpt1, dict) and 'model' in ckpt1 else ckpt1.float().eval()
            for p in teacher1.parameters():
                p.requires_grad_(False)

            LOGGER.info(f"[DTAB] Loading teacher2: {teacher2_path}")
            ckpt2 = torch.load(teacher2_path, map_location=device, weights_only=False)
            teacher2 = ckpt2['model'].float().eval() \
                if isinstance(ckpt2, dict) and 'model' in ckpt2 else ckpt2.float().eval()
            for p in teacher2.parameters():
                p.requires_grad_(False)

            self.teacher1 = teacher1.to(device)
            self.teacher2 = teacher2.to(device)

            fusion_w = semi_cfg.get("teacher_fusion_weight", semi_cfg.get("fusion_weights", [0.5, 0.5]))
            self.dual_teacher_fusion = DualTeacherFusion(
                num_classes=semi_cfg["num_classes"],
                ema_decay=semi_cfg.get("ema_decay", 0.9996),
                fusion_weights=fusion_w,
                device=device
            ).to(device)

            self.pseudo_label_gen = PseudoLabelGenerator(
                conf_thresh=semi_cfg.get("conf_thresh", 0.7),
                cluster_iou=semi_cfg.get("cluster_iou", 0.6),
                match_iou=semi_cfg.get("match_iou", 0.6),
                min_pseudo_conf=semi_cfg.get("min_pseudo_conf", 0.8),
                use_pseudo_filtering=semi_cfg.get("use_pseudo_filtering", True),
                use_gmm_filtering=semi_cfg.get("use_gmm_filtering", False),
                covariance_type=semi_cfg.get("covariance_type", "full"),
                device=device
            ).to(device)

            self.semi_loss_fn = SemiRTDETRLoss(
                num_classes=semi_cfg["num_classes"],
                device=device
            ).to(device)

            self.dropblock = DropBlock(
                block_size=semi_cfg.get("dropblock_size", 7),
                drop_prob=semi_cfg.get("dropblock_prob", 0.2)
            )

            self.burn_up_steps = semi_cfg.get("burn_up_steps", 5000)
            self.initial_unsup_weight = semi_cfg.get("initial_unsup_weight", 0.05)
            self.weight_increment = semi_cfg.get("weight_increment", 0.0)
            self.max_unsup_weight = semi_cfg.get("max_unsup_weight", 0.15)
            self.steps_per_epoch = semi_cfg.get("steps_per_epoch", 1479)
            self.global_step = 0

            LOGGER.info("[DTAB] Semi-supervised modules initialized successfully")
            LOGGER.info(f"[DTAB] Teacher1 params: {sum(p.numel() for p in self.teacher1.parameters())}")
            LOGGER.info(f"[DTAB] Teacher2 params: {sum(p.numel() for p in self.teacher2.parameters())}")
        else:
            LOGGER.warning("[DTAB] teacher1_path or teacher2_path not found in semi config")

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

        if mode == "train" and self.enable_semi:
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
        """DTAB-SSOD改进版无监督权重退火"""
        self.global_step += 1
        if self.global_step < self.burn_up_steps:
            new_weight = 0.0
        else:
            progress = min(
                (self.global_step - self.burn_up_steps) / (2 * self.steps_per_epoch),
                1.0
            )
            new_weight = self.initial_unsup_weight + progress * (self.max_unsup_weight - self.initial_unsup_weight)
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
        """Run backbone+neck+head forward, return (dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta)."""
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

        if (isinstance(out, (list, tuple)) and len(out) == 2 and
                isinstance(out[0], torch.Tensor) and out[0].dim() >= 2 and
                isinstance(out[1], (list, tuple)) and len(out[1]) >= 4):
            return out[1]

        return out

    def _compute_unsup_loss(self):
        """Compute unsupervised loss using teacher pseudo-labels."""
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
        x_unsup_weak = torch.flip(x_unsup, dims=[3])

        with torch.no_grad():
            t1_out = self._teacher_forward(self.teacher1, x_unsup)
            t2_out = self._teacher_forward(self.teacher2, x_unsup_weak)

            pseudo_boxes, pseudo_labels, pseudo_mask = self.pseudo_label_gen(
                t1_out, t2_out, flipped=True
            )

        _debug_log = None
        if self.global_step % 200 == 0:
            n_pseudo = pseudo_mask.sum().item() if pseudo_mask is not None else 0
            _debug_log = (f"[DTAB-DEBUG] step={self.global_step} unsup_weight={current_unsup_weight:.4f} "
                          f"pseudo_labels={pseudo_labels.shape if pseudo_labels is not None else 'None'} "
                          f"n_valid_pseudo={n_pseudo}")

        pseudo_targets = self.semi_loss_fn._build_pseudo_targets(
            pseudo_boxes, pseudo_labels, pseudo_mask
        )

        if pseudo_targets is None:
            if _debug_log is not None:
                _debug_log += " | pseudo_targets=None, skipping"
                sys.stderr.write(f"\r{_debug_log}\n")
                sys.stderr.flush()
            return torch.tensor(0.0, device=self.device)

        pred_unsup = self._teacher_forward(self.model, x_unsup)
        unsup_loss, _ = self.semi_loss_fn(
            self.model, pred_unsup,
            pseudo_boxes=pseudo_boxes, pseudo_labels=pseudo_labels,
            pseudo_mask=pseudo_mask, is_supervised=False
        )

        if _debug_log is not None:
            _debug_log += (f" | unsup_loss={unsup_loss.item():.4f} "
                          f"weighted={current_unsup_weight * unsup_loss.item():.4f}")
            sys.stderr.write(f"\r{_debug_log}\n")
            sys.stderr.flush()

        with torch.no_grad():
            y = []
            x_fp = x_unsup_weak
            unwrapped_model = unwrap_model(self.model)
            for m in unwrapped_model.model[:-1]:
                if m.f != -1:
                    x_fp = y[m.f] if isinstance(m.f, int) else \
                        [x_fp if j == -1 else y[j] for j in m.f]
                x_fp = m(x_fp)
                y.append(x_fp if m.i in unwrapped_model.save else None)

            head_inputs = [y[j] for j in unwrapped_model.model[-1].f]
            head_inputs = [self.dropblock(f) for f in head_inputs]

        head = unwrapped_model.model[-1]
        pred_fp = head(head_inputs, batch=pseudo_targets)
        fp_loss, _ = self.semi_loss_fn(
            self.model, pred_fp,
            pseudo_boxes=pseudo_boxes, pseudo_labels=pseudo_labels,
            pseudo_mask=pseudo_mask, is_supervised=False
        )

        unsup_loss = (unsup_loss + fp_loss) / 2

        if hasattr(self, 'dual_teacher_fusion'):
            self.dual_teacher_fusion.update_teachers(
                self.teacher1, self.teacher2, self.model
            )

        return current_unsup_weight * unsup_loss

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
            f"Starting training for " + (f"{self.args.time} hours..." if self.args.time else f"{self.args.epochs} epochs...")
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

                with autocast(self.amp):
                    if self.enable_semi and hasattr(self, 'semi_loss_fn') and self.semi_loss_fn is not None:
                        unsup_loss = self._compute_unsup_loss()
                        if unsup_loss.requires_grad:
                            self.scaler.scale(unsup_loss).backward()
                            torch.cuda.empty_cache()

                    batch = self.preprocess_batch(batch)
                    if self.args.compile:
                        preds = self.model(batch["img"])
                        loss, self.loss_items = unwrap_model(self.model).loss(batch, preds)
                    else:
                        loss, self.loss_items = self.model(batch)

                    self.loss = loss.sum()

                    if RANK != -1:
                        self.loss *= self.world_size
                    self.tloss = self.loss_items if self.tloss is None else (self.tloss * i + self.loss_items) / (i + 1)

                self.scaler.scale(self.loss).backward()
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni

                    if self.args.time:
                        self.stop = (time.time() - self.train_time_start) > (self.args.time * 3600)
                        if RANK != -1:
                            broadcast_list = [self.stop if RANK == 0 else None]
                            import torch.distributed as dist
                            dist.broadcast_object_list(broadcast_list, 0)
                            self.stop = broadcast_list[0]
                        if self.stop:
                            break

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

            if epoch == 0 and not hasattr(self, '_loading_validated'):
                self._loading_validated = True
                self.model.eval()
                _metrics, _fitness = self.validate()
                LOGGER.info(
                    f"[DTAB-DEBUG] Post-loading validation (no training): mAP50={_metrics['metrics/mAP50(B)']:.4f}"
                )
                self._model_train()

            final_epoch = epoch + 1 >= self.epochs
            if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                self._clear_memory(threshold=0.5)
                self.metrics, self.fitness = self.validate()

            if self._handle_nan_recovery(epoch):
                continue

            self.nan_recovery_attempts = 0
            if RANK in {-1, 0}:
                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch
                if self.args.time:
                    self.stop |= (time.time() - self.train_time_start) > (self.args.time * 3600)

                if self.args.save or final_epoch:
                    self.save_model()
                    self.run_callbacks("on_model_save")

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
    model_cfg = "./ultralytics/cfg/models/rt-detr/rtdetr-l-semi-conservative.yaml"
    data_cfg = "./datasets/coco_semi.yaml"
    project = "runs/semi_supervised"
    name = "DTAB-SSOD-Conservative"
    epochs = 120
    batch = 8
    imgsz = 640
    device = "0"
    workers = 0
    enable_semi = True

    LOGGER.info(colorstr("green", "=" * 60))
    LOGGER.info(colorstr("green", "DTAB-SSOD 保守配置半监督训练"))
    LOGGER.info(colorstr("green", "=" * 60))
    LOGGER.info(colorstr("green", f"Model: {model_cfg}"))
    LOGGER.info(colorstr("green", f"Data: {data_cfg}"))
    LOGGER.info(colorstr("green", f"Epochs: {epochs}"))
    LOGGER.info(colorstr("green", f"Batch size: {batch}"))
    LOGGER.info(colorstr("green", f"Image size: {imgsz}"))
    LOGGER.info(colorstr("green", f"Semi-supervised enabled: {enable_semi}"))
    LOGGER.info(colorstr("green", "=" * 60))
    LOGGER.info(colorstr("yellow", f"对比基准：纯监督基线 mAP50=0.678"))
    LOGGER.info(colorstr("yellow", f"目标：半监督性能 ≥ 0.68（提升1%以上）"))

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
        "lr0": 1e-05,
        "weight_decay": 0.0001,
        "mosaic": 1.0,
        "warmup_bias_lr": 0.0,
        "warmup_epochs": 1.0,
    }

    cfg = get_cfg(DEFAULT_CFG)
    trainer = SemiRTDETRTrainer(cfg=cfg, overrides=overrides, enable_semi=enable_semi)
    trainer.train()

    LOGGER.info(colorstr("green", "=" * 60))
    LOGGER.info(colorstr("green", "半监督训练完成"))
    LOGGER.info(colorstr("green", f"结果：{trainer.save_dir}"))
    LOGGER.info(colorstr("yellow", "请对比：半监督 vs 纯监督基线"))


if __name__ == "__main__":
    main()
