# Semi-DETR 半监督目标检测框架技术文档

## 目录

1. [项目概述](#1-项目概述)
2. [环境配置指南](#2-环境配置指南)
3. [核心技术原理](#3-核心技术原理)
4. [教师-学生模型机制](#4-教师-学生模型机制)
5. [代码模块详解](#5-代码模块详解)
6. [训练与评估流程](#6-训练与评估流程)
7. [常见问题与解决方案](#7-常见问题与解决方案)

---

## 1. 项目概述

### 1.1 项目简介

Semi-DETR是一个基于Transformer的半监督目标检测框架，结合了DINO-DETR检测器和Mean Teacher半监督学习范式。该框架通过教师-学生协同训练机制，利用大量无标注数据提升检测性能。

### 1.2 主要特性

- **基于Transformer架构**: 使用DINO-DETR作为基础检测器
- **Mean Teacher框架**: 通过EMA更新教师模型
- **分阶段混合匹配**: One-to-Many + One-to-One两阶段匹配策略
- **智能伪标签过滤**: GMM自适应阈值 + 固定置信度阈值双重过滤
- **跨视图一致性**: 弱增强-强增强视图间的一致性正则化

### 1.3 性能指标

| 数据集 | 设置 | mAP |
|--------|------|-----|
| COCO | 1% 标注 | 30.50% |
| COCO | 5% 标注 | 40.10% |
| COCO | 10% 标注 | 43.30% |
| COCO | 全量标注 | 50.50% |
| VOC | VOC12 | 86.1% AP50 |

---

## 2. 环境配置指南

### 2.1 系统要求

- **操作系统**: Ubuntu 18.04+ / Windows 11 (WSL2)
- **GPU**: NVIDIA GPU with CUDA support
- **CUDA**: 11.1+

### 2.2 依赖版本

| 组件 | 版本 |
|------|------|
| Python | 3.8 |
| PyTorch | 1.9.0+cu111 |
| mmcv-full | 1.3.16 |
| mmdetection | 2.16.0 |

### 2.3 安装步骤

#### 方案A: WSL2环境（推荐Windows用户）

```bash
# 1. 安装WSL2
wsl --install -d Ubuntu-22.04

# 2. 安装CUDA工具包
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install cuda-11-8 -y

# 3. 设置环境变量
echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export CUDA_HOME=/usr/local/cuda-11.8' >> ~/.bashrc
source ~/.bashrc

# 4. 创建conda环境
conda create -n semidetr python=3.8 -y
conda activate semidetr

# 5. 安装PyTorch
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

# 6. 安装mmcv-full
pip install mmcv-full==1.3.16 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html

# 7. 安装项目依赖
cd ~/projects/Semi-DETR-main
pip install -e . -q
pip install scikit-learn scipy wandb prettytable terminaltables timm==0.4.12 tensorboard future

# 8. 编译CUDA算子
cd detr_od/models/utils/ops
python setup.py build install
```

#### 方案B: Linux原生环境

```bash
# 1. 创建conda环境
conda create -n semidetr python=3.8 -y
conda activate semidetr

# 2. 安装PyTorch和mmcv
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.3.16 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html

# 3. 安装项目
cd Semi-DETR-main
cd thirdparty/mmdetection && pip install -e . && cd ../..
pip install -e .
```

### 2.4 数据准备

```bash
# 1. 创建数据目录
mkdir -p ~/data/coco
ln -s /path/to/your/COCO2017 ~/data/coco/dataset

# 2. 创建半监督数据分割
python scripts/generate_splits.py
```

---

## 3. 核心技术原理

### 3.1 分阶段混合匹配（Two-Stage Hybrid Matching）

#### 3.1.1 技术动机

传统DETR使用One-to-One匹配，虽然避免了NMS，但可能导致Recall不足。Semi-DETR采用分阶段策略：

- **阶段1 (One-to-Many)**: 提高Recall，让更多预测框参与训练
- **阶段2 (One-to-One)**: 提高Precision，精炼匹配结果

#### 3.1.2 代码实现

**文件位置**: `detr_od/core/bbox/assigners/o2m_assigner.py`

```python
@BBOX_ASSIGNERS.register_module()
class O2MAssigner(BaseAssigner):
    def __init__(self, candidate_topk=13, debug=False):
        self.candidate_topk = candidate_topk
    
    def assign(self, bbox_pred, cls_pred, gt_bboxes, gt_labels, img_meta, ...):
        # 1. 计算alignment metric
        overlaps = bbox_overlaps(pred_bboxes, gt_bboxes)
        bbox_scores = scores[:, gt_labels]
        alignment_metrics = bbox_scores ** alpha * overlaps ** beta
        
        # 2. One-to-Many: 选择top-k候选正样本
        _, candidate_idxs = alignment_metrics.topk(
            self.candidate_topk, dim=0, largest=True
        )
        
        # 3. 动态正样本选择
        if teacher_assign and multiple_pos:
            # 动态估计正样本数量
            dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)
            for gt_idx in range(num_gts):
                _, pos_idx = torch.topk(candidate_metrics[:, gt_idx], 
                                        k=dynamic_ks[gt_idx].item(), largest=True)
                is_pos[:, gt_idx][pos_idx] = 1
```

#### 3.1.3 配置

```python
# configs/dino_detr/dino_detr_ssod_r50_coco_120k.py
train_cfg=dict(
    assigner1=dict(type='O2MAssigner'),      # One-to-Many阶段
    assigner2=dict(type='HungarianAssigner'), # One-to-One阶段
    warm_up_step=60000
)
```

### 3.2 跨视图查询一致性（Cross-View Query Consistency）

#### 3.2.1 技术动机

利用教师模型在弱增强视图上的预测，指导学生模型在强增强视图上的学习，实现跨视图的一致性正则化。

#### 3.2.2 代码实现

**文件位置**: `detr_ssod/models/dino_detr_ssod.py`

```python
def compute_pseudo_label_loss(self, student_info, teacher_info):
    # 1. 计算视图间变换矩阵
    M = self._get_trans_mat(
        teacher_info["transform_matrix"],  # 弱增强变换
        student_info["transform_matrix"]   # 强增强变换
    )
    
    # 2. 将教师伪标签变换到学生视图
    pseudo_bboxes = self._transform_bbox(
        teacher_info["det_bboxes"], M, 
        [meta["img_shape"] for meta in student_info["img_metas"]]
    )
    
    # 3. 计算一致性损失
    unsup_loss = self.unsup_loss(
        student_info, teacher_info, 
        pseudo_bboxes, pseudo_labels, pseudo_scores
    )
    
    return unsup_loss

def _get_trans_mat(self, a, b):
    """计算从视图a到视图b的变换矩阵"""
    return [bt @ at.inverse() for bt, at in zip(b, a)]
```

#### 3.2.3 数据增强流程

```
原始图像
    │
    ├──► 弱增强 (Teacher) ──► 伪标签
    │       │                      │
    │       │                      ▼
    │       │              坐标变换对齐
    │       │                      │
    │       └──────────────────────┘
    │
    └──► 强增强 (Student) ──► 一致性损失
```

### 3.3 成本过滤（Cost-based Filtering）

#### 3.3.1 技术动机

伪标签质量参差不齐，需要智能过滤机制。通过分析Hungarian匹配的cost分布，使用GMM模型自动学习过滤阈值。

#### 3.3.2 代码实现

**文件位置**: `detr_ssod/models/dino_detr_ssod.py`

```python
def _fit_gmm(self, data_points, device=None):
    """使用GMM模型拟合匹配cost分布，自动学习过滤阈值"""
    
    # 1. 收集所有匹配成本
    pos_cost_gmm = data_points
    
    # 2. 初始化GMM (2个成分: 高质量/低质量)
    means_init = np.array([min_cost, max_cost]).reshape(2, 1)
    gmm = skm.GaussianMixture(
        2,
        weights_init=[0.5, 0.5],
        means_init=means_init,
        covariance_type='full'
    )
    
    # 3. 拟合并预测
    gmm.fit(pos_cost_gmm)
    gmm_assignment = gmm.predict(pos_cost_gmm)
    scores = gmm.score_samples(pos_cost_gmm)
    
    # 4. 选择高质量簇的最优阈值
    pseudo_mask = gmm_assignment == 0
    _, pseudo_thr_ind = scores[pseudo_mask].topk(1)
    cost_thr = pos_cost_gmm[pseudo_mask][pseudo_thr_ind]
    
    return cost_thr
```

#### 3.3.3 双重过滤机制

```python
def unsup_loss(self, student_info, teacher_info, ...):
    # 1. Hungarian匹配获取cost
    cost = cls_cost + reg_cost + iou_cost
    matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
    
    # 2. GMM自适应阈值过滤
    thr_ = self._fit_gmm(cost_)
    valid_gt_inds_1 = match_gt_inds[match_gt_cost <= thr_]
    
    # 3. 固定置信度阈值过滤
    base_thr = self.train_cfg.pseudo_label_initial_score_thr  # 0.4
    valid_gt_inds_2 = torch.nonzero(gt_scores >= base_thr)
    
    # 4. 融合: 取并集
    valid_gt_inds = torch.cat((valid_gt_inds_1, valid_gt_inds_2)).unique()
```

#### 3.3.4 图像级自适应阈值

```python
# 每张图像根据自身检测质量动态调整阈值
avg_score = torch.mean(proposal_box[:, -1])
std_score = torch.std(proposal_box[:, -1])
pseudo_thr = avg_score + std_score
```

---

## 4. 教师-学生模型机制

### 4.1 模型初始化

#### 4.1.1 架构设计

```python
# detr_ssod/models/dino_detr_ssod.py
class DinoDetrSSOD(MultiSteamDetector):
    def __init__(self, model: dict, train_cfg=None, test_cfg=None):
        super().__init__(
            dict(
                teacher=build_detector(model),  # 教师模型
                student=build_detector(model)   # 学生模型
            ),
            train_cfg=train_cfg,
            test_cfg=test_cfg
        )
        # 冻结教师模型
        self.freeze("teacher")
```

#### 4.1.2 预训练权重

```python
# configs/dino_detr/dino_detr_ssod_r50_coco_120k.py
backbone=dict(
    type='ResNet',
    depth=50,
    init_cfg=dict(
        type='Pretrained', 
        checkpoint='torchvision://resnet50'  # ImageNet预训练
    ),
    ...
)
```

| 模型 | 初始化方式 | 说明 |
|------|-----------|------|
| Student | 预训练权重 | ImageNet预训练backbone |
| Teacher | 克隆Student | 初始化时完全复制Student权重 |

### 4.2 EMA权重更新

#### 4.2.1 更新公式

```
θ_teacher = momentum × θ_teacher + (1 - momentum) × θ_student
```

#### 4.2.2 代码实现

**文件位置**: `detr_ssod/utils/hooks/mean_teacher.py`

```python
@HOOKS.register_module()
class MeanTeacher(Hook):
    def __init__(self, momentum=0.999, interval=1, warm_up=100):
        self.momentum = momentum
        self.warm_up = warm_up
        self.interval = interval
    
    def before_train_iter(self, runner):
        """每个iteration更新EMA"""
        curr_step = runner.iter
        
        # Warm-up: 动量从小到大递增
        momentum = min(
            self.momentum, 
            1 - (1 + self.warm_up) / (curr_step + 1 + self.warm_up)
        )
        
        self.momentum_update(model, momentum)
    
    def momentum_update(self, model, momentum):
        for (src_name, src_parm), (tgt_name, tgt_parm) in zip(
            model.student.named_parameters(), 
            model.teacher.named_parameters()
        ):
            tgt_parm.data.mul_(momentum).add_(src_parm.data, alpha=1 - momentum)
```

#### 4.2.3 Warm-up机制

```python
# 训练初期: 动量较小，Teacher快速跟随Student
# 训练后期: 动量接近0.999，Teacher趋于稳定

momentum = min(0.999, 1 - (1 + warm_up) / (curr_step + 1 + warm_up))
```

### 4.3 协同训练流程

```
┌─────────────────────────────────────────────────────────────────────┐
│                         每个Iteration训练流程                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                     有监督分支                                │   │
│  │                                                               │   │
│  │   Labeled Images ──────► Student ──────► Loss_sup            │   │
│  │                           │                                   │   │
│  │                           ▼                                   │   │
│  │                      梯度反向传播                              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                     无监督分支                                │   │
│  │                                                               │   │
│  │   Unlabeled Images (弱增强)                                   │   │
│  │          │                                                    │   │
│  │          ▼                                                    │   │
│  │   Teacher (冻结) ──────► 伪标签                               │   │
│  │          │              │                                     │   │
│  │          │              ▼                                     │   │
│  │          │       坐标变换对齐                                  │   │
│  │          │              │                                     │   │
│  │          └──────────────┘                                     │   │
│  │                         │                                     │   │
│  │   Unlabeled Images (强增强)                                   │   │
│  │          │                                                    │   │
│  │          ▼                                                    │   │
│  │   Student ──────► Loss_unsup (一致性损失)                     │   │
│  │       │                                                       │   │
│  │       ▼                                                       │   │
│  │   梯度反向传播                                                 │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                     EMA更新                                   │   │
│  │                                                               │   │
│  │   θ_teacher = 0.999 × θ_teacher + 0.001 × θ_student          │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.4 关键特性总结

| 特性 | Teacher | Student |
|------|---------|---------|
| **权重来源** | 克隆Student + EMA更新 | 预训练权重 + 梯度更新 |
| **是否可训练** | ❌ 冻结 | ✅ 可训练 |
| **数据增强** | 弱增强 | 强增强 |
| **输出** | 伪标签 | 检测结果 |
| **参与损失** | 仅推理 | 有监督 + 无监督 |

---

## 5. 代码模块详解

### 5.1 项目结构

```
Semi-DETR-main/
├── detr_od/                    # DINO-DETR检测器
│   ├── models/
│   │   ├── dino_detr.py        # 主检测器类
│   │   ├── dense_heads/
│   │   │   ├── dino_detr_head.py       # 检测头
│   │   │   ├── dino_detr_ssod_head.py  # 半监督检测头
│   │   │   └── dn_components.py        # DN组件
│   │   └── utils/
│   │       ├── transformer.py          # Transformer模块
│   │       └── ops/                    # CUDA算子
│   └── core/
│       └── bbox/
│           └── assigners/
│               └── o2m_assigner.py     # One-to-Many匹配器
│
├── detr_ssod/                  # 半监督框架
│   ├── models/
│   │   ├── dino_detr_ssod.py   # 半监督主类
│   │   └── multi_stream_detector.py
│   ├── datasets/
│   │   ├── samplers/
│   │   │   └── semi_sampler.py # 半监督采样器
│   │   └── pipelines/
│   │       └── transforms.py   # 数据增强
│   └── utils/
│       └── hooks/
│           └── mean_teacher.py # EMA更新Hook
│
├── configs/                    # 配置文件
│   ├── detr_ssod/
│   │   ├── detr_ssod_dino_detr_r50_coco_120k.py
│   │   └── base_dino_detr_ssod_coco.py
│   └── dino_detr/
│       └── dino_detr_ssod_r50_coco_120k.py
│
└── tools/                      # 训练/测试脚本
    ├── train_detr_ssod.py
    └── test.py
```

### 5.2 核心模块说明

#### 5.2.1 DinoDetrSSOD (半监督主类)

**文件**: `detr_ssod/models/dino_detr_ssod.py`

**主要方法**:

| 方法 | 功能 |
|------|------|
| `forward_train()` | 训练前向传播，协调有监督和无监督分支 |
| `foward_unsup_train()` | 无监督训练，生成伪标签并计算一致性损失 |
| `compute_pseudo_label_loss()` | 计算伪标签损失 |
| `unsup_loss()` | 无监督损失计算，包含GMM过滤 |
| `_fit_gmm()` | GMM模型拟合，学习过滤阈值 |
| `extract_teacher_info()` | 提取教师模型信息 |
| `extract_student_info()` | 提取学生模型信息 |

#### 5.2.2 O2MAssigner (One-to-Many匹配器)

**文件**: `detr_od/core/bbox/assigners/o2m_assigner.py`

**主要功能**:
- 计算alignment metric: `score^α × IoU^β`
- 选择top-k候选正样本
- 动态正样本数量估计

#### 5.2.3 MeanTeacher (EMA更新Hook)

**文件**: `detr_ssod/utils/hooks/mean_teacher.py`

**主要功能**:
- 初始化时克隆Student权重到Teacher
- 每个iteration执行EMA更新
- Warm-up期间动态调整动量

---

## 6. 训练与评估流程

### 6.1 训练命令

```bash
# 激活环境
conda activate semidetr
cd ~/projects/Semi-DETR-main
export PYTHONPATH=$(pwd):$PYTHONPATH

# 训练 (10%标注数据, fold=1)
python tools/train_detr_ssod.py \
    configs/detr_ssod/detr_ssod_dino_detr_r50_coco_120k.py \
    --work-dir work_dirs/detr_ssod_dino_detr_r50_coco_120k/10/1 \
    --cfg-options fold=1 percent=10
```

### 6.2 评估命令

```bash
# 评估模型
python tools/test.py \
    configs/detr_ssod/detr_ssod_dino_detr_r50_coco_120k.py \
    work_dirs/detr_ssod_dino_detr_r50_coco_120k/10/1/epoch_12.pth \
    --eval bbox
```

### 6.3 关键配置参数

```python
# configs/detr_ssod/detr_ssod_dino_detr_r50_coco_120k.py

# 数据配置
data = dict(
    samples_per_gpu=2,        # 每GPU batch size
    workers_per_gpu=5,        # 数据加载线程数
)

# 半监督配置
semi_wrapper = dict(
    type="DinoDetrSSOD",
    train_cfg=dict(
        pseudo_label_initial_score_thr=0.4,  # 伪标签置信度阈值
        unsup_weight=4.0,                     # 无监督损失权重
    ),
)

# EMA配置
custom_hooks = [
    dict(type="MeanTeacher", momentum=0.999, warm_up=0),
]

# 训练配置
runner = dict(type="IterBasedRunner", max_iters=120000)
```

---

## 7. 常见问题与解决方案

### 7.1 CUDA相关错误

**问题**: `CUSOLVER_STATUS_INTERNAL_ERROR`

**原因**: CUDA版本与PyTorch版本不兼容

**解决方案**:
```bash
# 确保CUDA环境变量正确
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
```

### 7.2 mmcv版本冲突

**问题**: `MMCV==2.0.0 is used but incompatible`

**解决方案**:
```python
# 注释掉版本检查
# mmdet/__init__.py
# assert (mmcv_version >= digit_version(mmcv_minimum_version)...)
```

### 7.3 显存不足

**问题**: `CUDA out of memory`

**解决方案**:
```python
# 减小batch size
data = dict(
    samples_per_gpu=2,  # 从5改为2
)
```

### 7.4 数据路径错误

**问题**: `Permission denied: '/root/paddlejob/...'`

**解决方案**:
```bash
# 批量替换配置文件中的路径
find ~/projects/Semi-DETR-main/configs -name "*.py" -exec \
    sed -i 's|/root/paddlejob/workspace/env_run/output/temp/data/coco|/home/xfb/data/coco/dataset|g' {} \;

# 修改mmdetection base配置
sed -i 's|/root/paddlejob/...|/home/xfb/data/coco/dataset|g' \
    ~/projects/Semi-DETR-main/thirdparty/mmdetection/configs/_base_/datasets/coco_detection.py
```

### 7.5 WSL重新进入

```bash
# 打开PowerShell
wsl -d Ubuntu-22.04

# 激活环境
conda activate semidetr
cd ~/projects/Semi-DETR-main
export PYTHONPATH=$(pwd):$PYTHONPATH
```

---

## 附录

### A. 预训练模型下载

| 设置 | mAP | 下载链接 |
|------|-----|----------|
| COCO 1% | 30.50% | [Google Drive](https://drive.google.com/file/d/1guWr-7Klvt8w16on082JUPdnsPBv8b_D/view) |
| COCO 5% | 40.10% | [Google Drive](https://drive.google.com/file/d/1R7FfkOkiR57WSleKJmHj2BitVj_xfqam/view) |
| COCO 10% | 43.30% | [Google Drive](https://drive.google.com/file/d/1gYBzI_SANfl9_HqklzWJ_hBE55Gh4wnI/view) |
| COCO Full | 50.50% | [Google Drive](https://drive.google.com/file/d/17OPojkoIU7wwRcT4xNuuXONNy4zIymA4/view) |

### B. 参考链接

- [DINO-DETR论文](https://arxiv.org/abs/2203.03605)
- [Mean Teacher论文](https://arxiv.org/abs/1703.01780)
- [MMDetection文档](https://mmdetection.readthedocs.io/)

---

*文档生成时间: 2026-05-09*
