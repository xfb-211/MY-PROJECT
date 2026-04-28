import warnings
warnings.filterwarnings('ignore')
import os
from ultralytics import RTDETR
import torch
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 模型配置文件路径
# model_yaml_path = './ultralytics/cfg/models/rt-detr/rtdetr-l.yaml'
# 数据集配置文件路径.
# data_yaml_path = './datasets/coco.yaml'
data_yaml_path = './datasets/coco_pretrain.yaml'

# if __name__ == '__main__':
#     # model = RTDETR(model_yaml_path)
#     model = RTDETR('D:/Projects/ultralytics-main/weights/rtdetr-l.pt')  # 加载预训练权重
#     # 训练模型
#     results = model.train(data=data_yaml_path,
#                           imgsz=640,
#                           epochs=100,
#                           batch=8,
#                           workers=0,
#                           device=0,
#                           lr0=0.00008,
#                           lrf=0.01,
#                           momentum=0.9,
#                           weight_decay=0.0005,
#                           warmup_epochs=1,
#                           optimizer='AdamW',
#                           project='runs/rtdetr-l',
#                           name='RTDETR-full',
#                           close_mosaic=15,
#                           patience=25,
#                           amp=True,
#                           exist_ok=True,
#                           save=True,
#                           val=True
#                           )

if __name__ == '__main__':
    model = RTDETR('rtdetr-l.pt')
    results = model.train(
        data='./datasets/coco_pretrain.yaml',
        imgsz=640,
        epochs=35,
        batch=8,
        workers=0,
        device=0,
        project='runs/coco_pretrain_teacher',
        name='teacher2',
        seed=123,
        optimizer='AdamW',
        lr0=0.00008,
        weight_decay=0.0008,
        lrf=0.008,
        patience=12,
        close_mosaic=8,
        hsv_h=0.012, hsv_s=0.6, hsv_v=0.35,
        fliplr=0.3, mosaic=0.4,
        amp=False,
        exist_ok=True,
        save=True,
        val=True,
    )

# if __name__ == '__main__':
#     model = RTDETR('rtdetr-l.pt')
#     results = model.train(
#         data='./datasets/coco_pretrain.yaml',
#         imgsz=640,
#         epochs=35,
#         batch=8,
#         workers=0,
#         device=0,
#         project='runs/coco_pretrain_teacher',
#         name='teacher1',
#         seed=42,
#         optimizer='AdamW',
#         lr0=0.0001,
#         weight_decay=0.0001,
#         lrf=0.001,
#         patience=10,
#         close_mosaic=10,
#         amp=False,
#         exist_ok=True,
#         save=True,
#         val=True,
#     )

# if __name__ == '__main__':
#     model = RTDETR(model_yaml_path)
#     # 训练模型
#     results = model.train(data=data_yaml_path,
#                           imgsz=1024,
#                           epochs=100,
#                           batch=4,
#                           workers=0,
#                           device=0,
#                           project='runs/semi_train',
#                           name='P-RTDETR',
#                           )

# if __name__ == '__main__':
#     model = RTDETR(model_yaml_path)
#     # 训练模型
#     results = model.train(data=data_yaml_path,
#                           imgsz=1024,
#                           epochs=50,
#                           batch=4,
#                           workers=0,
#                           device=0,
#                           project='runs/pretrain_teacher',
#                           name='teacher1',
#                           seed=42,
#                           )

# if __name__ == '__main__':
#     model = RTDETR(model_yaml_path)
#     # 训练模型
#     results = model.train(data=data_yaml_path,
#                           imgsz=1024,
#                           epochs=50,
#                           batch=4,
#                           workers=0,
#                           device=0,
#                           project='runs/pretrain_teacher',
#                           name='teacher2',
#                           seed=123,
#                           )

    # 可选：打印训练结果
    print("训练完成！最终指标：")
    print(f"mAP@0.5: {results.results_dict.get('metrics/mAP50(B)', 0):.4f}")
    print(f"mAP@0.5:0.95: {results.results_dict.get('metrics/mAP50-95(B)', 0):.4f}")


# if __name__ == '__main__':
#     # 1. 初始化RT-DETR模型
#     model = RTDETR(model_yaml_path)
#
#     # 2. 训练模型（补充更多实用参数，适配弱监督训练）
#     results = model.train(
#         data=data_yaml_path,          # 数据集配置文件
#         imgsz=640,                    # 输入图像尺寸
#         epochs=100,                   # 训练轮数
#         batch=4,                      # 批次大小（根据显存调整）
#         workers=0,                    # 数据加载线程（Windows下建议设为0）
#         project=project_path,         # 结果保存根目录
#         name='PseudoRTDETR',          # 实验名称（会在project下创建该文件夹）
#         device=0,                     # 使用第0块GPU（CPU设为cpu）
#         pretrained=True,              # 启用预训练权重
#         lr0=0.001,                    # 初始学习率（可根据需求调整）
#         patience=30,                  # 早停耐心值（50轮无提升则停止），方案1：50；方案2：30
#         save=True,                    # 保存最佳模型
#         val=True,                     # 训练时验证（建议开启）
#         # # 以下是弱监督相关（如果你的rtdetr-weak.yaml启用了弱监督）
#         # weak_supervised=True,        # 启用弱监督（需对应配置支持）
#         # reliability_weight=0.5,      # 伪标签可靠性损失权重
#     )