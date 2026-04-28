import os
import json
import shutil
import random
from tqdm import tqdm
from pycocotools.coco import COCO

# ================= 配置区域 =================
# 原始 COCO 数据集根目录 (包含 train2017, val2017, annotations 文件夹)
COCO_ROOT = './COCO'

# 输出目录 (即你要生成的 COCO_Semi)
OUTPUT_ROOT = './COCO_Semi'

# 有标签数据比例
LABELED_RATIO = 0.1


# ===========================================

def coco_to_yolo(bbox, img_width, img_height):
    """
    将 COCO 格式的 bbox [x, y, w, h] 转换为 YOLO 格式 [x_center, y_center, w, h] (归一化)
    """
    x, y, w, h = bbox

    # 计算中心点
    x_center = x + w / 2.0
    y_center = y + h / 2.0

    # 归一化到 0-1 之间
    x_center /= img_width
    y_center /= img_height
    w /= img_width
    h /= img_height

    return [x_center, y_center, w, h]


def convert_dataset(split_name, json_path, source_img_dir, target_img_dir, target_label_dir, is_labeled_split=False):
    """
    处理单个数据集部分 (训练或验证)
    """
    print(f"\n正在处理: {split_name}...")
    coco = COCO(json_path)

    # 获取所有图片 ID
    img_ids = coco.getImgIds()

    # 如果是训练集且需要切分有标签/无标签，这里进行随机打乱和切片
    # 注意：如果是 val 集，is_labeled_split 为 False，则处理所有图片
    if is_labeled_split:
        random.shuffle(img_ids)
        split_point = int(len(img_ids) * LABELED_RATIO)
        img_ids = img_ids[:split_point]
        print(f" -> 随机抽取 {len(img_ids)} 张图片作为有标签数据 (比例 {LABELED_RATIO})")

    # 创建目标文件夹
    os.makedirs(target_img_dir, exist_ok=True)
    os.makedirs(target_label_dir, exist_ok=True)

    # 建立 category_id 到 class_index 的映射
    # COCO 的 category_id 是不连续的 (如 1, 2, ..., 90)，YOLO 需要连续的 0, 1, 2...
    cats = coco.loadCats(coco.getCatIds())
    cat_id_to_idx = {cat['id']: idx for idx, cat in enumerate(cats)}

    # 保存类别名称到文件 (方便后续训练查看)
    if split_name == 'train_labeled':  # 只保存一次即可
        with open(os.path.join(OUTPUT_ROOT, 'classes.txt'), 'w') as f:
            for cat in sorted(cats, key=lambda k: k['id']):
                f.write(cat['name'] + '\n')

    for img_id in tqdm(img_ids, desc=f"转换 {split_name}"):
        # 1. 获取图片信息
        img_info = coco.loadImgs(img_id)[0]
        img_file_name = img_info['file_name']
        img_width = img_info['width']
        img_height = img_info['height']

        # 源图片路径
        src_img_path = os.path.join(source_img_dir, img_file_name)
        # 目标图片路径
        dst_img_path = os.path.join(target_img_dir, img_file_name)

        # 复制图片
        if os.path.exists(src_img_path):
            shutil.copy(src_img_path, dst_img_path)
        else:
            print(f"警告: 图片不存在 {src_img_path}")
            continue

        # 2. 获取标注信息
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        # 3. 写入 YOLO 格式 txt 文件
        label_file_name = os.path.splitext(img_file_name)[0] + '.txt'
        dst_label_path = os.path.join(target_label_dir, label_file_name)

        with open(dst_label_path, 'w') as f:
            for ann in anns:
                # 过滤掉 crowd 区域 (可选，通常建议过滤)
                if ann.get('iscrowd', 0) == 1:
                    continue

                category_id = ann['category_id']
                bbox = ann['bbox']

                # 转换为 YOLO 格式
                yolo_bbox = coco_to_yolo(bbox, img_width, img_height)

                # 获取类别索引 (0-based)
                class_idx = cat_id_to_idx[category_id]

                # 写入文件: <class> <x_center> <y_center> <w> <h>
                f.write(f"{class_idx} {' '.join(map(str, yolo_bbox))}\n")


def main():
    # 设置随机种子以保证复现性
    random.seed(42)

    # 1. 处理训练集 -> 拆分为 train_labeled (10%)
    convert_dataset(
        split_name='train_labeled',
        json_path=os.path.join(COCO_ROOT, 'annotations/instances_train2017.json'),
        source_img_dir=os.path.join(COCO_ROOT, 'train2017'),
        target_img_dir=os.path.join(OUTPUT_ROOT, 'images/train_labeled'),
        target_label_dir=os.path.join(OUTPUT_ROOT, 'labels/train_labeled'),
        is_labeled_split=True
    )

    # 2. 处理训练集 -> 拆分为 train_unlabeled (90%)
    # 注意：这里不复制标签，只复制图片
    print(f"\n正在处理: train_unlabeled (仅复制图片)...")
    coco_train = COCO(os.path.join(COCO_ROOT, 'annotations/instances_train2017.json'))
    all_train_ids = coco_train.getImgIds()
    random.shuffle(all_train_ids)
    split_point = int(len(all_train_ids) * LABELED_RATIO)
    unlabeled_ids = all_train_ids[split_point:]  # 取后 90%

    os.makedirs(os.path.join(OUTPUT_ROOT, 'images/train_unlabeled'), exist_ok=True)

    for img_id in tqdm(unlabeled_ids, desc="复制无标签图片"):
        img_info = coco_train.loadImgs(img_id)[0]
        src_path = os.path.join(COCO_ROOT, 'train2017', img_info['file_name'])
        dst_path = os.path.join(OUTPUT_ROOT, 'images/train_unlabeled', img_info['file_name'])
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)

    # 3. 处理验证集 -> val (100%)
    convert_dataset(
        split_name='val',
        json_path=os.path.join(COCO_ROOT, 'annotations/instances_val2017.json'),
        source_img_dir=os.path.join(COCO_ROOT, 'val2017'),
        target_img_dir=os.path.join(OUTPUT_ROOT, 'images/val'),
        target_label_dir=os.path.join(OUTPUT_ROOT, 'labels/val'),
        is_labeled_split=False
    )

    print("\n" + "=" * 30)
    print("转换完成！")
    print(f"输出目录结构: {OUTPUT_ROOT}")
    print("=" * 30)


if __name__ == '__main__':
    main()