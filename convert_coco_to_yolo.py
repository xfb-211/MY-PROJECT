import os
import json
import shutil
from tqdm import tqdm
from pycocotools.coco import COCO

# ================= 配置区域 =================
# 1. 原始 COCO 数据集根目录 (包含 train2017, val2017, annotations)
COCO_ROOT = './COCO'  # 请确保这个路径是正确的

# 2. 输出目录 (转换后的 YOLO 格式数据集)
OUTPUT_ROOT = './COCO_yolo'


# ===========================================

def coco_bbox_to_yolo(bbox, img_width, img_height):
    """
    将 COCO 格式的 bbox [x, y, w, h] 转换为 YOLO 格式 [x_center, y_center, w, h] (归一化)
    """
    x, y, w, h = bbox

    # 计算中心点
    x_center = x + w / 2.0
    y_center = y + h / 2.0

    # 归一化
    x_center /= img_width
    y_center /= img_height
    w /= img_width
    h /= img_height

    return [x_center, y_center, w, h]


def process_split(split_name, json_file, source_img_folder):
    """
    处理单个数据集划分 (train 或 val)
    """
    print(f"\n🚀 正在处理: {split_name} ...")

    # 初始化 COCO API
    coco = COCO(json_file)

    # 获取所有图片 ID
    img_ids = coco.getImgIds()

    # 建立 category_id 到 class_index 的映射
    cats = coco.loadCats(coco.getCatIds())
    # 排序确保类别顺序一致
    cats = sorted(cats, key=lambda x: x['id'])
    cat_id_to_idx = {cat['id']: idx for idx, cat in enumerate(cats)}

    # 如果是训练集，保存一份 classes.txt 方便后续查看
    if split_name == 'train2017':
        classes_path = os.path.join(OUTPUT_ROOT, 'classes.txt')
        with open(classes_path, 'w', encoding='utf-8') as f:
            for cat in cats:
                f.write(f"{cat['id']}: {cat['name']}\n")
        print(f"   类别文件已保存至: {classes_path}")

    # 定义输出路径
    out_img_dir = os.path.join(OUTPUT_ROOT, 'images', split_name)
    out_label_dir = os.path.join(OUTPUT_ROOT, 'labels', split_name)

    # 创建目录
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_label_dir, exist_ok=True)

    # 遍历所有图片
    for img_id in tqdm(img_ids, desc=f"转换 {split_name}"):
        # 1. 获取图片信息
        img_info = coco.loadImgs(img_id)[0]
        file_name = img_info['file_name']
        width = img_info['width']
        height = img_info['height']

        # 2. 复制图片
        src_img_path = os.path.join(source_img_folder, file_name)
        dst_img_path = os.path.join(out_img_dir, file_name)

        if os.path.exists(src_img_path):
            shutil.copy(src_img_path, dst_img_path)
        else:
            print(f"⚠️ 警告: 图片不存在 {src_img_path}")
            continue

        # 3. 获取并转换标注
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        # 生成对应的 txt 文件名
        txt_filename = os.path.splitext(file_name)[0] + '.txt'
        txt_path = os.path.join(out_label_dir, txt_filename)

        with open(txt_path, 'w', encoding='utf-8') as f:
            for ann in anns:
                # 跳过 crowd 标注
                if ann.get('iscrowd', 0) == 1:
                    continue

                cat_id = ann['category_id']
                bbox = ann['bbox']  # [x, y, w, h]

                # 获取 YOLO 类别索引
                class_idx = cat_id_to_idx[cat_id]

                # 坐标转换
                yolo_bbox = coco_bbox_to_yolo(bbox, width, height)

                # 写入文件: <class_idx> <x_center> <y_center> <w> <h>
                f.write(f"{class_idx} {' '.join(map(str, yolo_bbox))}\n")


def main():
    # 🛠️ 修复：在开始之前，先创建输出根目录，防止写入 classes.txt 时找不到目录
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    print(f"📂 源目录: {COCO_ROOT}")
    print(f"📦 输出目录: {OUTPUT_ROOT}")

    # 处理训练集
    process_split(
        split_name='train2017',
        json_file=os.path.join(COCO_ROOT, 'annotations/instances_train2017.json'),
        source_img_folder=os.path.join(COCO_ROOT, 'train2017')
    )

    # 处理验证集
    process_split(
        split_name='val2017',
        json_file=os.path.join(COCO_ROOT, 'annotations/instances_val2017.json'),
        source_img_folder=os.path.join(COCO_ROOT, 'val2017')
    )

    print("\n✅ 转换完成！")


if __name__ == '__main__':
    main()