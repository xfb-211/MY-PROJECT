import os
import random
import shutil


def split_dataset(temp_img_dir, temp_lbl_dir, labeled_img_dir, labeled_lbl_dir, unlabeled_img_dir, ratio=0.1):
    """
    按比例划分有标签/无标签训练集
    :param ratio: 有标签数据占比，如0.1表示10%
    """
    os.makedirs(labeled_img_dir, exist_ok=True)
    os.makedirs(labeled_lbl_dir, exist_ok=True)
    os.makedirs(unlabeled_img_dir, exist_ok=True)

    # 获取所有图像文件名 (确保扩展名匹配你的实际文件，这里假设是 .jpg)
    all_imgs = [f for f in os.listdir(temp_img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    random.shuffle(all_imgs)  # 随机打乱

    # 计算有标签数据数量
    num_labeled = int(len(all_imgs) * ratio)
    labeled_imgs = all_imgs[:num_labeled]
    unlabeled_imgs = all_imgs[num_labeled:]

    print(f"总训练图像数: {len(all_imgs)}")
    print(f"有标签图像数: {len(labeled_imgs)} ({ratio * 100}%)")
    print(f"无标签图像数: {len(unlabeled_imgs)} ({(1 - ratio) * 100}%)")

    # 移动有标签数据
    for img_name in labeled_imgs:
        # 移动图像
        src_img = os.path.join(temp_img_dir, img_name)
        dst_img = os.path.join(labeled_img_dir, img_name)
        if os.path.exists(src_img):
            shutil.move(src_img, dst_img)

        # 移动标注
        lbl_name = os.path.splitext(img_name)[0] + '.txt'  # 更加安全的获取文件名
        src_lbl = os.path.join(temp_lbl_dir, lbl_name)
        dst_lbl = os.path.join(labeled_lbl_dir, lbl_name)

        if os.path.exists(src_lbl):
            shutil.move(src_lbl, dst_lbl)
        else:
            # 如果没有对应的标注文件，可以选择打印警告或者跳过
            print(f"警告: 找不到标注文件 {src_lbl}")

    # 移动无标签数据（仅移动图像）
    for img_name in unlabeled_imgs:
        src_img = os.path.join(temp_img_dir, img_name)
        dst_img = os.path.join(unlabeled_img_dir, img_name)
        if os.path.exists(src_img):
            shutil.move(src_img, dst_img)

    # --- 修改部分开始 ---
    # 使用 shutil.rmtree 强制删除目录及其内容，避免因残留文件报错
    # 注意：这会删除目录下剩余的所有文件，请确保数据已经正确移动
    try:
        if os.path.exists(temp_img_dir):
            shutil.rmtree(temp_img_dir)
            print(f"已删除临时目录: {temp_img_dir}")

        if os.path.exists(temp_lbl_dir):
            shutil.rmtree(temp_lbl_dir)
            print(f"已删除临时目录: {temp_lbl_dir}")
    except Exception as e:
        print(f"删除临时目录时出错: {e}")
    # --- 修改部分结束 ---


# 设置随机种子，保证可复现
random.seed(42)

# 运行脚本
split_dataset(
    temp_img_dir='VisDrone_Semi/images/train_temp',
    temp_lbl_dir='VisDrone_Semi/labels/train_temp',
    labeled_img_dir='VisDrone_Semi/images/train_labeled',
    labeled_lbl_dir='VisDrone_Semi/labels/train_labeled',
    unlabeled_img_dir='VisDrone_Semi/images/train_unlabeled',
    ratio=0.2
)