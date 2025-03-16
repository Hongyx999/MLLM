import json
import pandas as pd
import os
import shutil
from tqdm import tqdm
import random
from PIL import Image
import logging
import argparse

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def preprocess_yelp_data(data_dir, output_dir, sample_size=5000, resize_images=True):
    """
    预处理Yelp数据集
    
    参数:
    data_dir: Yelp数据集所在的目录
    output_dir: 输出目录
    sample_size: 处理的样本数量
    resize_images: 是否调整图像大小
    """
    # 设置路径
    reviews_path = os.path.join(data_dir, "yelp_academic_dataset_review.json")
    business_path = os.path.join(data_dir, "yelp_academic_dataset_business.json")
    photos_dir = os.path.join(data_dir, "photos")
    photos_metadata_path = os.path.join(photos_dir, "photos.json")
    
    # 创建输出目录
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    
    # 加载商家数据
    logging.info("加载商家数据...")
    businesses = {}
    with open(business_path, 'r') as f:
        for line in tqdm(f):
            data = json.loads(line)
            businesses[data['business_id']] = data
    
    # 加载照片元数据
    logging.info("加载照片元数据...")
    photo_metadata = {}
    with open(photos_metadata_path, 'r') as f:
        photo_data = json.load(f)
        for photo in tqdm(photo_data):
            # 包含所有照片类别，但可以根据需要过滤
            photo_metadata[photo["photo_id"]] = photo
    
    # 创建照片到商家ID的映射
    photo_to_business = {}
    for photo_id, meta in photo_metadata.items():
        photo_to_business[photo_id] = meta["business_id"]
    
    # 加载评论数据
    logging.info("加载评论数据...")
    reviews = []
    with open(reviews_path, 'r') as f:
        count = 0
        for line in tqdm(f):
            if count >= sample_size * 3:  # 获取更多然后筛选
                break
            
            data = json.loads(line)
            
            # 只选择评分明确的评论(1,2,4,5星)
            if data['stars'] in [1, 2, 4, 5]:
                reviews.append(data)
                count += 1
    
    # 转换为DataFrame
    reviews_df = pd.DataFrame(reviews)
    
    # 添加情感标签 (0=负面, 1=正面)
    reviews_df['sentiment'] = reviews_df['stars'].apply(lambda x: 0 if x <= 2 else 1)
    
    # 平衡采样
    logging.info("平衡正面和负面样本...")
    n_per_class = min(sample_size // 2, len(reviews_df[reviews_df['sentiment'] == 0]), 
                      len(reviews_df[reviews_df['sentiment'] == 1]))
    
    neg_samples = reviews_df[reviews_df['sentiment'] == 0].sample(n=n_per_class, random_state=42)
    pos_samples = reviews_df[reviews_df['sentiment'] == 1].sample(n=n_per_class, random_state=42)
    balanced_df = pd.concat([neg_samples, pos_samples])
    
    logging.info(f"选择了 {len(balanced_df)} 条评论")
    
    # 为每个评论找到对应的商家照片
    logging.info("处理照片...")
    selected_reviews = []
    
    for _, review in tqdm(balanced_df.iterrows(), total=len(balanced_df)):
        business_id = review['business_id']
        
        # 查找该商家的照片
        business_photos = [p for p, b_id in photo_to_business.items() 
                          if b_id == business_id]
        
        if business_photos:
            # 随机选择一张照片
            photo_id = random.choice(business_photos)
            photo_path = os.path.join(photos_dir, f"{photo_id}.jpg")
            
            if os.path.exists(photo_path):
                # 处理并保存图像
                dest_path = os.path.join(output_dir, "images", f"{photo_id}.jpg")
                
                try:
                    if resize_images:
                        img = Image.open(photo_path)
                        img = img.convert('RGB')  # 确保是RGB格式
                        img = img.resize((224, 224))  # 调整大小为模型输入尺寸
                        img.save(dest_path)
                    else:
                        shutil.copy(photo_path, dest_path)
                        
                    # 添加到选定评论
                    review_dict = review.to_dict()
                    review_dict['photo_id'] = photo_id
                    selected_reviews.append(review_dict)
                except Exception as e:
                    logging.warning(f"处理图像 {photo_id} 时出错: {e}")
    
    # 创建最终数据集
    final_df = pd.DataFrame(selected_reviews)
    logging.info(f"最终数据集大小: {len(final_df)} 条带照片的评论")
    
    # 保存数据
    final_df.to_csv(os.path.join(output_dir, "reviews_with_photos.csv"), index=False)
    
    # 拆分训练/验证/测试集
    train_df = final_df.sample(frac=0.8, random_state=42)
    temp_df = final_df.drop(train_df.index)
    val_df = temp_df.sample(frac=0.5, random_state=42)
    test_df = temp_df.drop(val_df.index)
    
    # 保存拆分数据
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    
    logging.info("数据准备完成!")
    logging.info(f"训练集: {len(train_df)} 样本")
    logging.info(f"验证集: {len(val_df)} 样本")
    logging.info(f"测试集: {len(test_df)} 样本")
    logging.info(f"图像保存在: {os.path.join(output_dir, 'images')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='预处理Yelp数据集')
    parser.add_argument('--data_dir', type=str, default='yelp_dataset', 
                        help='Yelp数据集所在的目录')
    parser.add_argument('--output_dir', type=str, default='data/yelp', 
                        help='输出目录')
    parser.add_argument('--sample_size', type=int, default=5000, 
                        help='处理的样本数量')
    parser.add_argument('--no_resize', action='store_true', 
                        help='不调整图像大小')
    
    args = parser.parse_args()
    
    preprocess_yelp_data(
        args.data_dir, 
        args.output_dir, 
        args.sample_size, 
        not args.no_resize
    )