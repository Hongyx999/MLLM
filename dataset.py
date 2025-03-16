import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import os
from transformers import BertTokenizer

class YelpReviewDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, max_length=128):
        """
        Args:
            csv_file (string): 包含评论数据的CSV文件路径
            img_dir (string): 包含图像的目录
            transform (callable, optional): 图像转换操作
            max_length (int): BERT输入的最大长度
        """
        self.reviews_df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_length = max_length
    
    def __len__(self):
        return len(self.reviews_df)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # 获取数据
        row = self.reviews_df.iloc[idx]
        img_path = os.path.join(self.img_dir, f"{row['photo_id']}.jpg")
        review_text = row['text']  # Yelp中是'text'而不是'reviewText'
        rating = row['stars']      # Yelp中是'stars'而不是'overall'
        sentiment = row['sentiment']
        
        # 加载图像并应用转换
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # 创建一个空白的RGB图像作为替代
            image = Image.new('RGB', (224, 224), color='black')
            if self.transform:
                image = self.transform(image)
        
        # 处理文本
        encoding = self.tokenizer(
            review_text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 创建样本字典
        sample = {
            'business_id': row['business_id'],
            'review_id': row['review_id'],
            'image': image,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'rating': torch.tensor(rating, dtype=torch.float),
            'sentiment': torch.tensor(sentiment, dtype=torch.long),
            'text': review_text
        }
        
        return sample