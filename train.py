import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from dataset import YelpReviewDataset
from model import MultiModalSentimentModel
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import argparse

# 设置随机种子以复现结果
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
# 定义数据转换
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 数据集和数据加载器
def get_dataloaders(data_dir, batch_size=16):
    train_dataset = YelpReviewDataset(
        csv_file=os.path.join(data_dir, 'train.csv'),
        img_dir=os.path.join(data_dir, 'images'),
        transform=data_transforms
    )
    
    val_dataset = YelpReviewDataset(
        csv_file=os.path.join(data_dir, 'val.csv'),
        img_dir=os.path.join(data_dir, 'images'),
        transform=data_transforms
    )
    
    test_dataset = YelpReviewDataset(
        csv_file=os.path.join(data_dir, 'test.csv'),
        img_dir=os.path.join(data_dir, 'images'),
        transform=data_transforms
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4)
    
    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset

# 训练函数
def train_model(model, train_loader, val_loader, epochs=10, learning_rate=2e-5, device='cuda'):
    # 移动模型到设备
    model = model.to(device)
    
    # 定义损失函数
    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()
    
    # 定义优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    # 跟踪训练过程的指标
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'val_f1': []
    }
    
    best_val_f1 = 0.0
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            # 获取数据
            image = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            sentiment = batch['sentiment'].to(device)
            rating = batch['rating'].to(device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            sentiment_logits, rating_pred, _ = model(image, input_ids, attention_mask)
            
            # 计算损失
            cls_loss = criterion_cls(sentiment_logits, sentiment)
            reg_loss = criterion_reg(rating_pred.squeeze(), rating)
            loss = cls_loss + 0.3 * reg_loss  # 权重平衡
            
            # 反向传播
            loss.backward()
            
            # 更新参数
            optimizer.step()
            
            # 更新统计
            train_loss += loss.item()
            
            # 计算精度
            _, predicted = torch.max(sentiment_logits, 1)
            train_total += sentiment.size(0)
            train_correct += (predicted == sentiment).sum().item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # 获取数据
                image = batch['image'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                sentiment = batch['sentiment'].to(device)
                rating = batch['rating'].to(device)
                
                # 前向传播
                sentiment_logits, rating_pred, _ = model(image, input_ids, attention_mask)
                
                # 计算损失
                cls_loss = criterion_cls(sentiment_logits, sentiment)
                reg_loss = criterion_reg(rating_pred.squeeze(), rating)
                loss = cls_loss + 0.3 * reg_loss
                
                val_loss += loss.item()
                
                # 获取预测
                _, predicted = torch.max(sentiment_logits, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_targets.extend(sentiment.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = accuracy_score(val_targets, val_preds)
        _, _, val_f1, _ = precision_recall_fscore_support(
            val_targets, val_preds, average='binary', zero_division=0
        )
        
        # 学习率调度
        scheduler.step(avg_val_loss)
        
        # 记录历史
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
        
        # 保存最佳模型
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), 'models/best_multimodal_model.pth')
            print("Saved best model checkpoint.")
    
    # 保存训练历史
    with open('models/training_history.json', 'w') as f:
        json.dump(history, f)
    
    return history

# 绘制训练历史
def plot_training_history(history):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.plot(history['val_f1'], label='Val F1')
    plt.title('Metrics Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/training_history.png')
    plt.close()

# 主函数
def main(args):
    # 设置随机种子
    set_seed(args.seed)
    
    # 检查GPU
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}")
    
    # 获取数据加载器
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = get_dataloaders(
        args.data_dir, 
        batch_size=args.batch_size
    )
    print(f"Data loaded: {len(train_loader.dataset)} training, {len(val_loader.dataset)} validation")
    
    # 初始化模型
    model = MultiModalSentimentModel(num_classes=2)
    
    # 训练模型
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=device
    )
    
    # 绘制训练历史
    plot_training_history(history)
    
    print("Training complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='训练多模态情感分析模型')
    parser.add_argument('--data_dir', type=str, default='data/yelp',
                        help='数据目录')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批量大小')
    parser.add_argument('--epochs', type=int, default=10,
                        help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='学习率')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--cpu', action='store_true',
                        help='强制使用CPU')
    
    args = parser.parse_args()
    main(args)