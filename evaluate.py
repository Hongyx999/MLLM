import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from dataset import YelpReviewDataset
from model import MultiModalSentimentModel
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
from captum.attr import IntegratedGradients
from transformers import BertTokenizer
import wordcloud
from wordcloud import WordCloud
from nltk.corpus import stopwords
import nltk
import os
import argparse

# 下载NLTK数据
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# 数据转换
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 评估模型
def evaluate_model(model, test_loader, device='cuda'):
    model.eval()
    
    all_preds = []
    all_targets = []
    all_ratings = []
    all_pred_ratings = []
    all_attention_weights = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # 获取数据
            image = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            sentiment = batch['sentiment'].to(device)
            rating = batch['rating'].to(device)
            
            # 前向传播
            sentiment_logits, rating_pred, attention_weights = model(image, input_ids, attention_mask)
            
            # 获取预测
            _, predicted = torch.max(sentiment_logits, 1)
            
            # 保存结果
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(sentiment.cpu().numpy())
            all_ratings.extend(rating.cpu().numpy())
            all_pred_ratings.extend(rating_pred.squeeze().cpu().numpy())
            all_attention_weights.extend(attention_weights.cpu().numpy())
    
    # 计算评估指标
    cm = confusion_matrix(all_targets, all_preds)
    report = classification_report(all_targets, all_preds, output_dict=True)
    
    # 将评估结果转换为DataFrame
    report_df = pd.DataFrame(report).transpose()
    
    # 计算评分预测的MAE
    mae = np.mean(np.abs(np.array(all_ratings) - np.array(all_pred_ratings)))
    
    # 计算平均注意力权重
    avg_attention = np.mean(all_attention_weights, axis=0)
    
    return {
        'confusion_matrix': cm,
        'classification_report': report_df,
        'mae': mae,
        'avg_attention': avg_attention,
        'predictions': all_preds,
        'targets': all_targets,
        'ratings': all_ratings,
        'pred_ratings': all_pred_ratings,
        'attention_weights': all_attention_weights
    }

# 可视化混淆矩阵
def plot_confusion_matrix(cm):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('results/confusion_matrix.png')
    plt.close()

# 可视化模态注意力权重
def plot_attention_weights(avg_attention):
    plt.figure(figsize=(8, 6))
    labels = ['Image', 'Text']
    plt.bar(labels, avg_attention)
    plt.title('Average Attention Weights for Each Modality')
    plt.ylabel('Weight')
    plt.ylim(0, 1)
    for i, v in enumerate(avg_attention):
        plt.text(i, v + 0.01, f'{v:.4f}', ha='center')
    plt.savefig('results/modality_attention.png')
    plt.close()

# 特征归因可视化
# 特征归因可视化
# 特征归因可视化
def visualize_attributions(model, dataset, tokenizer, device='cuda', num_samples=5):
    # 创建一个包装模型类来处理特征归因
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model, image, attention_mask):
            super(ModelWrapper, self).__init__()
            self.model = model
            self.image = image
            self.attention_mask = attention_mask
            
        def forward(self, input_ids):
            return self.model(self.image, input_ids, self.attention_mask)[0]
    
    # 随机选择样本
    indices = np.random.choice(len(dataset), size=num_samples, replace=False)
    
    for idx in indices:
        sample = dataset[idx]
        
        # 准备数据
        image = sample['image'].unsqueeze(0).to(device)
        input_ids = sample['input_ids'].unsqueeze(0).to(device)
        attention_mask = sample['attention_mask'].unsqueeze(0).to(device)
        sentiment = sample['sentiment'].item()
        text = sample['text']
        business_id = sample['business_id']
        
        # 获取模型预测
        with torch.no_grad():
            sentiment_logits, rating_pred, attention_weights = model(image, input_ids, attention_mask)
            _, predicted = torch.max(sentiment_logits, 1)
        
        predicted = predicted.item()
        pred_rating = rating_pred.item()
        
        # 创建包装模型
        wrapped_model = ModelWrapper(model, image, attention_mask)
        
        # 创建IntegratedGradients实例
        ig = IntegratedGradients(wrapped_model)
        
        # 计算文本的归因度
        try:
            attributions_text = ig.attribute(
                input_ids,
                target=predicted,
                n_steps=20,  # 减少步骤以加快计算
                internal_batch_size=1,
                method='gausslegendre'
            )
            
            # 处理归因结果
            attributions_sum = attributions_text.sum(dim=2).squeeze(0)
            attributions_norm = attributions_sum / torch.norm(attributions_sum)
            attributions_np = attributions_norm.cpu().detach().numpy()
            
            # 获取词汇
            tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
            
            # 过滤掉特殊令牌
            valid_indices = [i for i, token in enumerate(tokens) if token not in ['[PAD]', '[CLS]', '[SEP]']]
            valid_tokens = [tokens[i] for i in valid_indices]
            valid_attributions = [attributions_np[i] for i in valid_indices]
            
            # 创建图形
            plt.figure(figsize=(10, 8))
            
            # 特征归因条形图
            plt.subplot(2, 1, 1)
            last_n = min(30, len(valid_tokens))
            plt.barh(range(last_n), valid_attributions[-last_n:])
            plt.yticks(range(last_n), valid_tokens[-last_n:])
            plt.title(f'Feature Attribution for Text (Sample {idx}, Pred: {predicted}, True: {sentiment})')
            plt.xlabel('Attribution Score')
            
            # 生成词云
            plt.subplot(2, 1, 2)
            # 将归因值映射到颜色和大小
            scores = {t: max(0, a) for t, a in zip(valid_tokens, valid_attributions)}
            # 确保scores不为空
            if not scores:
                scores = {"no_attributions": 1.0}
            wordcloud_plus = WordCloud(width=800, height=400, 
                                       background_color='white',
                                       stopwords=set(stopwords.words('english')),
                                       max_words=100).generate_from_frequencies(scores)
            plt.imshow(wordcloud_plus, interpolation='bilinear')
            plt.axis('off')
            plt.title('Word Cloud with Positive Attributions')
            
            plt.tight_layout()
            plt.savefig(f'results/attribution_sample_{idx}.png')
            plt.close()
            
        except Exception as e:
            print(f"Error computing attributions for sample {idx}: {e}")
            continue

# 可视化不同模态的模型性能
def compare_modalities(test_loader, device='cuda'):
    # 加载完整模型
    full_model = MultiModalSentimentModel(num_classes=2)
    full_model.load_state_dict(torch.load('models/best_multimodal_model.pth', map_location=device))
    full_model = full_model.to(device)
    
    # 创建仅图像模型
    image_model = MultiModalSentimentModel(num_classes=2)
    image_model.load_state_dict(torch.load('models/best_multimodal_model.pth', map_location=device))
    image_model = image_model.to(device)  # 确保模型在正确的设备上
    
    # 让文本特征变为0
    def forward_image_only(self, image, input_ids, attention_mask):
        # 正常处理图像
        img_features = self.image_encoder(image)
        
        # 文本特征设为0
        batch_size = img_features.shape[0]
        text_features = torch.zeros(batch_size, 512, device=image.device)  # 明确指定设备
        
        # 特征拼接
        combined_features = torch.cat([img_features, text_features], dim=1)
        
        # 固定注意力权重
        attention_weights = torch.tensor([[1.0, 0.0]], device=image.device).repeat(batch_size, 1)  # 明确指定设备
        
        # 加权融合
        img_attended = img_features * attention_weights[:, 0].unsqueeze(1)
        text_attended = text_features * attention_weights[:, 1].unsqueeze(1)
        multimodal_features = torch.cat([img_attended, text_attended], dim=1)
        
        # 融合特征处理
        fused_features = self.fusion(multimodal_features)
        
        # 多任务输出
        sentiment_logits = self.classifier(fused_features)
        rating_pred = self.regressor(fused_features)
        
        return sentiment_logits, rating_pred, attention_weights
    
    # 替换前向传播方法
    image_model.forward = forward_image_only.__get__(image_model)
    
    # 创建仅文本模型 - 同样的修改
    text_model = MultiModalSentimentModel(num_classes=2)
    text_model.load_state_dict(torch.load('models/best_multimodal_model.pth', map_location=device))
    text_model = text_model.to(device)  # 确保模型在正确的设备上
    
    # 让图像特征变为0
    def forward_text_only(self, image, input_ids, attention_mask):
        # 图像特征设为0
        batch_size = input_ids.shape[0]
        img_features = torch.zeros(batch_size, 512, device=image.device)  # 明确指定设备
        
        # 正常处理文本
        text_output = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_features = self.text_projector(text_output.pooler_output)
        
        # 特征拼接
        combined_features = torch.cat([img_features, text_features], dim=1)
        
        # 固定注意力权重
        attention_weights = torch.tensor([[0.0, 1.0]], device=image.device).repeat(batch_size, 1)  # 明确指定设备
        
        # 加权融合
        img_attended = img_features * attention_weights[:, 0].unsqueeze(1)
        text_attended = text_features * attention_weights[:, 1].unsqueeze(1)
        multimodal_features = torch.cat([img_attended, text_attended], dim=1)
        
        # 融合特征处理
        fused_features = self.fusion(multimodal_features)
        
        # 多任务输出
        sentiment_logits = self.classifier(fused_features)
        rating_pred = self.regressor(fused_features)
        
        return sentiment_logits, rating_pred, attention_weights
    
    # 替换前向传播方法
    text_model.forward = forward_text_only.__get__(text_model)
    
    # 评估每个模型
    full_results = evaluate_model(full_model, test_loader, device)
    image_results = evaluate_model(image_model, test_loader, device)
    text_results = evaluate_model(text_model, test_loader, device)
    
    # 比较不同模态的性能
    modalities = ['Multimodal', 'Image Only', 'Text Only']
    accuracies = [
        full_results['classification_report'].loc['accuracy', 'precision'],
        image_results['classification_report'].loc['accuracy', 'precision'],
        text_results['classification_report'].loc['accuracy', 'precision']
    ]
    
    f1_scores = [
        full_results['classification_report'].loc['1', 'f1-score'],  # 正面情感的F1
        image_results['classification_report'].loc['1', 'f1-score'],
        text_results['classification_report'].loc['1', 'f1-score']
    ]
    
    maes = [
        full_results['mae'],
        image_results['mae'],
        text_results['mae']
    ]
    
    # 绘制比较图
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.bar(modalities, accuracies, color=['blue', 'green', 'orange'])
    plt.title('Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.01, f'{v:.4f}', ha='center')
    
    plt.subplot(1, 3, 2)
    plt.bar(modalities, f1_scores, color=['blue', 'green', 'orange'])
    plt.title('F1 Score Comparison (Positive Sentiment)')
    plt.ylabel('F1 Score')
    plt.ylim(0, 1)
    for i, v in enumerate(f1_scores):
        plt.text(i, v + 0.01, f'{v:.4f}', ha='center')
    
    plt.subplot(1, 3, 3)
    plt.bar(modalities, maes, color=['blue', 'green', 'orange'])
    plt.title('MAE Comparison (Rating Prediction)')
    plt.ylabel('Mean Absolute Error')
    for i, v in enumerate(maes):
        plt.text(i, v + 0.01, f'{v:.4f}', ha='center')
    
    plt.tight_layout()
    plt.savefig('results/modality_comparison.png')
    plt.close()
    
    return {
        'multimodal': full_results,
        'image_only': image_results,
        'text_only': text_results
    }

# 主函数
def main(args):
    # 检查GPU
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}")
    
    # 创建结果目录
    os.makedirs('results', exist_ok=True)
    
    # 加载测试数据集
    test_dataset = YelpReviewDataset(
        csv_file=os.path.join(args.data_dir, 'test.csv'),
        img_dir=os.path.join(args.data_dir, 'images'),
        transform=data_transforms
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4)
    
    # 加载模型
    model = MultiModalSentimentModel(num_classes=2)
    model.load_state_dict(torch.load('models/best_multimodal_model.pth', map_location=device))
    model = model.to(device)
    
    # 评估模型
    results = evaluate_model(model, test_loader, device)
    
    # 打印评估结果
    print("\nClassification Report:")
    print(results['classification_report'])
    print(f"\nMAE for Rating Prediction: {results['mae']:.4f}")
    print(f"Average Attention Weights: Image={results['avg_attention'][0]:.4f}, Text={results['avg_attention'][1]:.4f}")
    
    # 可视化结果
    plot_confusion_matrix(results['confusion_matrix'])
    plot_attention_weights(results['avg_attention'])
    
    # 比较不同模态的性能
    modality_results = compare_modalities(test_loader, device)
    
    # 特征归因可视化
    if args.visualize_attributions:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        visualize_attributions(model, test_dataset, tokenizer, device, num_samples=args.num_samples)
    
    # 保存结果
    results['classification_report'].to_csv('results/classification_report.csv')
    
    print("\nEvaluation complete. Results saved to 'results' directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='评估多模态情感分析模型')
    parser.add_argument('--data_dir', type=str, default='data/yelp',
                        help='数据目录')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批量大小')
    parser.add_argument('--cpu', action='store_true',
                        help='强制使用CPU')
    parser.add_argument('--visualize_attributions', action='store_true',
                        help='是否可视化特征归因')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='特征归因可视化的样本数量')
    
    args = parser.parse_args()
    main(args)
