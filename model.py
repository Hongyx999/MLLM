import torch
import torch.nn as nn
from transformers import BertModel
import torchvision.models as models

class MultiModalSentimentModel(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.3):
        super(MultiModalSentimentModel, self).__init__()
        
        # 图像编码器 (使用预训练的ResNet-50)
        self.image_encoder = models.resnet50(pretrained=True)
        # 替换最后的全连接层
        num_features = self.image_encoder.fc.in_features
        self.image_encoder.fc = nn.Linear(num_features, 512)
        
        # 文本编码器 (使用预训练的BERT)
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.text_projector = nn.Linear(768, 512)
        
        # 注意力融合机制
        self.attention = nn.Sequential(
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512, 2),  # 2表示两种模态
            nn.Softmax(dim=1)
        )
        
        # 多模态融合层
        self.fusion = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 输出层
        self.classifier = nn.Linear(128, num_classes)  # 情感分类
        self.regressor = nn.Linear(128, 1)  # 评分预测
        
    def forward(self, image, input_ids, attention_mask):
        # 1. 图像特征提取
        img_features = self.image_encoder(image)  # [batch_size, 512]
        
        # 2. 文本特征提取
        text_output = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_features = self.text_projector(text_output.pooler_output)  # [batch_size, 512]
        
        # 3. 特征拼接
        combined_features = torch.cat([img_features, text_features], dim=1)  # [batch_size, 1024]
        
        # 4. 注意力权重计算
        attention_weights = self.attention(combined_features)  # [batch_size, 2]
        
        # 5. 加权融合
        img_attended = img_features * attention_weights[:, 0].unsqueeze(1)
        text_attended = text_features * attention_weights[:, 1].unsqueeze(1)
        multimodal_features = torch.cat([img_attended, text_attended], dim=1)  # [batch_size, 1024]
        
        # 6. 融合特征处理
        fused_features = self.fusion(multimodal_features)  # [batch_size, 128]
        
        # 7. 多任务输出
        sentiment_logits = self.classifier(fused_features)  # [batch_size, num_classes]
        rating_pred = self.regressor(fused_features)  # [batch_size, 1]
        
        return sentiment_logits, rating_pred, attention_weights