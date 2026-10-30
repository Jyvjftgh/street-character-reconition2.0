import os
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import timm
from config import config


class SwinTransformerModel(nn.Module):
    def __init__(self, class_num=11, pretrained=True, pretrained_path=None):
        super(SwinTransformerModel, self).__init__()

        if pretrained and pretrained_path and os.path.exists(pretrained_path):
            try:
                print(f"Loading pretrained weights from {pretrained_path}")
                self.backbone = timm.create_model(
                    config.swin_model,
                    pretrained=False,
                    num_classes=0,
                )
                state_dict = t.load(pretrained_path, map_location='cpu')
                state_dict = {k: v for k, v in state_dict.items() if not k.startswith('head.')}
                self.backbone.load_state_dict(state_dict, strict=False)
                print(f"Successfully loaded pretrained weights from {pretrained_path}")
            except Exception as e:
                print(f"Error loading pretrained weights: {e}")
                print("Falling back to non-pretrained model")
                self.backbone = timm.create_model(
                    config.swin_model,
                    pretrained=False,
                    num_classes=0,
                )
        else:
            print("Training from scratch without pretrained weights")
            self.backbone = timm.create_model(
                config.swin_model,
                pretrained=False,
                num_classes=0,
            )

        feat_dim = self.backbone.num_features

        self.attn1 = nn.Sequential(
            nn.Linear(feat_dim, 1024), nn.ReLU(), nn.Dropout(config.dropout_rate),
            nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(config.dropout_rate),
            nn.Linear(512, 1), nn.Sigmoid()
        )
        self.attn2 = nn.Sequential(
            nn.Linear(feat_dim, 1024), nn.ReLU(), nn.Dropout(config.dropout_rate),
            nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(config.dropout_rate),
            nn.Linear(512, 1), nn.Sigmoid()
        )
        self.attn3 = nn.Sequential(
            nn.Linear(feat_dim, 1024), nn.ReLU(), nn.Dropout(config.dropout_rate),
            nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(config.dropout_rate),
            nn.Linear(512, 1), nn.Sigmoid()
        )
        self.attn4 = nn.Sequential(
            nn.Linear(feat_dim, 1024), nn.ReLU(), nn.Dropout(config.dropout_rate),
            nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(config.dropout_rate),
            nn.Linear(512, 1), nn.Sigmoid()
        )

        self.fc1 = nn.Sequential(
            nn.Linear(feat_dim, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Dropout(config.dropout_rate), nn.Linear(512, class_num)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(feat_dim, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Dropout(config.dropout_rate), nn.Linear(512, class_num)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(feat_dim, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Dropout(config.dropout_rate), nn.Linear(512, class_num)
        )
        self.fc4 = nn.Sequential(
            nn.Linear(feat_dim, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Dropout(config.dropout_rate), nn.Linear(512, class_num)
        )

    def forward(self, x):
        features = self.backbone(x)
        a1 = self.attn1(features)
        a2 = self.attn2(features)
        a3 = self.attn3(features)
        a4 = self.attn4(features)
        feat1 = features * a1
        feat2 = features * a2
        feat3 = features * a3
        feat4 = features * a4
        fc1 = self.fc1(feat1)
        fc2 = self.fc2(feat2)
        fc3 = self.fc3(feat3)
        fc4 = self.fc4(feat4)
        return fc1, fc2, fc3, fc4


# Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, size_average=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average
        if alpha is not None:
            self.alpha_tensor = t.tensor(alpha, dtype=t.float32)
        else:
            self.alpha_tensor = None

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = t.exp(-ce_loss)
        if self.alpha_tensor is not None:
            alpha = self.alpha_tensor.to(inputs.device)
            alpha = alpha[targets]
        else:
            alpha = 1.0
        focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()


class EnhancedLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, smoothing=0.1, class_weights=None):
        super(EnhancedLoss, self).__init__()
        self.focal = FocalLoss(alpha, gamma)
        self.smoothing = smoothing
        self.class_weights = class_weights

    def forward(self, inputs, targets):
        focal_loss = self.focal(inputs, targets)
        log_probs = F.log_softmax(inputs, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)
        if self.class_weights is not None:
            weights = self.class_weights[targets]
            nll_loss = nll_loss * weights.to(nll_loss.device)
        smoothed_loss = -log_probs.mean(dim=-1)
        loss = (1 - self.smoothing) * nll_loss + self.smoothing * smoothed_loss
        return focal_loss + loss.mean()
