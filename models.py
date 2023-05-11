import torch
from torch import nn
from transformer_components import ViTEmbedding, ViTEncoder, ClassificationHead

# TODO: Enable masking with a pad token
class ViTClassifier(nn.Module):
    def __init__(self, in_channels=12, patch_size=20, hidden_size=768, seq_length=5000, depth=12, n_classes=2, **kwargs):
        super(ViTClassifier, self).__init__()
        
        self.patch_embedding = ViTEmbedding(in_channels, seq_length, patch_size, hidden_size)
        self.encoder = ViTEncoder(depth, hidden_size=hidden_size, **kwargs)
        self.clf_head = ClassificationHead(hidden_size, n_classes)
        
    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.encoder(x)
        x = self.clf_head(x)

        return x


class ECG_CNN(nn.Module):
    def __init__(self, num_signals):
        super(ECG_CNN, self).__init__()
        self.conv1 = nn.Conv1d(num_signals, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(256)
        self.conv4 = nn.Conv1d(256, 512, kernel_size=5, padding=2)
        self.bn4 = nn.BatchNorm1d(512)
        self.conv5 = nn.Conv1d(512, 1024, kernel_size=5, padding=2)
        self.bn5 = nn.BatchNorm1d(1024)

        # Pooling and dropout layers
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(p=0.2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(p=0.2)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout(p=0.2)
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout4 = nn.Dropout(p=0.2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(1024, 512)
        self.bn6 = nn.BatchNorm1d(512)
        self.dropout5 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dropout6 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(256, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.functional.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.functional.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = nn.functional.relu(x)
        x = self.pool3(x)
        x = self.dropout3(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = nn.functional.relu(x)
        x = self.pool4(x)
        x = self.dropout4(x)
        
        x = self.conv5(x)
        x = self.bn5(x)
        x = nn.functional.relu(x)
        x = nn.functional.avg_pool1d(x, kernel_size=x.shape[-1])
        x = torch.flatten(x, 1)
        
        x = self.fc1(x)
        x = self.bn6(x)
        x = nn.functional.relu(x)
        x = self.dropout5(x)
        
        x = self.fc2(x)
        x = self.bn7(x)
        x = nn.functional.relu(x)
        x = self.dropout6(x)
        
        x = self.fc3(x)
        
        return x

