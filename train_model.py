from argparse import ArgumentParser, BooleanOptionalAction

from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
import torch
from torch import nn, optim
from torch.nn import functional as F

from dataset import ECGDataset
from models import ECG_CNN, ViTClassifier
from plotting import plot_training_metrics, plot_confmat


def get_device(use_cuda):
    if use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    
    return device


def make_dataloaders(label_column, batch_size):
    train_dataset = ECGDataset('data_csvs/train.csv', label_column=label_column)
    val_dataset = ECGDataset('data_csvs/val.csv', label_column=label_column)
    test_dataset = ECGDataset('data_csvs/test.csv', label_column=label_column)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_dataloader, val_dataloader, test_dataloader


def build_model(num_signals, patch_size, hidden_size, seq_length, depth, n_classes):
    model = ViTClassifier(in_channels=num_signals,
                          patch_size=patch_size,
                          hidden_size=hidden_size,
                          seq_length=seq_length,
                          depth=depth,
                          n_classes=n_classes)
    
    return model    


def train_model(model, train_dataloader, val_dataloader, num_epochs, lr, momentum, use_cuda):
    device = get_device(use_cuda)
    if device.type == 'cuda':
        print('Training with CUDA-enabled device')
    else:
        print('Training on CPU')
    
    model = model.to(device)
    
    # Create optimizer and loss function
    loss_fn = nn.CrossEntropyLoss()
    if momentum:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)
    
    history = {k: [] for k in ('train_loss', 'train_acc', 'val_loss', 'val_acc')}
    for ep in range(num_epochs):
        # Training phase
        running_train_loss, running_train_acc, num_train_batches = 0, 0, 0
        pbar = tqdm(train_dataloader, desc=f'Epoch {ep + 1}')
        for i, data in enumerate(pbar):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs.float())
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item()
            running_train_acc += torch.sum(torch.argmax(F.softmax(outputs, dim=-1), dim=-1) == labels).item() / inputs.shape[0]
            num_train_batches += 1
            
            pbar.set_postfix_str(f'Train Loss: {running_train_loss / (i + 1):.4f}, Train Accuracy: {100 * (running_train_acc / (i + 1)):.4f}%')
        
        # Validation phase
        with torch.no_grad():
            running_val_loss, running_val_acc, num_val_batches = 0, 0, 0
            for j, data in enumerate(val_dataloader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs.float())
                loss = loss_fn(outputs, labels)
                
                running_val_loss += loss.item()
                running_val_acc += torch.sum(torch.argmax(F.softmax(outputs, dim=-1), dim=-1) == labels).item() / inputs.shape[0]
                num_val_batches += 1
        print(f'Val. Loss: {running_val_loss / (num_val_batches + 1):.4f}, Val. Accuracy: {100 * (running_val_acc / (num_val_batches + 1)):.4f}%')
        
        history['train_loss'].append(running_train_loss / num_train_batches)
        history['train_acc'].append(running_train_acc / num_train_batches)
        history['val_loss'].append(running_val_loss / num_val_batches)
        history['val_acc'].append(running_val_acc / num_val_batches)
                
    return history    


def test_model(model, test_dataloader, use_cuda):
    device = get_device(use_cuda)
    if device.type == 'cuda':
        print('Testing with CUDA-enabled device')
    else:
        print('Testing on CPU')
    
    model = model.to(device)
    
    with torch.no_grad():
        pred_labels, gt_labels = [], []
        for _, data in enumerate(tqdm(test_dataloader, desc="Testing Model")):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs.float())
            batch_preds = torch.argmax(F.softmax(outputs, dim=-1), dim=-1).cpu().numpy()
            pred_labels.extend(batch_preds)
            gt_labels.extend(labels.cpu().numpy())
    
    pred_labels, gt_labels = np.array(pred_labels), np.array(gt_labels)
    print(f'Testing Accuracy: {100 * (np.sum(pred_labels == gt_labels) / gt_labels.shape[0])}%')
    plot_confmat(gt_labels, pred_labels)
    

def main():
    parser = ArgumentParser(description='CNN Training Script')
    parser.add_argument('--label_column', dest='label_column', type=str, default='AFIB')
    parser.add_argument('--num_signals', dest='num_signals', type=int, default=12)
    parser.add_argument('--num_epochs', dest='num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', dest='lr', type=float, default=1e-3)
    parser.add_argument('--momentum', dest='momentum', type=float, default=None)
    parser.add_argument('--use_cuda', dest='use_cuda', action='store_true', default=False)
    args = parser.parse_args()
    
    # Set up model and dataloaders
    train_dataloader, val_dataloader, test_dataloader = make_dataloaders(label_column=args.label_column, batch_size=args.batch_size)
    model = build_model(args.num_signals,
                        patch_size=20,
                        hidden_size=768,
                        seq_length=5000,
                        depth=12,
                        n_classes=2)
    
    # Train the model
    history = train_model(model,
                          train_dataloader,
                          val_dataloader,
                          num_epochs=args.num_epochs,
                          lr=args.lr,
                          momentum=args.momentum,
                          use_cuda=args.use_cuda)
    
    # Plot the training metrics
    plot_training_metrics(history)
    
    # Evaluate on test set
    test_model(model, test_dataloader, use_cuda=args.use_cuda)
    


if __name__ == '__main__':
    main()
    