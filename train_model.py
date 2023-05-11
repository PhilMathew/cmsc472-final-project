from argparse import ArgumentParser

from pathlib import Path
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
import torch
from torch import nn, optim
from torch.nn import functional as F

from dataset import ECGDataset, RandomLengthECGDataset
from models import ViTClassifier
from plotting import plot_training_metrics, plot_confmat


def get_device(use_cuda):
    if use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    
    return device


def make_dataloaders(train_csv, val_csv, test_csv, label_column, batch_size, patch_size, randomize_training_lengths):
    if randomize_training_lengths:
        train_dataset = RandomLengthECGDataset(str(train_csv), label_column=label_column, patch_size=patch_size)
    else:
        train_dataset = ECGDataset(str(train_csv), label_column=label_column)
    # TODO Maybe make train and test use variable signal lengths?
    val_dataset = ECGDataset(str(val_csv), label_column=label_column)
    test_dataset = ECGDataset(str(test_csv), label_column=label_column)
    
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


def train_model(model, train_dataloader, val_dataloader, num_epochs, lr, momentum, use_cuda, use_random_lengths, save_dir):
    device = get_device(use_cuda)
    if device.type == 'cuda':
        print('Training with CUDA-enabled device')
    else:
        print('Training on CPU')
    
    model = model.to(device)
    
    # Create optimizer and loss function
    # TODO: Try BCELoss
    loss_fn = nn.CrossEntropyLoss()
    if momentum:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)
    
    history = {k: [] for k in ('train_loss', 'train_acc', 'val_loss', 'val_acc')}
    best_val_loss = np.inf
    for ep in range(num_epochs):
        # Training phase
        running_train_loss, running_train_acc, num_train_batches = 0, 0, 0
        pbar = tqdm(train_dataloader, desc=f'Epoch {ep + 1}')
        for i, data in enumerate(pbar):
            optimizer.zero_grad()
            
            if use_random_lengths:
                inputs, labels, mask = data
                inputs, labels, mask = inputs.to(device), labels.to(device), mask.to(device)
                outputs = model(inputs.float(), mask=mask.bool())
            else:    
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
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
        
        if (running_val_loss / num_val_batches) < best_val_loss: # save best model
            print('Saving new best model weights!')
            torch.save(model.state_dict(), str(save_dir / 'best_model_weights.pth'))
            best_val_loss = running_val_loss / num_val_batches
                
    return history, save_dir / 'best_model_weights.pth'    


def test_model(model, model_weights_path, test_dataloader, use_cuda, save_dir):
    device = get_device(use_cuda)
    if device.type == 'cuda':
        print('Testing with CUDA-enabled device')
    else:
        print('Testing on CPU')
    
    
    model = model.to(device)
    model.load_state_dict(torch.load(str(model_weights_path)))
    
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
    test_acc = 100 * (np.sum(pred_labels == gt_labels) / gt_labels.shape[0])
    print(f'Testing Accuracy: {test_acc:.4f}%')
    plot_confmat(gt_labels, pred_labels, title=f'Test Accuracy: {test_acc:.4f}%', save_path=str(save_dir / 'confmat.png'))
    

def main():
    parser = ArgumentParser(description='ViT Training Script')
    parser.add_argument('--train_csv', dest='train_csv', help='Path to CSV with training data')
    parser.add_argument('--val_csv', dest='val_csv', help='Path to CSV with validation data')
    parser.add_argument('--test_csv', dest='test_csv', help='Path to CSV with testing data')
    parser.add_argument('-o', '--output_dir', dest='output_dir', default='training_results', help='Path to directory to output training results to')
    parser.add_argument('--randomize_training_signal_lengths', dest='randomize_training_lengths', action='store_true', default=False, help="Enables training with signals of varying length")
    parser.add_argument('--use_cuda', dest='use_cuda', action='store_true', default=False, help='Enables CUDA usage for computations')
    parser.add_argument('--dx_label', dest='label_column', type=str, default='SB', help='Diagnosis code for disease of interest in ConditionNames_SNOMED-CT.csv')
    parser.add_argument('--num_signals', dest='num_signals', type=int, default=12, help='Number of signals in each .mat file')
    parser.add_argument('--max_seq_length', dest='seq_length', type=int, default=5000, help='Maximum number of samples in each sequence')
    parser.add_argument('--patch_size', dest='patch_size', type=int, default=20, help='Size of convolution patches used by the ViT')
    parser.add_argument('--hidden_size', dest='hidden_size', type=int, default=256, help='Embedding dimension size for ViT')
    parser.add_argument('--encoder_depth', dest='encoder_depth', type=int, default=12, help='Number of blocks in ViT encoder')
    parser.add_argument('--num_classes', dest='num_classes', type=int, default=2, help='Number of classes to predict')
    parser.add_argument('--num_epochs', dest='num_epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=16, help='Size of training, validation, and test batches')
    parser.add_argument('--learning_rate', dest='lr', type=float, default=1e-3, help='Optimizer learning rate')
    parser.add_argument('--momentum', dest='momentum', type=float, default=None, help='Momentum parameter for SGD (uses ADAM optimizer if None)')
    args = parser.parse_args()
    
    # Make paths into pathlib objects
    train_csv, val_csv, test_csv = Path(args.train_csv), Path(args.val_csv), Path(args.test_csv)
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        output_dir.mkdir()
    
    # Set up model and dataloaders
    train_dataloader, val_dataloader, test_dataloader = make_dataloaders(train_csv=train_csv,
                                                                         val_csv=val_csv,
                                                                         test_csv=test_csv,
                                                                         label_column=args.label_column, 
                                                                         batch_size=args.batch_size, 
                                                                         patch_size=args.patch_size,
                                                                         randomize_training_lengths=args.randomize_training_lengths)
    
    # Create the model
    model = build_model(num_signals=args.num_signals,
                        patch_size=args.patch_size,
                        hidden_size=args.hidden_size,
                        seq_length=args.seq_length,
                        depth=args.encoder_depth,
                        n_classes=args.num_classes)
    
    # Train the model
    history, best_weights = train_model(model,
                                        train_dataloader,
                                        val_dataloader,
                                        num_epochs=args.num_epochs,
                                        lr=args.lr,
                                        momentum=args.momentum,
                                        use_cuda=args.use_cuda,
                                        use_random_lengths=args.randomize_training_lengths,
                                        save_dir=output_dir)
    
    # Plot the training metrics
    plot_training_metrics(history, save_path=str(output_dir / 'training_metrics.png'))
    
    # Evaluate on test set
    test_model(model,
               best_weights, 
               test_dataloader, 
               save_dir=output_dir,
               use_cuda=args.use_cuda)


if __name__ == '__main__':
    main()
    