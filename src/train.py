import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import joblib
from tqdm import tqdm
import time

from utils import load_and_prepare_data, collate_fn
from model import get_model

def create_results_dir():
    os.makedirs('../results', exist_ok=True)
    os.makedirs('../results/plots', exist_ok=True)
    os.makedirs('../results/models', exist_ok=True)

def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(loader, desc=f'Epoch {epoch} [Train]')
    for batch in pbar:
        seqs, labels, mask = batch
        seqs = seqs.to(device)
        labels = labels.to(device)
        mask = mask.to(device)
        
        logits = model(seqs, src_key_padding_mask=mask)
        loss = criterion(logits, labels)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        
        total_loss += loss.item() * seqs.size(0)
        
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return avg_loss, acc, f1

def validate(model, loader, criterion, device, epoch):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(loader, desc=f'Epoch {epoch} [Val]')
    with torch.no_grad():
        for batch in pbar:
            seqs, labels, mask = batch
            seqs = seqs.to(device)
            labels = labels.to(device)
            mask = mask.to(device)
            
            logits = model(seqs, src_key_padding_mask=mask)
            loss = criterion(logits, labels)
            
            total_loss += loss.item() * seqs.size(0)
            
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return avg_loss, acc, f1, all_preds, all_labels

def plot_training_curves(train_losses, val_losses, train_accs, val_accs, train_f1s, val_f1s, model_name):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    axes[0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    axes[0].set_title('Training and Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(epochs, train_accs, 'b-', label='Train Acc', linewidth=2)
    axes[1].plot(epochs, val_accs, 'r-', label='Val Acc', linewidth=2)
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    axes[2].plot(epochs, train_f1s, 'b-', label='Train F1', linewidth=2)
    axes[2].plot(epochs, val_f1s, 'r-', label='Val F1', linewidth=2)
    axes[2].set_title('Training and Validation F1-Score')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('F1-Score')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(f'../results/plots/{model_name}_training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names, model_name):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    plt.savefig(f'../results/plots/{model_name}_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return cm

def save_classification_report(y_true, y_pred, class_names, model_name, metrics):
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    
    with open(f'../results/{model_name}_classification_report.txt', 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Best Validation Accuracy: {metrics['best_val_acc']:.4f}\n")
        f.write(f"Best Validation F1-Score: {metrics['best_val_f1']:.4f}\n")
        f.write(f"Final Validation Accuracy: {metrics['final_val_acc']:.4f}\n")
        f.write(f"Final Validation F1-Score: {metrics['final_val_f1']:.4f}\n\n")
        f.write("Detailed Classification Report:\n")
        f.write(report)
    
    print(f"Classification report saved to ../results/{model_name}_classification_report.txt")
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()



def train_model(model_name='cnn_transformer', num_epochs=50, batch_size=64, learning_rate=5e-4):
    print(f"{'='*60}")
    print(f"TRAINING {model_name.upper()} MODEL")
    print(f"{'='*60}")
    
    create_results_dir()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading data...")
    data = load_and_prepare_data()
    
    # Create data loaders
    # train_loader = DataLoader(data['train_dataset'], batch_size=batch_size, shuffle=True, 
    #                          collate_fn=collate_fn, drop_last=True, num_workers=2)
    # val_loader = DataLoader(data['val_dataset'], batch_size=batch_size, shuffle=False, 
    #                        collate_fn=collate_fn, num_workers=0)
    train_loader = DataLoader(data['train_dataset'], batch_size=batch_size, shuffle=True, 
                         collate_fn=collate_fn, drop_last=True, num_workers=2,
                         pin_memory=True if torch.cuda.is_available() else False)
    val_loader = DataLoader(data['val_dataset'], batch_size=batch_size, shuffle=False, 
                       collate_fn=collate_fn, num_workers=2,
                       pin_memory=True if torch.cuda.is_available() else False)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    print(f"Creating {model_name} model...")
    model = get_model(model_name, data['num_features'], data['num_classes'], 
                     hidden_dim=256, n_heads=8, n_layers=2, dropout=0.4).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Add label smoothing
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4, 
                       betas=(0.9, 0.999))  # Better optimizer
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)  # Better scheduler
    early_stopping = EarlyStopping(patience=7, min_delta=0.001)
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    train_f1s, val_f1s = [], []
    learning_rates = []

    
    best_val_f1 = 0.0
    best_val_acc = 0.0
    best_epoch = 0
    
    print(f"\nStarting training for {num_epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(1, num_epochs + 1):
        # Train
        train_loss, train_acc, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validate
        val_loss, val_acc, val_f1, val_preds, val_labels = validate(model, val_loader, criterion, device, epoch)
        
        # Update scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        # Track metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        train_f1s.append(train_f1)
        val_f1s.append(val_f1)
        
        print(f"Epoch {epoch}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"  Val   Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
        print(f"  LR: {current_lr:.6f}")

        
        if val_f1 > best_val_f1 and (train_f1 - val_f1) < 0.20:
            best_val_f1 = val_f1
            best_val_acc = val_acc
            best_epoch = epoch
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_f1': val_f1,
                'val_acc': val_acc,
                'encoders': data['encoders'],
                'scaler': data['scaler']
            }, f'../results/models/{model_name}_best_model.pth')
            
            print(f"  -> New best model saved! (F1: {val_f1:.4f})")
        
        print("-" * 60)
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time/3600:.2f} hours")
    print(f"Best model: Epoch {best_epoch}, Val F1: {best_val_f1:.4f}, Val Acc: {best_val_acc:.4f}")
    
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_f1': val_f1,
        'val_acc': val_acc,
        'encoders': data['encoders'],
        'scaler': data['scaler']
    }, f'../results/models/{model_name}_final_model.pth')
    
    print("\nGenerating training curves...")
    plot_training_curves(train_losses, val_losses, train_accs, val_accs, 
                         train_f1s, val_f1s, model_name)
    
    print("Loading best model for final evaluation...")
    checkpoint = torch.load(f'../results/models/{model_name}_best_model.pth', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    final_val_loss, final_val_acc, final_val_f1, final_preds, final_labels = validate(
        model, val_loader, criterion, device, 'Final')
    
    class_names = data['encoders']['gesture'].classes_
    
    print("Generating confusion matrix...")
    plot_confusion_matrix(final_labels, final_preds, class_names, model_name)
    
    metrics = {
        'best_val_acc': best_val_acc,
        'best_val_f1': best_val_f1,
        'final_val_acc': final_val_acc,
        'final_val_f1': final_val_f1
    }
    save_classification_report(final_labels, final_preds, class_names, model_name, metrics)
    
    history_df = pd.DataFrame({
        'epoch': range(1, num_epochs + 1),
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_acc': train_accs,
        'val_acc': val_accs,
        'train_f1': train_f1s,
        'val_f1': val_f1s
    })
    history_df.to_csv(f'../results/{model_name}_training_history.csv', index=False)
    
    print(f"\nTraining complete!")
    print(f"Best validation F1-Score: {best_val_f1:.4f}")
    print(f"Final validation F1-Score: {final_val_f1:.4f}")
    print(f"Results saved in ../results/")
    
    return {
        'model': model,
        'best_val_f1': best_val_f1,
        'final_val_f1': final_val_f1,
        'history': history_df,
        'encoders': data['encoders'],
        'scaler': data['scaler']
    }

def compare_models():
    """Train and compare different model architectures"""
    print("="*60)
    print("COMPARING DIFFERENT MODEL ARCHITECTURES")
    print("="*60)
    
    models_to_test = [
        ('cnn_transformer', 'CNN-Transformer'),
        ('rnn', 'BiLSTM')
    ]
    
    results = {}
    
    for model_type, model_display_name in models_to_test:
        print(f"\nTraining {model_display_name}...")
        result = train_model(model_name=model_type, num_epochs=25, batch_size=32)
        results[model_display_name] = {
            'best_val_f1': result['best_val_f1'],
            'final_val_f1': result['final_val_f1']
        }
    
    print("\n" + "="*60)
    print("MODEL COMPARISON RESULTS")
    print("="*60)
    
    comparison_df = pd.DataFrame(results).T
    comparison_df.columns = ['Best Val F1', 'Final Val F1']
    print(comparison_df.round(4))
    
    comparison_df.to_csv('../results/model_comparison.csv')
    
    plt.figure(figsize=(10, 6))
    x = range(len(comparison_df))
    width = 0.35
    
    plt.bar([i - width/2 for i in x], comparison_df['Best Val F1'], width, 
            label='Best Val F1', alpha=0.8)
    plt.bar([i + width/2 for i in x], comparison_df['Final Val F1'], width, 
            label='Final Val F1', alpha=0.8)
    
    plt.xlabel('Model')
    plt.ylabel('F1-Score')
    plt.title('Model Comparison - F1 Scores')
    plt.xticks(x, comparison_df.index, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig('../results/plots/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results

if __name__ == "__main__":
    print("Starting single model training...")
    result = train_model(model_name='cnn_transformer', num_epochs=50, batch_size=64, learning_rate=5e-4)
    
    print("\nStarting model comparison...")
    comparison_results = compare_models()