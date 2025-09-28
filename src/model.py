import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CNNTransformer(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=128, n_heads=4, n_layers=2, dropout=0.3):
        super(CNNTransformer, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, 
                               kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        self.conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, 
                               kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout, max_len=1000)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=n_heads, 
            dim_feedforward=hidden_dim*4, 
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Classification head
        # self.classifier = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim // 2),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(hidden_dim // 2, num_classes)
        # )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),  
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, x, src_key_padding_mask=None):

        batch_size, seq_len, input_dim = x.shape
        
        x = x.permute(0, 2, 1)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        new_seq_len = x.shape[2]
        
        x = x.permute(0, 2, 1)
        
        x = self.pos_encoder(x)
        
        if src_key_padding_mask is not None:
            new_mask = src_key_padding_mask[:, ::2]  
            if new_mask.shape[1] != new_seq_len:
                if new_mask.shape[1] > new_seq_len:
                    new_mask = new_mask[:, :new_seq_len]
                else:
                    padding_needed = new_seq_len - new_mask.shape[1]
                    new_mask = F.pad(new_mask, (0, padding_needed), value=True)
        else:
            new_mask = None
        
        x = self.transformer(x, src_key_padding_mask=new_mask)
        
        if new_mask is not None:
            mask_expanded = new_mask.unsqueeze(-1).expand_as(x)
            x_masked = x.masked_fill(mask_expanded, 0)
            seq_lengths = (~new_mask).sum(dim=1, keepdim=True).float()
            x = x_masked.sum(dim=1) / seq_lengths.clamp(min=1)
        else:
            x = x.mean(dim=1)
        
        logits = self.classifier(x)  
        return logits


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):

        batch_size, seq_len, d_model = x.shape
        x = x + self.pe[:seq_len, :].transpose(0, 1)
        return self.dropout(x)


class SimpleRNN(nn.Module):

    def __init__(self, input_dim, num_classes, hidden_dim=128, num_layers=2, dropout=0.3, **kwargs):
        super(SimpleRNN, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, 
                          batch_first=True, dropout=dropout, bidirectional=True)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x, src_key_padding_mask=None):
        
        rnn_out, (hidden, cell) = self.rnn(x)
        
   
        if src_key_padding_mask is not None:
            seq_lengths = (~src_key_padding_mask).sum(dim=1) - 1  
            batch_size = x.shape[0]
            last_outputs = []
            for i in range(batch_size):
                last_outputs.append(rnn_out[i, seq_lengths[i].item()])
            x = torch.stack(last_outputs)
        else:
            x = rnn_out[:, -1, :]  
        
        logits = self.classifier(x)
        return logits


def get_model(model_type, input_dim, num_classes, **kwargs):
    
    if model_type == 'cnn_transformer':
        return CNNTransformer(input_dim, num_classes, **kwargs)
    elif model_type == 'rnn':
        return SimpleRNN(input_dim, num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on device: {device}")
    
    batch_size = 4
    seq_len = 50
    input_dim = 100
    num_classes = 18
    
    x = torch.randn(batch_size, seq_len, input_dim).to(device)
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool).to(device)
    mask[:, -10:] = True 
    
    print("Testing CNN-Transformer...")
    model = CNNTransformer(input_dim, num_classes, hidden_dim=64).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    with torch.no_grad():
        output = model(x, mask)
        print(f"Output shape: {output.shape}")
        print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
    
    print("\nTesting Simple RNN...")
    rnn_model = SimpleRNN(input_dim, num_classes, hidden_dim=64).to(device)
    
    print(f"RNN parameters: {sum(p.numel() for p in rnn_model.parameters()):,}")
    
    with torch.no_grad():
        rnn_output = rnn_model(x, mask)
        print(f"RNN Output shape: {rnn_output.shape}")
        print(f"RNN Output range: [{rnn_output.min().item():.3f}, {rnn_output.max().item():.3f}]")
    
    print("Model tests completed successfully!")