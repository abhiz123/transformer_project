import torch
import torchtext; torchtext.disable_torchtext_deprecation_warning()
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torchtext.data.metrics import bleu_score
import math
import time
from torch.nn.utils.rnn import pad_sequence

class TransformerModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout=0.1):
        super(TransformerModel, self).__init__()
        
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        
        self.d_model = d_model

    def forward(self, src, tgt):
        src_embed = self.positional_encoding(self.src_embedding(src) * math.sqrt(self.d_model))
        tgt_embed = self.positional_encoding(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
        
        src_mask = self.generate_square_subsequent_mask(src.size(1)).to(src.device)
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        
        output = self.transformer(src_embed.transpose(0, 1), tgt_embed.transpose(0, 1), 
                                  src_mask=src_mask, tgt_mask=tgt_mask)
        return self.fc_out(output.transpose(0, 1))

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class BaselineModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, hidden_size):
        super(BaselineModel, self).__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, hidden_size)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, hidden_size)
        self.encoder = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.attention = nn.Linear(hidden_size * 2, 1)
        self.fc_out = nn.Linear(hidden_size * 2, tgt_vocab_size)

    def forward(self, src, tgt):
        src_embed = self.src_embedding(src)
        tgt_embed = self.tgt_embedding(tgt)
        
        encoder_outputs, (h_n, c_n) = self.encoder(src_embed)
        decoder_outputs, _ = self.decoder(tgt_embed, (h_n, c_n))
        
        # Attention mechanism
        attn_weights = []
        for i in range(decoder_outputs.size(1)):
            attn_input = torch.cat([decoder_outputs[:, i:i+1].repeat(1, encoder_outputs.size(1), 1), 
                                    encoder_outputs], dim=2)
            attn_weights.append(self.attention(attn_input).squeeze(-1))
        attn_weights = torch.stack(attn_weights, dim=1)
        attn_weights = torch.softmax(attn_weights, dim=2)
        
        context = torch.bmm(attn_weights, encoder_outputs)
        
        output = self.fc_out(torch.cat([decoder_outputs, context], dim=2))
        return output

class TranslationDataset(Dataset):
    def __init__(self, src_sentences, tgt_sentences, src_vocab, tgt_vocab):
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, idx):
        src = torch.tensor([self.src_vocab.get(word, self.src_vocab['<unk>']) for word in self.src_sentences[idx].split()])
        tgt = torch.tensor([self.tgt_vocab.get(word, self.tgt_vocab['<unk>']) for word in self.tgt_sentences[idx].split()])
        return src, tgt

def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_item, tgt_item in batch:
        src_batch.append(src_item)
        tgt_batch.append(tgt_item)
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=0)
    return src_batch, tgt_batch

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for src, tgt in train_loader:
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad()
        output = model(src, tgt[:, :-1])
        loss = criterion(output.view(-1, output.size(-1)), tgt[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for src, tgt in val_loader:
            src, tgt = src.to(device), tgt.to(device)
            output = model(src, tgt[:, :-1])
            loss = criterion(output.view(-1, output.size(-1)), tgt[:, 1:].reshape(-1))
            total_loss += loss.item()
    return total_loss / len(val_loader)

def calculate_bleu(model, test_loader, tgt_vocab, device):
    model.eval()
    bleu_scores = []
    idx_to_word = {idx: word for word, idx in tgt_vocab.items()}
    with torch.no_grad():
        for src, tgt in test_loader:
            src = src.to(device)
            output = model(src, torch.zeros_like(tgt).to(device))
            predicted = output.argmax(dim=2)
            for pred, target in zip(predicted, tgt):
                pred_sentence = ' '.join([idx_to_word.get(idx.item(), '<unk>') for idx in pred if idx != 0])
                target_sentence = ' '.join([idx_to_word.get(idx.item(), '<unk>') for idx in target if idx != 0])
                bleu_scores.append(bleu_score([pred_sentence.split()], [[target_sentence.split()]]))
    
    if not bleu_scores:
        return 0.0
    return sum(bleu_scores) / len(bleu_scores)

def visualize_attention(model, src_sentence, tgt_sentence, src_vocab, tgt_vocab, device):
    model.eval()
    src = torch.tensor([[src_vocab.get(word, src_vocab['<unk>']) for word in src_sentence.split()]]).to(device)
    tgt = torch.tensor([[tgt_vocab.get(word, tgt_vocab['<unk>']) for word in tgt_sentence.split()]]).to(device)
    
    src_mask = model.generate_square_subsequent_mask(src.size(1)).to(device)
    tgt_mask = model.generate_square_subsequent_mask(tgt.size(1)).to(device)
    
    src_padding_mask = (src == src_vocab['<pad>']).to(device)
    tgt_padding_mask = (tgt == tgt_vocab['<pad>']).to(device)
    
    src_embed = model.positional_encoding(model.src_embedding(src) * math.sqrt(model.d_model))
    tgt_embed = model.positional_encoding(model.tgt_embedding(tgt) * math.sqrt(model.d_model))
    
    with torch.no_grad():
        memory = model.transformer.encoder(src_embed.transpose(0, 1), mask=src_mask, src_key_padding_mask=src_padding_mask)
        
        # Get encoder self-attention weights
        encoder_layer = model.transformer.encoder.layers[-1]
        encoder_attention = encoder_layer.self_attn
        q = k = src_embed.transpose(0, 1)
        attn_output, attn_output_weights = encoder_attention(q, k, q)
        
        # Get decoder-encoder attention weights
        outs = model.transformer.decoder(tgt_embed.transpose(0, 1), memory, tgt_mask=tgt_mask, 
                                         memory_mask=None,
                                         tgt_key_padding_mask=tgt_padding_mask,
                                         memory_key_padding_mask=src_padding_mask)
        decoder_layer = model.transformer.decoder.layers[-1]
        decoder_attention = decoder_layer.multihead_attn
        q = outs
        k = v = memory
        dec_attn_output, dec_attn_output_weights = decoder_attention(q, k, v)
    
    # Visualize encoder self-attention
    plt.figure(figsize=(10, 8))
    plt.imshow(attn_output_weights[0].cpu().numpy(), cmap='viridis', aspect='auto')
    plt.xlabel('Source words')
    plt.ylabel('Source words')
    plt.title('Encoder Self-Attention weights')
    plt.xticks(range(len(src_sentence.split())), src_sentence.split(), rotation=90)
    plt.yticks(range(len(src_sentence.split())), src_sentence.split())
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('encoder_self_attention_visualization.png')
    plt.close()

    print("Encoder self-attention visualization saved as 'encoder_self_attention_visualization.png'")

    # Visualize decoder-encoder attention
    plt.figure(figsize=(10, 8))
    plt.imshow(dec_attn_output_weights[0].cpu().numpy(), cmap='viridis', aspect='auto')
    plt.xlabel('Source words')
    plt.ylabel('Target words')
    plt.title('Decoder-Encoder Attention weights')
    plt.xticks(range(len(src_sentence.split())), src_sentence.split(), rotation=90)
    plt.yticks(range(len(tgt_sentence.split())), tgt_sentence.split())
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('decoder_encoder_attention_visualization.png')
    plt.close()

    print("Decoder-Encoder attention visualization saved as 'decoder_encoder_attention_visualization.png'")