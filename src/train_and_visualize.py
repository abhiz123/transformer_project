import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchtext.data.utils import get_tokenizer
from torchtext.utils import download_from_url
from collections import Counter
from transformer_implementation import TransformerModel, BaselineModel, train, evaluate, calculate_bleu, visualize_attention
import matplotlib.pyplot as plt
import time
import io
import warnings
import json

# Suppress warnings
warnings.filterwarnings("ignore")

# Disable deprecation warning
import torchtext
torchtext.disable_torchtext_deprecation_warning()

# Set random seed for reproducibility
torch.manual_seed(42)

# Define constants
BATCH_SIZE = 32
MAX_VOCAB_SIZE = 10000
MAX_SEQ_LENGTH = 50
NUM_EPOCHS = 1000

# Define dataset class
class TranslationDataset(Dataset):
    def __init__(self, data, src_vocab, tgt_vocab, max_seq_length):
        self.data = data
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src, tgt = self.data[idx]
        src = [self.src_vocab.get(word, self.src_vocab['<unk>']) for word in src.split()][:self.max_seq_length]
        tgt = [self.tgt_vocab.get(word, self.tgt_vocab['<unk>']) for word in tgt.split()][:self.max_seq_length]
        return torch.tensor(src), torch.tensor(tgt)

# Function to build vocabulary
def build_vocab(data, max_size):
    counter = Counter()
    for sentence in data:
        counter.update(sentence.split())
    vocab = {'<pad>': 0, '<unk>': 1, '<sos>': 2, '<eos>': 3}
    vocab.update({word: idx + 4 for idx, (word, _) in enumerate(counter.most_common(max_size - 4))})
    return vocab

def load_data():
    url_base = 'https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/'
    train_urls = (url_base + 'train.de', url_base + 'train.en')
    val_urls = (url_base + 'val.de', url_base + 'val.en')
    test_urls = (url_base + 'test_2016_flickr.de', url_base + 'test_2016_flickr.en')

    def download_and_read(url):
        path = download_from_url(url)
        with io.open(path, encoding='utf-8', errors='ignore') as f:
            return f.readlines()

    train_de, train_en = [download_and_read(url) for url in train_urls]
    val_de, val_en = [download_and_read(url) for url in val_urls]
    test_de, test_en = [download_and_read(url) for url in test_urls]

    train_data = list(zip(train_en[:5000], train_de[:5000]))  # Limit to 5000 sentences
    valid_data = list(zip(val_en, val_de))
    test_data = list(zip(test_en, test_de))

    src_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
    tgt_tokenizer = get_tokenizer('spacy', language='de_core_news_sm')

    def tokenize_data(data, tokenizer):
        return [' '.join(tokenizer(sentence.strip())) for sentence in data]

    src_data = tokenize_data(train_en[:5000], src_tokenizer)
    tgt_data = tokenize_data(train_de[:5000], tgt_tokenizer)

    src_vocab = build_vocab(src_data, MAX_VOCAB_SIZE)
    tgt_vocab = build_vocab(tgt_data, MAX_VOCAB_SIZE)

    train_dataset = TranslationDataset(train_data, src_vocab, tgt_vocab, MAX_SEQ_LENGTH)
    valid_dataset = TranslationDataset(valid_data, src_vocab, tgt_vocab, MAX_SEQ_LENGTH)
    test_dataset = TranslationDataset(test_data, src_vocab, tgt_vocab, MAX_SEQ_LENGTH)

    return train_dataset, valid_dataset, test_dataset, src_vocab, tgt_vocab

# Collate function for DataLoader
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_item, tgt_item in batch:
        src_batch.append(src_item)
        tgt_batch.append(tgt_item)
    src_batch = nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_batch = nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=0)
    return src_batch, tgt_batch

# Main function
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    train_dataset, valid_dataset, test_dataset, src_vocab, tgt_vocab = load_data()
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    # Initialize models
    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)
    d_model = 256
    nhead = 8
    num_encoder_layers = 3
    num_decoder_layers = 3
    dim_feedforward = 512
    dropout = 0.1

    transformer_model = TransformerModel(src_vocab_size, tgt_vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout).to(device)
    baseline_model = BaselineModel(src_vocab_size, tgt_vocab_size, d_model).to(device)

    # Define loss function and optimizers
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    transformer_optimizer = optim.Adam(transformer_model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    baseline_optimizer = optim.Adam(baseline_model.parameters(), lr=0.0001)

    # Training loop
    transformer_losses = []
    baseline_losses = []
    transformer_bleu_scores = []
    baseline_bleu_scores = []

    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        
        transformer_loss = train(transformer_model, train_loader, transformer_optimizer, criterion, device)
        baseline_loss = train(baseline_model, train_loader, baseline_optimizer, criterion, device)
        
        transformer_val_loss = evaluate(transformer_model, valid_loader, criterion, device)
        baseline_val_loss = evaluate(baseline_model, valid_loader, criterion, device)
        
        transformer_bleu = calculate_bleu(transformer_model, test_loader, tgt_vocab, device)
        baseline_bleu = calculate_bleu(baseline_model, test_loader, tgt_vocab, device)
        
        end_time = time.time()
        
        transformer_losses.append(transformer_val_loss)
        baseline_losses.append(baseline_val_loss)
        transformer_bleu_scores.append(transformer_bleu)
        baseline_bleu_scores.append(baseline_bleu)
        
        print(f'Epoch: {epoch+1}')
        print(f'Time: {end_time - start_time:.2f}s')
        print(f'Transformer - Train Loss: {transformer_loss:.4f}, Val Loss: {transformer_val_loss:.4f}, BLEU: {transformer_bleu:.4f}')
        print(f'Baseline - Train Loss: {baseline_loss:.4f}, Val Loss: {baseline_val_loss:.4f}, BLEU: {baseline_bleu:.4f}')
        print()

    # Visualize results
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(transformer_losses, label='Transformer')
    plt.plot(baseline_losses, label='Baseline')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.legend()
    plt.title('Validation Loss Comparison')

    plt.subplot(1, 2, 2)
    plt.plot(transformer_bleu_scores, label='Transformer')
    plt.plot(baseline_bleu_scores, label='Baseline')
    plt.xlabel('Epoch')
    plt.ylabel('BLEU Score')
    plt.legend()
    plt.title('BLEU Score Comparison')

    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.close()

    # Save model comparison data
    comparison_data = {
        'transformer_losses': transformer_losses,
        'baseline_losses': baseline_losses,
        'transformer_bleu_scores': transformer_bleu_scores,
        'baseline_bleu_scores': baseline_bleu_scores
    }
    
    with open('model_comparison.json', 'w') as f:
        json.dump(comparison_data, f)

    print("Model comparison data saved to 'model_comparison.json'")

    # Visualize attention for a sample sentence
    src_sentence = "A group of people stand in front of a building."
    tgt_sentence = "Eine Gruppe von Menschen steht vor einem Gebäude."
    visualize_attention(transformer_model, src_sentence, tgt_sentence, src_vocab, tgt_vocab, device)

if __name__ == '__main__':
    main()