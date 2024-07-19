import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch.utils.data import DataLoader
from src.transformer_implementation import TransformerModel, BaselineModel, TranslationDataset, train, evaluate, calculate_bleu, visualize_attention, collate_fn
import matplotlib.pyplot as plt

def test_models():
    # Sample data
    src_sentences = ["Hello world", "How are you", "Nice to meet you"]
    tgt_sentences = ["Bonjour le monde", "Comment allez-vous", "Ravi de vous rencontrer"]
    
    # Create vocabularies
    src_vocab = {'<pad>': 0, '<unk>': 1}
    tgt_vocab = {'<pad>': 0, '<unk>': 1}
    for sent in src_sentences:
        for word in sent.split():
            if word not in src_vocab:
                src_vocab[word] = len(src_vocab)
    for sent in tgt_sentences:
        for word in sent.split():
            if word not in tgt_vocab:
                tgt_vocab[word] = len(tgt_vocab)
    
    # Create dataset and dataloader
    dataset = TranslationDataset(src_sentences, tgt_sentences, src_vocab, tgt_vocab)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    
    # Initialize models
    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)
    d_model = 32
    nhead = 2
    num_encoder_layers = 2
    num_decoder_layers = 2
    dim_feedforward = 64
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    transformer_model = TransformerModel(src_vocab_size, tgt_vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward).to(device)
    baseline_model = BaselineModel(src_vocab_size, tgt_vocab_size, d_model).to(device)
    
    # Test forward pass
    for src, tgt in dataloader:
        src, tgt = src.to(device), tgt.to(device)
        print("Source shape:", src.shape)
        print("Target shape:", tgt.shape)
        
        # Ensure tgt input doesn't include the last token (which should be the target for the last prediction)
        tgt_input = tgt[:, :-1]
        print("Target input shape:", tgt_input.shape)
        
        try:
            transformer_output = transformer_model(src, tgt_input)
            print("Transformer output shape:", transformer_output.shape)
            print("Transformer output example:")
            print(transformer_output[0, 0, :10])  # Print first 10 logits of the first token in the first sequence
        except RuntimeError as e:
            print("Error in Transformer model:", str(e))
        
        try:
            baseline_output = baseline_model(src, tgt_input)
            print("Baseline output shape:", baseline_output.shape)
            print("Baseline output example:")
            print(baseline_output[0, 0, :10])  # Print first 10 logits of the first token in the first sequence
        except RuntimeError as e:
            print("Error in Baseline model:", str(e))
        
        break
    
    # Train and evaluate both models
    num_epochs = 100
    transformer_optimizer = torch.optim.Adam(transformer_model.parameters(), lr=0.001)
    baseline_optimizer = torch.optim.Adam(baseline_model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    
    transformer_losses = []
    baseline_losses = []
    
    for epoch in range(num_epochs):
        transformer_loss = train(transformer_model, dataloader, transformer_optimizer, criterion, device)
        baseline_loss = train(baseline_model, dataloader, baseline_optimizer, criterion, device)
        
        transformer_losses.append(transformer_loss)
        baseline_losses.append(baseline_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Transformer loss: {transformer_loss:.4f}")
            print(f"Baseline loss: {baseline_loss:.4f}")
            print()
    
    print("Final losses:")
    print(f"Transformer: {transformer_losses[-1]:.4f}")
    print(f"Baseline: {baseline_losses[-1]:.4f}")
    
    # Evaluate both models
    transformer_val_loss = evaluate(transformer_model, dataloader, criterion, device)
    baseline_val_loss = evaluate(baseline_model, dataloader, criterion, device)
    
    print("\nValidation losses:")
    print(f"Transformer: {transformer_val_loss:.4f}")
    print(f"Baseline: {baseline_val_loss:.4f}")
    
    # Calculate BLEU scores for both models
    try:
        transformer_bleu = calculate_bleu(transformer_model, dataloader, tgt_vocab, device)
        baseline_bleu = calculate_bleu(baseline_model, dataloader, tgt_vocab, device)
        
        print("\nBLEU scores:")
        print(f"Transformer: {transformer_bleu:.4f}")
        print(f"Baseline: {baseline_bleu:.4f}")
        
        # Print some example translations
        idx_to_word = {idx: word for word, idx in tgt_vocab.items()}
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            transformer_output = transformer_model(src, torch.zeros_like(tgt).to(device))
            baseline_output = baseline_model(src, torch.zeros_like(tgt).to(device))
            transformer_predicted = transformer_output.argmax(dim=2)
            baseline_predicted = baseline_output.argmax(dim=2)
            
            for i in range(len(src)):
                src_sentence = ' '.join([idx_to_word.get(idx.item(), '<unk>') for idx in src[i] if idx != 0])
                tgt_sentence = ' '.join([idx_to_word.get(idx.item(), '<unk>') for idx in tgt[i] if idx != 0])
                transformer_pred = ' '.join([idx_to_word.get(idx.item(), '<unk>') for idx in transformer_predicted[i] if idx != 0])
                baseline_pred = ' '.join([idx_to_word.get(idx.item(), '<unk>') for idx in baseline_predicted[i] if idx != 0])
                
                print(f"Source: {src_sentence}")
                print(f"Target: {tgt_sentence}")
                print(f"Transformer: {transformer_pred}")
                print(f"Baseline: {baseline_pred}")
                print()
            break
    except Exception as e:
        print("Error in BLEU score calculation or translation:", str(e))
    
    # Visualize losses
    plt.figure(figsize=(10, 5))
    plt.plot(transformer_losses, label='Transformer')
    plt.plot(baseline_losses, label='Baseline')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.savefig('loss_comparison.png')
    plt.close()
    
    print("Loss comparison plot saved as 'loss_comparison.png'")
    
    # Test attention visualization for Transformer
    try:
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        visualize_attention(transformer_model, "Hello world", "Bonjour le monde", src_vocab, tgt_vocab, device)
    except Exception as e:
        print("Error visualizing attention:", str(e))
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_models()