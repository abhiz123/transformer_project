# Transformer for Machine Translation

This project implements the Transformer model for machine translation, based on the paper "Attention Is All You Need" by Vaswani et al. It compares the Transformer model with a baseline Seq2Seq model on the Multi30k dataset (English to German translation).

## Project Structure

```
transformer_project/
├── data/
├── src/
│   ├── transformer_implementation.py
│   └── train_and_visualize.py
├── tests/
│   └── test_transformers.py
├── results/
│   ├── model_comparison.png
│   ├── attention_visualization.png
│   └── model_comparison.json
├── README.md
├── requirements.txt
└── .gitignore
```

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/transformer_project.git
   cd transformer_project
   ```

2. Install requirements:
   ```
   pip install -r requirements.txt
   ```

3. Download spaCy models:
   ```
   python -m spacy download en_core_web_sm
   python -m spacy download de_core_news_sm
   ```

## Usage

Run the training and visualization script:
```
python src/train_and_visualize.py
```

This script will:
- Download and process the Multi30k dataset
- Train both Transformer and baseline models
- Evaluate models using BLEU score
- Generate visualizations for model comparison and attention weights

## Running Tests

To run the tests, execute the following command from the project root:
```
python -m unittest discover tests
```

## Implementation Details

- Dataset: Multi30k (English to German translation)
- Transformer model: Implemented as per the original paper, with slight modifications for a smaller dataset
- Baseline model: Seq2Seq model with attention
- Training: Both models are trained for 20 epochs (adjustable in the code)
- Evaluation: Using BLEU score on the test set

## Limitations

Due to computational constraints and the use of a smaller dataset, the results may not fully reflect the capabilities of the Transformer architecture as described in the original paper. This implementation serves as a proof of concept.

## Future Work

- Train on a larger dataset for improved performance
- Implement more sophisticated tokenization (e.g., BPE)
- Experiment with different hyperparameters and model sizes
- Expand test coverage

## References

- Vaswani, A., et al. (2017). Attention Is All You Need. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- Multi30k Dataset: [GitHub Repository](https://github.com/multi30k/dataset)