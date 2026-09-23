# Next Word Prediction Using LSTM

## Project Overview

This project implements a Next Word Prediction system using a Long Short-Term Memory (LSTM) neural network.

The system predicts the next word based on a sequence of previously entered words and provides an interactive web interface using Flask.

Two models were developed:

1. Shakespeare Baseline Model – trained only on Shakespeare text.
2. Combined Corpus Model – trained using Shakespeare, Jane Austen, and Wikipedia text.

The combined corpus was introduced to provide greater language diversity compared with the Shakespeare-only model.

## Objectives

- Build a next-word prediction model using LSTM.
- Perform text preprocessing and tokenization.
- Generate sequential training data.
- Train and evaluate an LSTM language model.
- Compare a Shakespeare-only baseline with a diverse combined corpus.
- Generate multiple words from a given seed phrase.
- Improve text generation using temperature and top-k sampling.
- Deploy the trained model through a Flask web application.

## Dataset

The final model uses a combined corpus consisting of:

1. Shakespeare – used as the baseline literary corpus.
2. Jane Austen – provides additional narrative and conversational language patterns.
3. Wikipedia – provides encyclopedic and factual writing patterns.

The three datasets are combined into:

dataset/combined_corpus.txt

The combined corpus contains approximately:

- 6.19 million characters
- 50,965 cleaned lines

## Project Structure

Next_word_prediction/
│
├── dataset/
│   ├── shakespeare.txt
│   ├── austen.txt
│   ├── wikipedia.txt
│   └── combined_corpus.txt
│
├── models/
│   ├── shakespeare_model.keras
│   ├── shakespeare_tokenizer.pkl
│   ├── best_next_word_model.keras
│   ├── next_word_model.keras
│   └── tokenizer.pkl
│
├── plots/
│   ├── combined_training_validation_accuracy.png
│   └── combined_training_validation_loss.png
│
├── notebooks/
│   ├── 01_Shakespeare_Baseline.ipynb
│   └── 02_Combined_Corpus_Model.ipynb
│
├── templates/
│   └── index.html
│
├── static/
│   └── style.css
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore

## Technologies Used

Programming Language:
- Python

Machine Learning and Deep Learning:
- TensorFlow
- Keras
- LSTM
- NumPy
- Scikit-learn

Natural Language Processing:
- Text preprocessing
- Tokenization
- Sequence generation
- Padding
- Next-word prediction

Web Development:
- Flask
- HTML
- CSS
- JavaScript

Development Tools:
- Jupyter Notebook
- VS Code

## Text Preprocessing

The following preprocessing steps were applied:

1. Converted text to lowercase.
2. Removed punctuation and special characters.
3. Removed unnecessary whitespace.
4. Preserved individual lines for sequence generation.
5. Converted words into numerical token IDs.
6. Limited the vocabulary to 20,000 words.
7. Generated sequences using a maximum context length of 20 words.
8. Applied padding to make all input sequences the same length.

## Tokenization

Tokenization converts words into numerical integer IDs that can be processed by the neural network.

A vocabulary limit of 20,000 words was used to control the size of the model's output layer and keep training computationally manageable.

## Sequence Generation

For next-word prediction, sequences of previous words are created from the processed text.

For example:

Input:
to be

Target:
or

Another example:

Input:
the world is

Target:
next word

A maximum sequence length of 20 words was used.

## Training Data

The combined corpus produced:

Total sequences before limiting: 1,017,498

To keep memory usage and training time manageable, 300,000 sequences were selected for model training.

The data was divided into:

Training samples: 240,000
Validation samples: 60,000

An 80:20 train-validation split was used.

## LSTM Architecture

The final model uses the following architecture:

Input Sequence
      ↓
Embedding Layer
      ↓
LSTM (128 units)
      ↓
Dropout (0.3)
      ↓
Dense Layer
      ↓
Softmax
      ↓
Predicted Next Word

Model Configuration:

- Vocabulary size: 20,000
- Sequence length: 20
- Embedding dimension: 128
- LSTM units: 128
- Dropout: 0.3
- Optimizer: Adam
- Loss: Sparse Categorical Cross-Entropy
- Maximum epochs: 15
- Batch size: 64
- Early stopping: Yes

## Model Training

The model was trained using the Adam optimizer and sparse categorical cross-entropy loss.

Early stopping was used to monitor validation loss and prevent unnecessary training when validation performance stopped improving.

A model checkpoint was also used to save the model with the lowest validation loss.

The combined model stopped after 7 epochs because validation loss stopped improving.

## Training Results

Combined Corpus Model:

Best Epoch: 4
Best Validation Loss: 5.7770
Best Validation Accuracy: 13.70%

The training and validation curves are available in the plots directory.

Training and Validation Accuracy:
plots/combined_training_validation_accuracy.png

Training and Validation Loss:
plots/combined_training_validation_loss.png

## Baseline vs Combined Corpus

Two models were developed:

1. Shakespeare-only baseline model
2. Combined corpus model

The baseline model provides a reference point for evaluating the effect of using a more diverse corpus.

Feature Comparison:

Shakespeare Baseline:
- Corpus: Shakespeare
- Sequence length: 20
- LSTM units: 128
- Dropout: 0.2
- Training samples: 135,873
- Validation samples: 33,969
- Optimizer: Adam
- Loss: Sparse Categorical Cross-Entropy

Combined Corpus:
- Corpus: Shakespeare + Jane Austen + Wikipedia
- Vocabulary: 20,000
- Sequence length: 20
- LSTM units: 128
- Dropout: 0.3
- Training samples: 240,000
- Validation samples: 60,000
- Optimizer: Adam
- Loss: Sparse Categorical Cross-Entropy

Recorded Validation Performance:

Shakespeare Baseline:
- Best Validation Accuracy: approximately 10.14%
- Best Validation Loss: approximately 6.2261

Combined Corpus:
- Best Validation Accuracy: 13.70%
- Best Validation Loss: 5.7770
- Best Epoch: 4

The combined-corpus model showed lower validation loss and higher validation accuracy in the recorded training runs.

## Text Generation

The project supports two text-generation approaches.

1. Greedy / Argmax Generation

The basic prediction method selects the word with the highest predicted probability.

Model probabilities
        ↓
Highest probability word
        ↓
Next word

This method is simple and deterministic but can produce repetitive outputs.

2. Temperature and Top-K Sampling

The improved generation method uses:

- Temperature: 0.8
- Top-K: 10

Temperature controls the randomness of the generated output, while top-k sampling limits the selection to the most probable candidate words.

This provides more variation than always selecting the single highest-probability word.

## Human Evaluation

Generated text was evaluated qualitatively using:

- Fluency
- Coherence
- Contextual relevance
- Repetition

Sample 1:

Seed Text:
to be or

Generated Text:
to be or in the whole state of this and a few minutes she had heard the other of a friend of the

Observation:
The model produces a reasonable continuation initially, although coherence decreases toward the end.

Sample 2:

Seed Text:
the world is

Generated Text:
the world is a species of beetle in the national of the united kingdom with a british united states in the united states

Observation:
The output demonstrates encyclopedic language learned from the Wikipedia portion of the corpus, but some concepts are combined incorrectly.

Sample 3:

Seed Text:
she was very

Generated Text:
she was very much to hear her but i have been at the time and i am sure that i have not been

Observation:
The output has a more natural narrative style but is not completely coherent over the full sequence.

Human Evaluation Summary:

to be or
- Fluency: Fair
- Coherence: Fair
- Context Relevance: Fair

the world is
- Fluency: Fair
- Coherence: Poor
- Context Relevance: Fair

she was very
- Fluency: Good
- Coherence: Fair
- Context Relevance: Good

Overall Observation:

The combined-corpus model is capable of generating text with different writing styles, including literary and encyclopedic patterns.

However, generated text is not always grammatically or semantically coherent over longer sequences. This is expected from a relatively small LSTM language model trained on a limited subset of a larger corpus.

Temperature and top-k sampling provide more variation in generated text, but they do not guarantee grammatically or semantically perfect sentences.

## Flask Web Application

The trained model was integrated into a Flask web application.

The application provides:

- Text input
- Example prompts
- Number of words to generate
- Creativity / temperature control
- Next-word text generation
- Generated output display
- Copy-to-clipboard functionality
- Responsive user interface

The application loads the saved LSTM model and tokenizer, so the model does not need to be retrained every time the web application starts.

## Running the Application

1. Open the project folder in VS Code.

2. Install the required dependencies:

pip install -r requirements.txt

3. Start the Flask application:

python app.py

4. Open the application in a web browser:

http://127.0.0.1:5000

## Example Usage

Enter a seed phrase such as:

To be or

Choose the number of words to generate.

Adjust the creativity value if required.

Click:

Generate Text

The application generates a continuation using the trained LSTM model.

## Project Workflow

Data Collection
      ↓
Shakespeare + Jane Austen + Wikipedia
      ↓
Data Cleaning
      ↓
Tokenization
      ↓
Sequence Generation
      ↓
Padding
      ↓
Train / Validation Split
      ↓
LSTM Model
      ↓
Model Training
      ↓
Evaluation
      ↓
Next-Word Prediction
      ↓
Temperature + Top-K Sampling
      ↓
Flask Web Application

## Limitations

The current model has some limitations:

- Generated sentences can become incoherent over longer sequences.
- The model can sometimes combine unrelated concepts.
- The model may repeat certain words or phrases.
- The training corpus is limited compared with modern large language models.
- The model has only one LSTM layer.
- The model does not understand language in the same way as large Transformer-based models.
- Prediction quality depends strongly on the training corpus.

## Future Improvements

Possible improvements include:

- Training on a larger and more diverse corpus.
- Increasing the size of the training dataset.
- Using a multi-layer LSTM or GRU architecture.
- Using pretrained word embeddings.
- Experimenting with beam search.
- Improving sampling strategies.
- Using Transformer-based language models.
- Training with more computational resources.
- Adding automatic text-quality evaluation metrics.
- Improving sentence-level coherence.
- Deploying the application to a public cloud platform.

## Conclusion

This project demonstrates the complete development of an LSTM-based next-word prediction system.

A Shakespeare-only baseline model and a combined-corpus model were developed and evaluated.

The final combined corpus contains:

- Shakespeare
- Jane Austen
- Wikipedia

The project includes:

- Data collection
- Text preprocessing
- Tokenization
- Sequence generation
- Padding
- LSTM model development
- Model training
- Validation
- Accuracy and loss visualization
- Next-word prediction
- Temperature and top-k sampling
- Human evaluation
- Flask web application

The trained model was successfully integrated into an interactive web application where users can enter text and generate a continuation.

The complete workflow is:

Data Collection
      ↓
Data Preprocessing
      ↓
Tokenization
      ↓
Sequence Generation
      ↓
LSTM Training
      ↓
Evaluation
      ↓
Text Generation
      ↓
Flask Web Application

## Author

Sri Charana

B.Tech – Computer Science and Engineering (AI & ML)