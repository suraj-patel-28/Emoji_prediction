
---

# Emoji Prediction Using Bidirectional LSTM

This project focuses on predicting emojis based on textual input using a Bidirectional LSTM model. The model is trained on sentences with associated emojis and can predict the most appropriate emoji for new sentences.

## Project Overview

The main goal of this project is to build a deep learning model capable of predicting the correct emoji for a given sentence. This model utilizes word embeddings from a pre-trained Word2Vec model and is trained using a Bidirectional LSTM network to learn contextual relationships in the text.

## Installation

To set up and run this project, you will need to install the necessary dependencies. You can install them using pip:

```bash
pip install numpy pandas emoji gensim matplotlib tensorflow nltk
```

## Usage

### 1. Preprocess Text Data

The text data is preprocessed by converting all words to lowercase and removing stopwords. The sentences are then converted into word embeddings using a pre-trained Word2Vec model.

### 2. Model Architecture

The model is built using a Bidirectional LSTM network. The architecture consists of:
- Two Bidirectional LSTM layers with 256 units each
- Dropout layers to prevent overfitting
- A Dense output layer with softmax activation for emoji prediction

### 3. Training the Model

The model is trained on the provided dataset for 50 epochs. The training process includes monitoring validation accuracy to ensure that the model generalizes well.

### 4. Evaluating the Model

After training, the model's performance is evaluated using accuracy metrics and a confusion matrix. The validation accuracy over epochs is also plotted to visualize the training process.

### 5. Predicting Emojis

The trained model can be used to predict emojis for new sentences. An example function `predict_emoji` is provided to make predictions based on input text.

## Project Files

- **`train_emoji.csv`**: The training dataset containing sentences and corresponding emoji labels.
- **`test_emoji.csv`**: The test dataset used for evaluating the model's performance.
- **`emoji_prediction.py`**: The main Python script that contains the complete code for training, evaluation, and prediction.
- **`plot_bi.jpg`**: A plot showing the training loss and accuracy over epochs.
- **`confusion_matrix.png`**: The confusion matrix plot showing the model's performance on the test data.

## Example Usage

To predict the emoji for a sentence, you can use the following code:

```python
input_sentence = "i want something to eat"
predicted_emoji = predict_emoji(input_sentence, lstm_model, word2vec_model)
print(f"Input Sentence: {input_sentence}")
print(f"Predicted Emoji: {predicted_emoji}")
```

This will output:

```
Input Sentence: i want something to eat
Predicted Emoji: 🍴
```

## Visualization

You can visualize the validation accuracy over epochs and the confusion matrix by running the provided scripts. These visualizations help in understanding the model's performance.

## Requirements

- Python 3.x
- NumPy
- Pandas
- Emoji
- Gensim
- Matplotlib
- TensorFlow
- NLTK

---
