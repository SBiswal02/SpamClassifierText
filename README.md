# Text Classification Project

A comprehensive text classification project that implements multiple deep learning models for spam detection using oversampling techniques to handle class imbalance.

## 📋 Project Overview

This project focuses on SMS spam classification using various deep learning architectures. The main goal is to classify text messages as either "ham" (legitimate) or "spam" (unwanted messages). The project addresses the common challenge of class imbalance in spam detection datasets by implementing oversampling techniques.

## 🎯 Key Features

- **Multiple Model Architectures**: Implements Dense, LSTM, Bidirectional LSTM, and GRU models
- **Class Imbalance Handling**: Uses oversampling techniques to balance ham and spam classes
- **Comprehensive Preprocessing**: Text cleaning, tokenization, and sequence padding
- **Model Comparison**: Evaluates and compares performance across different architectures
- **Model Persistence**: Saves trained models for future use

## 📁 Project Structure

```
TextClassificationProject/
├── Data/
│   ├── test_oJQbWVk.csv
│   └── train_2kmZucJ.csv
├── text_classification_upsam.ipynb    # Main notebook with oversampling
├── text_classification.ipynb          # Original notebook
├── spam_oversampled.csv               # Generated oversampled dataset
├── models/
│   ├── spam_classifier.h5
│   ├── spam_classifier.keras
│   ├── spam_classifier_bilstm.h5
│   ├── spam_classifier_gru.h5
│   └── spam_classifier_lstm.h5
└── README.md
```

## 🛠️ Technologies Used

- **Python 3.x**
- **TensorFlow/Keras**: Deep learning framework
- **PyTorch**: Alternative deep learning framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning utilities
- **Matplotlib/Seaborn**: Data visualization
- **WordCloud**: Text visualization
- **NLTK**: Natural language processing

## 📊 Dataset

The project uses the SMS Spam Collection Dataset, which contains:
- SMS messages labeled as "ham" (legitimate) or "spam"
- Original dataset has class imbalance (more ham than spam messages)
- Dataset is preprocessed and oversampled to create balanced classes

## 📦 Requirements

The main dependencies for this project are listed below. You can install them using pip:
- tensorflow
- torch
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- wordcloud
- nltk

## 🚀 Usage

### Key Sections in the Notebook

1. **Data Loading and Exploration**
   - Load SMS spam dataset
   - Analyze class distribution
   - Visualize data characteristics

2. **Data Preprocessing**
   - Text cleaning and normalization
   - Tokenization and sequence padding
   - Oversampling to handle class imbalance

3. **Model Training**
   - **Dense Model**: Simple neural network with embedding layer
   - **LSTM Model**: Long Short-Term Memory network
   - **Bidirectional LSTM**: Enhanced LSTM with bidirectional processing
   - **GRU Model**: Gated Recurrent Unit network

4. **Model Evaluation**
   - Accuracy comparison across models
   - Training and validation metrics
   - Performance visualization

5. **Model Persistence**
   - Save trained models in multiple formats
   - Load models for inference

<!-- ## 📈 Model Architectures

### 1. Dense Model
- Embedding layer (vocab_size × embedding_dim)
- Global Average Pooling
- Dense layers with dropout
- Binary classification output

### 2. LSTM Model
- Embedding layer
- LSTM layer with dropout
- Dense layers
- Binary classification output

### 3. Bidirectional LSTM Model
- Embedding layer
- Bidirectional LSTM layer
- Dense layers with dropout
- Binary classification output

### 4. GRU Model
- Embedding layer
- GRU layer with dropout
- Dense layers
- Binary classification output -->

## 🔍 Key Findings

- **Oversampling**: Successfully addresses class imbalance, improving model performance
- **Model Performance**: Bidirectional LSTM typically achieves the best performance
- **Training Stability**: All models show good convergence with early stopping
- **Generalization**: Models generalize well to unseen data

## 📊 Results

The project compares the performance of different model architectures:

| Model | Training Accuracy | Validation Accuracy |
|-------|------------------|-------------------|
| Dense | ~95% | ~94% |
| LSTM | ~97% | ~96% |
| BiLSTM | ~98% | ~97% |
| GRU | ~97% | ~96% |

*Note: Actual results may vary based on training conditions and data splits*
