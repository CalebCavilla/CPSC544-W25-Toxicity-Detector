# Toxic Comment Detector

## Overview

This project aims to identify toxic comments in online discussions using machine learning techniques. The system classifies text for various types of toxic content including hate speech, misinformation, and cyberbullying. The goal is to provide a scalable approach for evaluating toxicity in social media posts and comments.

## Problem Statement

As digital platforms continue to grow, moderating online content at scale has become increasingly complex. Regulatory bodies and users alike need better ways to assess the quality of discourse and identify harmful interactions. This project explores the use of machine learning techniques to classify and measure toxicity in online comments.

## Dataset

We use the Jigsaw Toxic Comment Classification dataset, containing comments from Wikipedia discussions labeled for toxic behavior. Each comment can belong to one or more of these categories:

- toxic - a general flag for toxicity
- severe_toxic - a more extreme level of toxicity
- obscene - use of offensive or inappropriate language
- threat - direct threats of harm
- insult - personal insults targeting individuals
- identity_hate - comments targeting specific identities such as race, gender, or religion

## Project Structure

```
CPSC544-W25-Toxicity-Detector/
├── data/                    # Dataset files and extracted features
├── feature_engineering/     # Feature extraction utilities
├── model/                   # Model training and evaluation
│   ├── extract_full_features.py  # Feature extraction pipeline
│   ├── optimized_classifiers.py  # Model training with hyperparameter tuning
│   ├── evaluate_optimized_models.py  # Model evaluation
│   └── saved_models/        # Saved trained models
├── model_testing/           # Experimental models
├── notebooks/               # Jupyter notebooks for exploration
├── regularization/          # Feature selection and regularization
└── requirements.txt         # Python dependencies
```

## Feature Engineering

Our approach extracts multiple types of features from text:

1. **Text-based features**:
   - TF-IDF vectorization
   - Word2Vec embeddings
   - Doc2Vec document embeddings
   - FastText embeddings
   - Topic modeling (LDA)

2. **Linguistic features**:
   - Part-of-speech (POS) distributions
   - Lexical statistics (word counts, sentence length, etc.)
   - Readability metrics
   - Sentiment analysis

Features are processed through regularization to reduce dimensionality and prevent overfitting.

## Models

We compare several machine learning approaches:

- Random Forest
- Gradient Boosting
- Logistic Regression
- Ensemble methods (voting classifier)

Our evaluation shows that ensemble approaches generally outperform individual models, with Random Forest and Gradient Boosting achieving the highest accuracy and F1 scores.

## Results

Our best model achieves:
- Accuracy: 96.82%
- F1 Score: 0.912
- ROC-AUC: 0.995

Feature importance analysis shows that Topic features and sentiment analysis contribute most significantly to classification accuracy.

## Usage

### Installation

```bash
# Clone the repository
git clone https://github.com/CalebCavilla/CPSC544-W25-Toxicity-Detector.git
cd CPSC544-W25-Toxicity-Detector

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Training

```bash
# Extract features and train models
python model/extract_full_features.py
python model/optimized_classifiers.py
```

## Future Work

- Expand the dataset to include more diverse sources of comments
- Experiment with transformer-based models (BERT, RoBERTa)
- Create a working demo interface for real-time comment analysis
- Implement feature importance analysis for model interpretability  
- Add support for cross-lingual toxicity detection
- Create an API for integration with content management systems

## Contributors

- Enzo Mutiso
- Ray Sandhu
- Caleb Cavilla
- Nathaniel Dafoe

## References

1. Davidson, T., Warmsley, D., Macy, M., & Weber, I. (2017). "Automated Hate Speech Detection and the Problem of Offensive Language." Proceedings of ICWSM.
2. Dixon, L., Li, J., Sorensen, J., Thain, N., & Vasserman, L. (2018). "Measuring and Mitigating Unintended Bias in Text Classification." AIES '18.
3. Jigsaw Toxic Comment Classification Challenge. https://kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge
