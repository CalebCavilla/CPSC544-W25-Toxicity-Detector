import pandas as pd
import numpy as np
import sys
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import make_pipeline

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Feature extractor
from feature_engineering import ToxicFeatureExtractor

def main():

    # Initialize extractor
    feature_extractor = ToxicFeatureExtractor(data_dir='../data')
    df = feature_extractor.load_data(dataset='train', verbose=True)
    if df is None:
        print("Training data not found. Please check your data directory.")
        return

    train_df, val_df = feature_extractor.split_data(test_size=0.2, random_state=42, verbose=True)

    label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    # Extract texts and multi-labels for training and validation sets
    train_comments = train_df['comment_text'].tolist()
    val_comments = val_df['comment_text'].tolist()
    
    # Extract multi-labels as a 2D array
    train_labels = train_df[label_cols].values
    val_labels = val_df[label_cols].values

    # extract all features for training 
    print("Extracting training features")
    x_train = feature_extractor.extract_all_features(train_comments, verbose=True)
    print("transforming validation features")
    x_val = feature_extractor.transform_new_data(val_comments, verbose=True)

    print("Features extracted, building and fitting the model")
    # logistic regression Model
    base_model = LogisticRegression(max_iter=5000, solver='lbfgs')
    pipeline = make_pipeline(StandardScaler(), base_model)
    model = MultiOutputClassifier(pipeline)

    # fit the model
    model.fit(x_train, train_labels)

    # test model on the validation set
    print("Model fit, making predictions")
    predictions = model.predict(x_val)
    for idx, col in enumerate(label_cols):
        print(f"\nClassification Report for '{col}':")
        print(classification_report(val_labels[:, idx], predictions[:, idx]))
        acc = accuracy_score(val_labels[:, idx], predictions[:, idx])
        print(f"Accuracy for '{col}': {acc:.4f}")

if __name__ == "__main__":
    main()