import sys
from pathlib import Path
import os
import pandas as pd
import pickle

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import modules
from feature_engineering import ToxicFeatureExtractor
from regularization.L2Regularization import Regularization

# Flag to control regularization
APPLY_REGULARIZATION = True
SAVE_FEATURES = True

import os

def main():
    print("=== Extracting features for full training data ===")

    # Full training data path
    train_file = project_root / "data" / "train.csv"
    features_output = project_root / "data" / "features_train.csv"

    # Load data
    print("Loading data...")
    df = pd.read_csv(train_file)
    print(f"Loaded {len(df)} records")

    # Initialize feature extractor with reduced TFIDF parameters
    tfidf_params = {
        'max_features': 1000,  # Reduced from default 5000
        'min_df': 3,          # Increased minimum document frequency
        'max_df': 0.9,
        'ngram_range': (1, 2)
    }
    
    word2vec_params = {
        'vector_size': 300,
        'window': 5,
        'min_count': 5,        # Increased minimum count
        'workers': 4
    }
    
    topic_params = {
        'n_topics': 10,
        'max_features': 1000  # Reduced from default 5000
    }
    
    extractor = ToxicFeatureExtractor(
        tfidf_params=tfidf_params,
        word2vec_params=word2vec_params,
        topic_params=topic_params
    )

    temp_features_path = project_root / "data" / "temp_features.csv"

    if os.path.exists(temp_features_path):
        print(f"Temporary features file {temp_features_path} exists. Skipping feature extraction.")
        feature_df = pd.read_csv(temp_features_path)
    else:
        print("Extracting features...")
        try:
            # Extract selected features
            features = extractor.extract_all_features(
                df['comment_text'].tolist(),
                verbose=True,
                # Feature selection flags
                include_tfidf=True,
                include_word2vec=True,
                include_doc2vec=False,
                include_fasttext=False,
                include_topic=True,
                include_pos=True,
                include_sentiment=True,
                include_lexical=True,
                include_readability=True
            )

            # Show results
            print(f"\nCombined feature matrix shape: {features.shape}")

            # features are already in a DataFrame with named columns if using updated extractor
            feature_df = features
        except Exception as e:
            print(f"Error during feature extraction: {str(e)}")
            import traceback
            traceback.print_exc()
            return  # Exit the function if an error occurs

    if APPLY_REGULARIZATION:
        print("\nApplying regularization for feature selection...")

        # Save temporary features for regularization
        temp_features_path = project_root / "data" / "temp_features.csv"
        feature_df.to_csv(temp_features_path, index=False)

        # Initialize regularization
        reg = Regularization(
        features_path=str(temp_features_path),
        target_path=str(train_file),
        threshold_value=0.02,  # Increased threshold to select fewer features
        correlation_threshold=0.5  # Increased threshold for more aggressive filtering
        )

        # Apply regularization
        print("Running regularization pipeline...")
        print("Loading data...")
        reg.load_data()
        new_data, mse, best_alpha, best_l1_ratio = reg.feature_selection(sample_fraction=0.1)

        print(
            f"Feature selection complete. Original features: {feature_df.shape[1]}, Selected features: {new_data.shape[1]}")
        print(f"Test MSE: {mse}, Best alpha: {best_alpha}, Best l1_ratio: {best_l1_ratio}")

        # Use regularized features
        feature_df = new_data

    if SAVE_FEATURES:
        print(f"Saving features to {features_output}")
        feature_df.to_csv(features_output, index=False)
        print(f"Features saved. Shape: {feature_df.shape}")
        
        # Optionally save feature type information
        if hasattr(extractor, 'feature_ranges') and extractor.feature_ranges:
            feature_info_path = project_root / "data" / "feature_info.pkl"
            with open(feature_info_path, 'wb') as f:
                pickle.dump({
                    'feature_ranges': extractor.feature_ranges,
                    'feature_info': extractor.feature_info
                }, f)
            print(f"Feature metadata saved to {feature_info_path}")

if __name__ == "__main__":
    main()
