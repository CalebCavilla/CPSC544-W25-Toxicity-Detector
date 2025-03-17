import sys
from pathlib import Path

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Feature extractor
from feature_engineering import ToxicFeatureExtractor

def main():
    print("=== Toxic Comment Feature Test ===")
    
    # Sample data path (replace with train.csv for full dataset)
    sample_file = project_root / "data" / "DEMO_train.csv"
    
    # Load data and initialize extractor with custom parameters
    print("Loading data...")
    df = pd.read_csv(sample_file)
    
    # Feature extraction parameters
    tfidf_params = {
        'max_features': 10,
        'min_df': 1,
        'max_df': 0.9,
        'ngram_range': (1, 3)
    }

    topic_params = {
        'n_topics': 5,
        'max_features': 50
    }

    word2vec_params = {
        'vector_size': 50,
        'window': 5,
        'min_count': 1,
        'workers': 4
    }

    doc2vec_params = {
        'vector_size': 50,
        'min_count': 2,
        'epochs': 40
    }

    fasttext_params = {
        'vector_size': 50,
        'window': 5,
        'min_count': 1,
        'workers': 4
    }

    # Initialize extractor
    extractor = ToxicFeatureExtractor(
        tfidf_params=tfidf_params,
        topic_params=topic_params,
        word2vec_params=word2vec_params,
        doc2vec_params=doc2vec_params,
        fasttext_params=fasttext_params
    )
    
    print("\nExtracting features...")
    
    # Extract all features
    try:
        features = extractor.extract_all_features(
            df['comment_text'].tolist(),
            verbose=True,
            # Feature selection flags (Feel free to change these to test stuff out)
            include_tfidf=True,
            include_word2vec=True,
            include_doc2vec=False,
            include_fasttext=False,
            include_topic=True,
            include_pos=False,
            include_sentiment=True,
            include_lexical=True,
            include_readability=True
        )
        
        # Show results
        print(f"\nCombined feature matrix shape: {features.shape}")

        # May be good practice to save these features to a file for later use
        print("Saving features to data/features_DEMO.csv")
        pd.DataFrame(features).to_csv(project_root / "data" / "features_DEMO.csv", index=False)

        
    except Exception as e:
        print(f"Error during feature extraction: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
