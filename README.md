# CPSC544-W25-Toxicity-Detector

A machine learning project for detecting toxic comments in online platforms.

## Project Structure

```
CPSC544-W25-Toxicity-Detector/
├── data/                           # Data storage for datasets and processed features
├── feature_engineering/            # Feature extraction and processing components
│   └── toxic_feature_extractor.py  # Main feature extraction class
├── examples/                       # Example code
│   └── demo_all_features.py        # Demo showing feature extraction capabilities
├── models_test/                    # Model testing and development
│   └── extract_full_features.py    # Feature extraction for model training
├── notebooks/                      # Jupyter notebooks for analysis and visualization (if we decide to use notebooks in the future)
├── requirements.txt                # Project dependencies [Dont upload your own requirements.txt until we check that there are no version mismatch]
└── README.md                       # Project documentation
```

## Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

This will install the core packages including:
- nltk: For natural language processing tasks
- scikit-learn: For machine learning models and feature processing
- pandas: For data manipulation
- gensim: For word embeddings (Word2Vec, Doc2Vec, FastText)
- textstat: For readability metrics
- numpy: For numerical operations

## Feature Engineering

The project implements a comprehensive feature extraction system through the `ToxicFeatureExtractor` class, which offers various methods to extract different types of features from text data.

### Adding New Packages/Dependencies

If you need to add new packages for additional feature extraction or model training:

1. Install the package with pip:
   ```bash
   pip install new_package_name
   ```

2. Update the requirements.txt file:
   ```bash
   pip freeze > requirements.txt
   ```

3. Import and use the package in your code as needed.

### Extending Feature Extraction (Shouldn't be needed)

To add a new feature extraction method:

1. Create a new method in the `ToxicFeatureExtractor` class following the existing pattern:
   ```python
   def extract_new_feature(self, texts: List[str], verbose: bool = False, **kwargs) -> np.ndarray:
       # Your implementation here
       return feature_matrix
   ```

2. Update the `extract_all_features` method to include your new feature type.

## Running the Feature Demo

The project includes a demo script that showcases the feature extraction capabilities. Run it with:

```bash
python examples/demo_all_features.py
```

This demonstration will:
1. Load a sample demo dataset
2. Initialize the feature extractor
3. Extract multiple types of features from the sample text
4. Display information about the extracted features

## Using `extract_all_features`

The `extract_all_features` method in the `ToxicFeatureExtractor` class is the central feature extraction function that allows you to customize which features are extracted and how they are processed.

### Basic Usage

```python
from feature_engineering import ToxicFeatureExtractor

# Initialize the extractor with default parameters
extractor = ToxicFeatureExtractor()

# Extract features from a list of text comments
texts = ["This is a normal comment.", "This is a toxic comment, you idiot!"]
features = extractor.extract_all_features(texts, verbose=True)
```

### Customizing Feature Parameters

You can configure feature extraction parameters during initialization:

```python
# Example parameters for different feature types
tfidf_params = {
    'max_features': 10000,
    'min_df': 2,
    'max_df': 0.95,
    'ngram_range': (1, 3)
}

topic_params = {
    'n_topics': 6,
    'max_features': 1000,
    'random_state': 42
}

word2vec_params = {
    'vector_size': 300,
    'window': 5,
    'min_count': 1,
    'workers': 4
}

doc2vec_params = {
    'vector_size': 50, 
    'min_count': 3, 
    'epochs': 12
}

fasttext_params = {
    'vector_size': 150, 
    'window': 3, 
    'min_count': 3, 
    'workers': 5
}

# Initialize extractor with custom parameters
extractor = ToxicFeatureExtractor(
    tfidf_params=tfidf_params,
    topic_params=topic_params,
    word2vec_params=word2vec_params
    doc2vec_params=doc2vec_params,
    fasttext_params=fasttext_params
)
```

### Controlling Feature Selection

You can select which features to include during extraction:

```python
features = extractor.extract_all_features(
    texts,
    verbose=True,
    include_tfidf=True,        # TF-IDF vectorization features
    include_word2vec=True,     # Word2Vec embeddings
    include_doc2vec=False,     # Doc2Vec embeddings
    include_fasttext=False,    # FastText embeddings
    include_topic=True,        # Topic modeling features
    include_pos=False,         # Part-of-speech features
    include_sentiment=True,    # Sentiment analysis features
    include_lexical=True,      # Lexical statistics
    include_readability=True   # Readability metrics
)
```

### Default Parameters

The extractor uses the following default parameters if not specified:

```python
# TF-IDF defaults
tfidf_params = {
    'max_features': 5000,
    'min_df': 2,
    'max_df': 0.95,
    'ngram_range': (1, 2)
}

# Word2Vec defaults
word2vec_params = {
    'vector_size': 100, 
    'window': 5, 
    'min_count': 1, 
    'workers': 4
}

# Doc2Vec defaults
doc2vec_params = {
    'vector_size': 100, 
    'min_count': 2, 
    'epochs': 40
}

# FastText defaults
fasttext_params = {
    'vector_size': 100, 
    'window': 5, 
    'min_count': 1, 
    'workers': 4
}

# Topic modeling defaults
topic_params = {
    'n_topics': 10,
    'max_features': 5000,
    'max_iter': 10,
    'random_state': 42
}
```

### Debugging

Set `verbose=True` to get detailed information about the extraction process, which is helpful for debugging and understanding the feature dimensions.

### Output

The function returns a numpy array containing all the selected features combined. The shape of the output matrix will be `(n_samples, n_features)`, where `n_samples` is the number of texts, and `n_features` is the total number of extracted features.

## Model Training

### Stacking Classifier Approach (Evaluated and Discontinued)

We initially explored using a stacking classifier approach for toxic comment detection:
   - The single Random Forest and Gradient Boosting models outperformed the stacking classifier
   - Stacking introduced additional computational overhead without performance benefits
   - Individual models offered better interpretability with similar or better accuracy
   
Based on these results, we decided to focus on optimizing individual models (particularly Random Forest and Gradient Boosting) rather than pursuing the stacking approach further.
