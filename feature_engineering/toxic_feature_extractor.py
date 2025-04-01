import re
import string
from pathlib import Path
from typing import List, Tuple, Union

# NLP libraries
import nltk
import numpy as np
import pandas as pd
# Readability metrics
import textstat
# Large models
from gensim.models import Word2Vec, Doc2Vec, FastText
from gensim.models.doc2vec import TaggedDocument
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
# Scikit-learn
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif, SelectFromModel
from sklearn.model_selection import train_test_split

# Download NLTK resources
nltk.download(['punkt_tab', 'stopwords', 'wordnet', 'averaged_perceptron_tagger_eng', 'vader_lexicon'], quiet=True)

class ToxicFeatureExtractor:
    """Comprehensive feature extraction for toxic comment classification."""

    # INITIALIZATION
    def __init__(self, data_dir: str = 'data', tfidf_params=None, word2vec_params=None, 
                 doc2vec_params=None, fasttext_params=None, topic_params=None):
        self.data_dir = Path(data_dir)
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Set default feature extraction parameters
        self.tfidf_params = {
            'max_features': 5000,
            'min_df': 2,
            'max_df': 0.95,
            'ngram_range': (1, 2)
        }
        
        self.word2vec_params = {
            'vector_size': 100, 
            'window': 5, 
            'min_count': 1, 
            'workers': 4
        }
        
        self.doc2vec_params = {
            'vector_size': 100, 
            'min_count': 2, 
            'epochs': 40
        }
        
        self.fasttext_params = {
            'vector_size': 100, 
            'window': 5, 
            'min_count': 1, 
            'workers': 4
        }
        
        self.topic_params = {
            'n_topics': 10,
            'max_features': 5000,
            'max_iter': 10,
            'random_state': 42
        }
        
        # Update with user-provided parameters if any
        if tfidf_params:
            self.tfidf_params.update(tfidf_params)
        if word2vec_params:
            self.word2vec_params.update(word2vec_params)
        if doc2vec_params:
            self.doc2vec_params.update(doc2vec_params)
        if fasttext_params:
            self.fasttext_params.update(fasttext_params)
        if topic_params:
            self.topic_params.update(topic_params)
            
        self._init_data_paths()
        self._init_models()

    def _init_data_paths(self):
        self.train_file = self.data_dir / 'train.csv'
        self.test_file = self.data_dir / 'test.csv'
        self.test_labels_file = self.data_dir / 'test_labels.csv'

    def _init_models(self):
        self.vectorizer = None
        self.feature_selector = None
        self.word2vec_model = None
        self.doc2vec_model = None
        self.lda_model = None
        self.feature_names = None
        self.feature_ranges = None  # To store feature column ranges
        self.feature_info = None    # To store detailed feature metadata

    # DATA LOADING AND PREPROCESSING
    def load_data(self, dataset: str = 'train', verbose: bool = False) -> pd.DataFrame:
        """Load dataset (train/test)."""
        file_map = {
            'train': self.train_file,
            'test': self.test_file,
            'test_labels': self.test_labels_file
        }
        
        if verbose:
            print(f"Loading {dataset} data from {file_map[dataset]}...")
            
        df = pd.read_csv(file_map[dataset]) if file_map[dataset].exists() else None
        
        if verbose and df is not None:
            print(f"Loaded {len(df)} records with columns: {df.columns.tolist()}")
            
        return df

    def split_data(self, test_size: float = 0.2, random_state: int = 42, verbose: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split training data into train/validation sets."""
        if verbose:
            print(f"Splitting data with test_size={test_size}, random_state={random_state}...")
            
        df = self.load_data('train', verbose=verbose)
        if df is None:
            raise ValueError("No training data loaded")
            
        stratify = df.filter(regex='toxic|severe|obscene|threat|insult|identity').any(axis=1) if any(
            c in df.columns for c in ['toxic', 'severe_toxic']) else None
        
        train_df, val_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=stratify)
        
        if verbose:
            print(f"Split data into train ({len(train_df)} records) and validation ({len(val_df)} records)")
            
        return train_df, val_df

    def preprocess_text(self, text: str, verbose: bool = False, **kwargs) -> str:
        """Configurable text preprocessing pipeline."""
        if pd.isna(text): return ""
        
        if verbose:
            print(f"Preprocessing text: {text[:50]}..." if len(text) > 50 else f"Preprocessing text: {text}")
        
        # Text normalization
        text = text.lower() if kwargs.get('lowercase', True) else text
        text = re.sub(r'http\S+|www\S+|https\S+', '', text) if kwargs.get('remove_urls', True) else text
        text = re.sub(r'@\w+', '', text) if kwargs.get('remove_mentions', True) else text
        
        # Tokenization and cleaning
        tokens = word_tokenize(text.translate(str.maketrans('', '', string.punctuation))
                              if kwargs.get('remove_punctuation', True) else text)
        
        # Stopword removal and lemmatization
        tokens = [t for t in tokens if t not in self.stop_words] if kwargs.get('remove_stopwords', True) else tokens
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens] if kwargs.get('lemmatization', True) else tokens
        tokens = [self.stemmer.stem(t) for t in tokens] if kwargs.get('stemming', False) else tokens
        
        preprocessed = ' '.join(tokens)
        
        if verbose:
            print(f"Preprocessed: {preprocessed[:50]}..." if len(preprocessed) > 50 else f"Preprocessed: {preprocessed}")
            
        return preprocessed

    def preprocess_texts(self, texts: List[str], verbose: bool = False, **kwargs) -> List[str]:
        """Preprocess a list of texts."""
        if verbose:
            print(f"Preprocessing {len(texts)} texts...")
        return [self.preprocess_text(text, verbose=False, **kwargs) for text in texts]

    # FEATURE EXTRACTION
    def _vectorize(self, texts: List[str], vectorizer_class, preprocessed: bool = False, verbose: bool = False, **kwargs) -> np.ndarray:
        """Generic vectorization method."""
        if verbose:
            print(f"Vectorizing {len(texts)} texts using {vectorizer_class.__name__}...")
            
        if not preprocessed:
            if verbose:
                print("Preprocessing texts...")
            texts = [self.preprocess_text(t, verbose=False, **kwargs) for t in texts]
        
        vectorizer_params = kwargs.get('vectorizer_params', {})
        if verbose:
            print(f"Vectorizer parameters: {vectorizer_params}")
            
        self.vectorizer = vectorizer_class(**vectorizer_params)
        X = self.vectorizer.fit_transform(texts)
        self.feature_names = getattr(self.vectorizer, 'get_feature_names_out', lambda: [])() 
        
        if verbose:
            print(f"Vectorization complete. Matrix shape: {X.shape}")
            if len(self.feature_names) > 0:
                print(f"Top features: {self.feature_names[:5]}...")
            if isinstance(X, np.ndarray):
                print(f"Sample values (first row): {X[0, :5]}")
            else:
                print(f"Sample values (first row): {X[0, :5].toarray().flatten()}")
                
        return X

    def extract_bag_of_words(self, texts: List[str], verbose: bool = False, **kwargs) -> np.ndarray:
        return self._vectorize(texts, CountVectorizer, verbose=verbose, **kwargs)

    def extract_tfidf(self, texts: List[str], verbose: bool = False, **kwargs) -> np.ndarray:
        return self._vectorize(texts, TfidfVectorizer, verbose=verbose, **kwargs)

    def _embed_texts(self, texts: List[str], model_class, verbose: bool = False, **kwargs) -> np.ndarray:
        """Generic embedding method for word2vec/doc2vec/fasttext."""
        if verbose:
            print(f"Creating {model_class.__name__} embeddings for {len(texts)} texts...")
            
        preprocessed = kwargs.get('preprocessed', False)
        if verbose and not preprocessed:
            print("Preprocessing texts...")
            
        tokenized = [t.split() if preprocessed else self.preprocess_text(t, verbose=False, **kwargs).split() for t in texts]
        
        model_params = kwargs.get('model_params', {})
        if not model_params:
            # Default parameters for embedding models
            if model_class == Word2Vec:
                model_params = {'vector_size': 100, 'window': 5, 'min_count': 1, 'workers': 4}
            elif model_class == Doc2Vec:
                model_params = {'vector_size': 100, 'min_count': 2, 'epochs': 40}
        
        if verbose:
            print(f"Model parameters: {model_params}")
            
        vector_size = model_params.get('vector_size', 100)
        
        if model_class == Doc2Vec:
            tagged_docs = [TaggedDocument(words=tokens, tags=[str(i)]) for i, tokens in enumerate(tokenized)]
            model = model_class(tagged_docs, **model_params)
            embeddings = np.array([model.infer_vector(doc) for doc in tokenized])
        else:
            model = model_class(tokenized, **model_params)
            embeddings = np.array([np.mean([model.wv[w] for w in doc if w in model.wv] or [np.zeros(vector_size)], axis=0) for doc in tokenized])
        
        if verbose:
            print(f"Embedding complete. Matrix shape: {embeddings.shape}")
            print(f"Sample embeddings (first row, first 5 values): {embeddings[0, :5]}")
            
        return embeddings

    def extract_word2vec(self, texts: List[str], verbose: bool = False, **kwargs) -> np.ndarray:
        return self._embed_texts(texts, Word2Vec, verbose=verbose, **kwargs)

    def extract_doc2vec(self, texts: List[str], verbose: bool = False, **kwargs) -> np.ndarray:
        return self._embed_texts(texts, Doc2Vec, verbose=verbose, **kwargs)
    
    def extract_fasttext(self, texts: List[str], verbose: bool = False, **kwargs) -> np.ndarray:
        """Extract FastText embeddings."""
        return self._embed_texts(texts, FastText, verbose=verbose, **kwargs)

    def extract_topic_features(self, texts: List[str], verbose: bool = False, **kwargs) -> np.ndarray:
        """Extract topic modeling features using LDA."""
        # Get n_topics from instance parameters or kwargs
        n_topics = kwargs.get('n_topics', self.topic_params['n_topics'])
        
        if verbose:
            print(f"Extracting topic features for {len(texts)} texts with {n_topics} topics...")
            
        preprocessed = kwargs.get('preprocessed', False)
        
        # First get TF-IDF representation
        if not preprocessed:
            if verbose:
                print("Preprocessing texts...")
            preprocessed_texts = [self.preprocess_text(t, verbose=False, **kwargs) for t in texts]
        else:
            preprocessed_texts = texts
            
        vectorizer = TfidfVectorizer(max_features=self.topic_params.get('max_features', 5000))
        X = vectorizer.fit_transform(preprocessed_texts)
        
        if verbose:
            print(f"TF-IDF matrix shape: {X.shape}")
        
        # Apply LDA
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=self.topic_params.get('random_state', 42),
            max_iter=self.topic_params.get('max_iter', 10)
        )
        
        self.lda_model = lda
        topic_distribution = lda.fit_transform(X)
        
        if verbose:
            print(f"Topic distribution matrix shape: {topic_distribution.shape}")
            print(f"Sample topic distribution (first row): {topic_distribution[0]}")
            
            # Show top words per topic
            feature_names = vectorizer.get_feature_names_out()
            top_n = 5
            for topic_idx, topic in enumerate(lda.components_):
                if topic_idx < 3:  # Show only first 3 topics for brevity
                    top_words_idx = topic.argsort()[:-top_n-1:-1]
                    top_words = [feature_names[i] for i in top_words_idx]
                    print(f"Topic #{topic_idx}: {', '.join(top_words)}")
            
        return topic_distribution

    def extract_lexical_features(self, texts: List[str], verbose: bool = False, **kwargs) -> np.ndarray:
        """Extract lexical features from texts."""
        if verbose:
            print(f"Extracting lexical features for {len(texts)} texts...")
            
        preprocessed = kwargs.get('preprocessed', False)
        features = []
        
        for i, text in enumerate(texts):
            if pd.isna(text) or not text:
                features.append(np.zeros(10))
                continue
                
            # Count features
            tokens = word_tokenize(text)
            words = [w for w in tokens if w.isalpha()]
            
            # Basic counts
            char_count = len(text)
            word_count = len(words) 
            unique_word_count = len(set(words))
            
            # Sentence count (approximate)
            sentence_count = max(1, len(re.findall(r'[.!?]+', text)) + 1)
            
            # Average lengths
            avg_word_len = np.mean([len(w) for w in words]) if words else 0
            avg_sentence_len = word_count / sentence_count if sentence_count > 0 else 0
            
            # Special character counts
            punctuation_count = sum(1 for c in text if c in string.punctuation)
            uppercase_count = sum(1 for c in text if c.isupper())
            digit_count = sum(1 for c in text if c.isdigit())
            
            # Lexical diversity (type-token ratio)
            ttr = unique_word_count / word_count if word_count > 0 else 0
            
            features.append([
                char_count, word_count, unique_word_count, sentence_count,
                avg_word_len, avg_sentence_len, punctuation_count,
                uppercase_count, digit_count, ttr
            ])
        
        result = np.array(features)
        
        if verbose:
            print(f"Lexical features complete. Matrix shape: {result.shape}")
            print(f"Sample features (first row): {result[0]}")
            print(f"Feature names: ['char_count', 'word_count', 'unique_word_count', 'sentence_count', " +
                  f"'avg_word_len', 'avg_sentence_len', 'punctuation_count', 'uppercase_count', " +
                  f"'digit_count', 'type_token_ratio']")
            
        return result

    def extract_pos_features(self, texts: List[str], verbose: bool = False, **kwargs) -> np.ndarray:
        """Extract part-of-speech features from texts."""
        if verbose:
            print(f"Extracting POS features for {len(texts)} texts...")
            
        # Define POS categories to track
        pos_categories = {
            'noun': ['NN', 'NNS', 'NNP', 'NNPS'],
            'verb': ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
            'adj': ['JJ', 'JJR', 'JJS'],
            'adv': ['RB', 'RBR', 'RBS'],
            'pron': ['PRP', 'PRP$', 'WP', 'WP$'],
            'prep': ['IN'],
            'conj': ['CC'],
            'det': ['DT', 'PDT', 'WDT'],
            'interj': ['UH']
        }
        
        features = []
        
        for text in texts:
            if pd.isna(text) or not text:
                features.append(np.zeros(len(pos_categories)))
                continue
            
            try:
                # Tokenize and tag
                tokens = word_tokenize(text)
                tagged = pos_tag(tokens)
                
                # Count POS tag frequencies
                total_tokens = len(tagged)
                if total_tokens == 0:
                    features.append(np.zeros(len(pos_categories)))
                    continue
                
                # Count tags in each category
                pos_counts = {category: 0 for category in pos_categories}
                
                for _, tag in tagged:
                    for category, tag_list in pos_categories.items():
                        if tag in tag_list:
                            pos_counts[category] += 1
                
                # Normalize by text length
                pos_freqs = [pos_counts[category] / total_tokens for category in pos_categories]
                features.append(pos_freqs)
                
            except Exception as e:
                if verbose:
                    print(f"Error in POS tagging: {e}")
                features.append(np.zeros(len(pos_categories)))
        
        result = np.array(features)
        
        if verbose:
            print(f"POS features complete. Matrix shape: {result.shape}")
            print(f"Sample features (first row): {result[0]}")
            print(f"Feature names: {list(pos_categories.keys())}")
            
        return result

    def extract_sentiment_features(self, texts: List[str], verbose: bool = False, **kwargs) -> np.ndarray:
        """Extract sentiment features using VADER."""
        if verbose:
            print(f"Extracting sentiment features for {len(texts)} texts...")
            
        features = []
        
        for text in texts:
            if pd.isna(text) or not text:
                features.append([0, 0, 0, 0])
                continue
            
            # Get sentiment scores
            sentiment = self.sentiment_analyzer.polarity_scores(text)
            features.append([
                sentiment['pos'],    # Positive
                sentiment['neg'],    # Negative
                sentiment['neu'],    # Neutral
                sentiment['compound'] # Compound score
            ])
        
        result = np.array(features)
        
        if verbose:
            print(f"Sentiment features complete. Matrix shape: {result.shape}")
            print(f"Sample features (first 3 rows):")
            if len(result) >= 3:
                print(result[:3])
            else:
                print(result)
            print(f"Feature names: ['positive', 'negative', 'neutral', 'compound']")
            
        return result

    def extract_readability_features(self, texts: List[str], verbose: bool = False, **kwargs) -> np.ndarray:
        """Extract readability metrics from texts."""
        if verbose:
            print(f"Extracting readability features for {len(texts)} texts...")
            
        features = []
        
        for text in texts:
            if pd.isna(text) or not text or len(text.strip()) < 10:
                features.append([0, 0, 0, 0, 0])
                continue
                
            # Compute readability scores
            try:
                flesch_reading_ease = textstat.flesch_reading_ease(text)
                flesch_kincaid_grade = textstat.flesch_kincaid_grade(text)
                smog_index = textstat.smog_index(text)
                coleman_liau_index = textstat.coleman_liau_index(text)
                automated_readability_index = textstat.automated_readability_index(text)
                
                features.append([
                    flesch_reading_ease,
                    flesch_kincaid_grade,
                    smog_index,
                    coleman_liau_index,
                    automated_readability_index
                ])
            except Exception as e:
                if verbose:
                    print(f"Error calculating readability: {e}")
                features.append([0, 0, 0, 0, 0])
        
        result = np.array(features)
        
        if verbose:
            print(f"Readability features complete. Matrix shape: {result.shape}")
            print(f"Sample features (first row): {result[0]}")
            print(f"Feature names: ['flesch_reading_ease', 'flesch_kincaid_grade', 'smog_index', " +
                  f"'coleman_liau_index', 'automated_readability_index']")
            
        return result

    # ADD A NEW FEATURE EXTRACTION METHOD HERE (IF NEEDED)

    # FEATURE SELECTION (WIP, Cant really test yet)
    def select_features(self, X: np.ndarray, y: np.ndarray, method: str = 'chi2', k: int = None, verbose: bool = False):
        """Select best features using various methods."""
        if verbose:
            print(f"Selecting features using {method} method...")
            
        if method == 'chi2':
            selector = SelectKBest(chi2, k=k or min(1000, X.shape[1]))
        elif method == 'f_classif':
            selector = SelectKBest(f_classif, k=k or min(1000, X.shape[1]))
        elif method == 'mutual_info':
            selector = SelectKBest(mutual_info_classif, k=k or min(1000, X.shape[1]))
        elif method == 'model_based':
            from sklearn.linear_model import LogisticRegression
            selector = SelectFromModel(LogisticRegression(C=1, penalty='l1', solver='liblinear'))
        else:
            raise ValueError(f"Unknown feature selection method: {method}")
            
        X_new = selector.fit_transform(X, y)
        self.feature_selector = selector
        
        if verbose:
            print(f"Feature selection complete. Reduced feature matrix from {X.shape} to {X_new.shape}")
            if hasattr(selector, 'get_support'):
                n_selected = np.sum(selector.get_support())
                print(f"Selected {n_selected} features out of {X.shape[1]}")
            
        return X_new

    # USE THIS METHOD TO GET FEATURES YOU WANT
    def extract_all_features(self, texts: List[str], verbose: bool = False, **kwargs) -> Union[np.ndarray, pd.DataFrame]:
        """Extract and combine all specified features."""
        if verbose:
            print(f"Beginning feature extraction for {len(texts)} texts...")
            
        features = []
        
        # Preprocess texts for reuse
        if verbose:
            print("Preprocessing texts...")
            
        preprocessed = [self.preprocess_text(t, verbose=False, **kwargs) for t in texts]
        
        if verbose:
            print("Preprocessing complete.")
        
        # Build feature extraction plan using class parameters
        feature_extractors = {
            'tfidf': (self.extract_tfidf, {
                'preprocessed': True, 
                'vectorizer_params': self.tfidf_params
            }),
            'word2vec': (self.extract_word2vec, {
                'preprocessed': True,
                'model_params': self.word2vec_params
            }),
            'doc2vec': (self.extract_doc2vec, {
                'preprocessed': True,
                'model_params': self.doc2vec_params
            }),
            'fasttext': (self.extract_fasttext, {
                'preprocessed': True,
                'model_params': self.fasttext_params
            }),
            'lexical': (self.extract_lexical_features, {
                'preprocessed': True, 
                'original_text': texts
            }),
            'pos': (self.extract_pos_features, {
                'preprocessed': True
            }),
            'readability': (self.extract_readability_features, {}),
            'sentiment': (self.extract_sentiment_features, {}),
            'topic': (self.extract_topic_features, {
                'preprocessed': True, 
                'n_topics': self.topic_params['n_topics']
            })
        }
        
        # Track feature metadata for debugging
        feature_info = {}
        
        # Extract selected features
        feature_ranges = {}  # To track feature column ranges
        start_idx = 0
        
        for name, (func, params) in feature_extractors.items():
            if kwargs.get(f'include_{name}', True):
                try:
                    if name in ['sentiment', 'readability']:
                        # These work on original text
                        feature = func(texts, verbose=verbose, **params)
                    else:
                        # These work on preprocessed text
                        feature = func(preprocessed, verbose=verbose, **params)
                    
                    feature_matrix = feature if isinstance(feature, np.ndarray) else feature.toarray()
                    features.append(feature_matrix)
                    
                    # Store feature metadata
                    end_idx = start_idx + feature_matrix.shape[1]
                    feature_ranges[name] = (start_idx, end_idx, feature_matrix.shape[1])
                    start_idx = end_idx
                    
                    feature_info[name] = {
                        'shape': feature_matrix.shape,
                        'sample': feature_matrix[0, :min(5, feature_matrix.shape[1])],
                        'range': (feature_ranges[name][0], feature_ranges[name][1])
                    }
                    
                    if verbose:
                        print(f"✓ Extracted {name} features: {feature_matrix.shape}")
                except Exception as e:
                    if verbose:
                        print(f"✗ Failed to extract {name} features: {str(e)}")
        
        if not features:
            raise ValueError("No features were successfully extracted")
            
        # Combine features
        combined = np.hstack(features)
        
        if verbose:
            print(f"Combined feature matrix shape: {combined.shape}")
            print(f"Sample features (first row, first 10 features):")
            print(combined[0, :10])
            print("Feature types breakdown:")
            for name, info in feature_info.items():
                print(f"  {name}: {info['shape']}")
        
        # Create named DataFrame
        # Generated column names with feature type prefixes
        column_names = []
        for name, (start, end, size) in feature_ranges.items():
            # Create feature names with prefixes
            col_names = [f"{name.upper()}_{i+1}" for i in range(size)]
            column_names.extend(col_names)
        
        # Save feature ranges for later reference
        self.feature_ranges = feature_ranges
        self.feature_info = feature_info
            
        # Convert to DataFrame with named columns
        if kwargs.get('return_dataframe', True):
            result_df = pd.DataFrame(combined, columns=column_names)
            if verbose:
                print(f"Created DataFrame with named columns. Example columns: {column_names[:5]}...")
            return result_df
        else:
            return combined

    def transform_new_data(self, texts: List[str], verbose: bool = False, **kwargs) -> Union[np.ndarray, pd.DataFrame]:
        """
        Transform new data using the same feature extraction pipeline as during training.
        This updated method ensures that all feature types (TF-IDF, word2vec, doc2vec,
        fasttext, lexical, POS, sentiment, readability, topic) are included.
        """
        if verbose:
            print(f"Transforming {len(texts)} new texts...")

        # Preprocess texts
        preprocessed = [self.preprocess_text(t, verbose=False, **kwargs) for t in texts]
        extracted_features = []

        # TF-IDF features (using the fitted vectorizer)
        if self.vectorizer is not None:
            if verbose:
                print("Applying TF-IDF vectorization...")
            tfidf_features = self.vectorizer.transform(preprocessed)
            extracted_features.append(tfidf_features)
        
        # Word2Vec features
        if verbose:
            print("Extracting Word2Vec features...")
        word2vec_features = self.extract_word2vec(preprocessed, verbose=verbose, preprocessed=True, model_params=self.word2vec_params)
        extracted_features.append(word2vec_features)

        # Doc2Vec features
        if verbose:
            print("Extracting Doc2Vec features...")
        doc2vec_features = self.extract_doc2vec(preprocessed, verbose=verbose, preprocessed=True, model_params=self.doc2vec_params)
        extracted_features.append(doc2vec_features)
        
        # FastText features
        if verbose:
            print("Extracting FastText features...")
        fasttext_features = self.extract_fasttext(preprocessed, verbose=verbose, preprocessed=True, model_params=self.fasttext_params)
        extracted_features.append(fasttext_features)

        # LDA topic features (if fitted)
        if self.lda_model is not None and self.vectorizer is not None:
            if verbose:
                print("Extracting topic features...")
            X = self.vectorizer.transform(preprocessed)
            topic_features = self.lda_model.transform(X)
            extracted_features.append(topic_features)
        
        # Always include these feature types as they don't require fitting
        if kwargs.get('include_lexical', True):
            lexical_features = self.extract_lexical_features(texts, verbose=verbose)
            extracted_features.append(lexical_features)

        if kwargs.get('include_pos', True):
            pos_features = self.extract_pos_features(preprocessed, verbose=verbose, preprocessed=True)
            extracted_features.append(pos_features)

        if kwargs.get('include_sentiment', True):
            sentiment_features = self.extract_sentiment_features(texts, verbose=verbose)
            extracted_features.append(sentiment_features)

        if kwargs.get('include_readability', True):
            readability_features = self.extract_readability_features(texts, verbose=verbose)
            extracted_features.append(readability_features)

        # Combine all features ensuring the order and number of columns match training
        combined_features = np.hstack([
            f.toarray() if not isinstance(f, np.ndarray) else f
            for f in extracted_features
        ])

        # Apply feature selection if it was used during training
        if self.feature_selector is not None:
            if verbose:
                print("Applying feature selection...")
            combined_features = self.feature_selector.transform(combined_features)

        if verbose:
            print(f"Transformation complete. Output matrix shape: {combined_features.shape}")

        # If we previously created feature ranges, use them to name columns
        if hasattr(self, 'feature_ranges') and kwargs.get('return_dataframe', True):
            column_names = []
            for name, (start, end, size) in self.feature_ranges.items():
                col_names = [f"{name.upper()}_{i+1}" for i in range(size)]
                column_names.extend(col_names)
                
            result_df = pd.DataFrame(combined_features, columns=column_names)
            if verbose:
                print(f"Created DataFrame with named columns. Example columns: {column_names[:5]}...")
            return result_df
        else:
            return combined_features

