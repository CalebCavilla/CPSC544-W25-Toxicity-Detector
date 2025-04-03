import os
import sys
from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, VotingClassifier,
    AdaBoostClassifier, ExtraTreesClassifier
)
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV, train_test_split, StratifiedKFold
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

os.environ['LOKY_MAX_CPU_COUNT'] = '8'

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import utilities
from model.utils import (
    rf_param_dist, gb_param_dist, nb_param_dist, svm_param_dist,
    xgb_param_dist, lgbm_param_dist, et_param_dist, ada_param_dist,
    evaluate_smote_methods, apply_dimensionality_reduction,
    create_stacking_ensemble, create_weighted_voting_ensemble,
    RANDOM_STATE
)

# Configuration
DEBUG_MODE = False  # Set to True for reduced parameter search
SAMPLE_FRACTION = 1.0  # Fraction of data to use (0.0-1.0)
N_JOBS = -1  # Use all available cores
SAVE_PATH = project_root / "model" / "saved_models"

# Create the save directory if it doesn't exist
os.makedirs(SAVE_PATH, exist_ok=True)


class OptimizedClassifierTrainer:
    """Class to train and optimize classifiers for toxic comment detection."""

    def __init__(self, sample_fraction=SAMPLE_FRACTION, debug_mode=DEBUG_MODE, n_jobs=N_JOBS, random_state=RANDOM_STATE,
                 resampling_method=None, dim_reduction=None):
        """Init method."""
        self.sample_fraction = sample_fraction
        self.debug_mode = debug_mode
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.resampling_method = resampling_method
        self.dim_reduction = dim_reduction
        self.models = {}
        self.feature_importances = {}
        self.results = {}
        self.validation_scores = {}
        self.resampler = None
        self.reducer = None

    def load_data(self, features_path=None, target_path=None):
        """
        Load feature and target data.
        
        Args:
            features_path: Path to feature CSV file (default: data/features_train.csv)
            target_path: Path to target CSV file (default: data/train.csv)
            
        Returns:
            X: Feature matrix
            y: Target vector
        """
        # Set default paths if not provided
        if features_path is None:
            features_path = project_root / "data" / "features_train.csv"
        if target_path is None:
            target_path = project_root / "data" / "train.csv"
            
        print(f"Loading features from {features_path}...")
        X = pd.read_csv(features_path)
        print(f"Loaded features with shape: {X.shape}")
        
        print(f"Loading targets from {target_path}...")
        targets = pd.read_csv(target_path)
        # Regularization removes everything but the toxic column
        y = targets["toxic"]
        print(f"Loaded targets with shape: {y.shape}")
        
        return X, y
    
    def split_data(self, X, y, test_size=0.2):
        """
        Split data into train and test sets (with sample fraction).
        
        Args:
            X: Feature matrix
            y: Target vector
            test_size: Proportion of data to use for testing
            
        Returns:
            X_train, X_test, y_train, y_test: Split datasets
        """
        print("Splitting data into train/test sets...")
        
        # If using sample fraction, sample before splitting
        if self.sample_fraction < 1.0:
            print(f"Using {self.sample_fraction:.1%} of data for training/testing")
            X_sample, _, y_sample, _ = train_test_split(
                X, y, 
                train_size=self.sample_fraction,
                random_state=self.random_state,
                stratify=y
            )
            X, y = X_sample, y_sample
            print(f"Sampled data shape: {X.shape}")
            
        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size,
            random_state=self.random_state,
            stratify=y
        )
        
        print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
        print(f"Class distribution in train set: {pd.Series(y_train).value_counts().to_dict()}")
        print(f"Class distribution in test set: {pd.Series(y_test).value_counts().to_dict()}")

        return X_train, X_test, y_train, y_test

    def preprocess_data(self, X_train, X_test, y_train, y_test):
        """Apply preprocessing steps: dimensionality reduction and resampling."""
        # Apply dimensionality reduction if specified
        if self.dim_reduction:
            print(f"\nApplying {self.dim_reduction} dimensionality reduction...")
            X_train, X_test, self.reducer = apply_dimensionality_reduction(
                X_train, X_test, method=self.dim_reduction
            )

        # Determine best resampling method if not specified
        if self.resampling_method is None:
            print("\nEvaluating resampling methods...")
            _, best_method, self.resampler = evaluate_smote_methods(X_train, y_train, X_test, y_test, self.random_state)
            self.resampling_method = best_method

        # Apply resampling to training data if a method is specified and it's not 'original'
        if self.resampling_method and self.resampling_method != 'original':
            print(f"\nApplying {self.resampling_method} resampling...")
            X_train, y_train = self.resampler.fit_resample(X_train, y_train)

            print(f"After resampling - Train set: {X_train.shape}")
            print(f"Class distribution in resampled train set: {pd.Series(y_train).value_counts().to_dict()}")

        return X_train, X_test, y_train, y_test

    def optimize_random_forest(self, X_train, y_train):
        """
        Optimize a Random Forest classifier using HalvingRandomSearchCV.
        
        Args:
            X_train: Training feature matrix
            y_train: Training target vector
            
        Returns:
            Optimized Random Forest classifier
        """
        print("\nOptimizing Random Forest...")
        start_time = time.time()

        # Create parameter distribution - use simplified for debug mode
        if self.debug_mode:
            param_dist = {
                'n_estimators': [50, 100],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'max_features': ['sqrt'],
                'bootstrap': [True],
                'class_weight': ['balanced', None]
            }
        else:
            param_dist = rf_param_dist

        base_rf = RandomForestClassifier(random_state=self.random_state)

        # Make sure there are enough samples
        min_samples = 1500  # At least 1000 samples or 10% of data

        halving_search = HalvingRandomSearchCV(
            base_rf,
            param_distributions=param_dist,
            factor=3,  # Reduce candidates by a factor of 3 at each iteration
            resource='n_samples',  # Use fraction of samples for efficiency
            min_resources=min_samples,  # Start with enough samples for reliable CV
            max_resources=min(50000, len(X_train)),  # Use 50k or the full set if smaller
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=1,
            scoring='f1'
        )

        halving_search.fit(X_train, y_train)
        best_rf = halving_search.best_estimator_
        elapsed_time = time.time() - start_time

        print(f"\nRandom Forest optimization completed in {elapsed_time:.2f} seconds")
        print(f"Best parameters: {halving_search.best_params_}")
        print(f"Best F1 score: {halving_search.best_score_:.4f}")

        self.models['random_forest'] = best_rf
        self.validation_scores['random_forest'] = halving_search.best_score_

        if hasattr(best_rf, 'feature_importances_'):
            self.feature_importances['random_forest'] = best_rf.feature_importances_

        return best_rf

    def optimize_gradient_boosting(self, X_train, y_train):
        """
        Optimize a Gradient Boosting classifier using HalvingRandomSearchCV.
        
        Args:
            X_train: Training feature matrix
            y_train: Training target vector
            
        Returns:
            Optimized Gradient Boosting classifier
        """
        print("\nOptimizing Gradient Boosting...")
        start_time = time.time()

        if self.debug_mode:
            param_dist = {
                'n_estimators': [50, 100],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'subsample': [0.8, 1.0],
                'max_features': ['sqrt']
            }
        else:
            param_dist = gb_param_dist

        base_gb = GradientBoostingClassifier(random_state=self.random_state)
        min_samples = 1500

        halving_search = HalvingRandomSearchCV(
            base_gb,
            param_distributions=param_dist,
            factor=3,
            resource='n_samples',
            min_resources=min_samples,
            max_resources=min(50000, len(X_train)),
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=1,
            scoring='f1'
        )

        halving_search.fit(X_train, y_train)
        best_gb = halving_search.best_estimator_
        elapsed_time = time.time() - start_time
        
        print(f"\nGradient Boosting optimization completed in {elapsed_time:.2f} seconds")
        print(f"Best parameters: {halving_search.best_params_}")
        print(f"Best F1 score: {halving_search.best_score_:.4f}")
        
        self.models['gradient_boosting'] = best_gb
        self.validation_scores['gradient_boosting'] = halving_search.best_score_

        if hasattr(best_gb, 'feature_importances_'):
            self.feature_importances['gradient_boosting'] = best_gb.feature_importances_
        
        return best_gb
    
    def optimize_multinomial_nb(self, X_train, y_train):
        """
        Optimize a Multinomial Naive Bayes classifier using HalvingRandomSearchCV.
        
        Args:
            X_train: Training feature matrix
            y_train: Training target vector
            
        Returns:
            Optimized Naive Bayes classifier
        """
        print("\nOptimizing Multinomial Naive Bayes...")
        start_time = time.time()

        # Ensure no negative values for Naive Bayes (cheap workaround)
        X_train_nb = X_train.copy()
        X_train_nb[X_train_nb < 0] = 0

        param_dist = nb_param_dist

        base_nb = MultinomialNB()
        min_samples = 1500

        halving_search = HalvingRandomSearchCV(
            base_nb,
            param_distributions=param_dist,
            factor=3,
            resource='n_samples',
            min_resources=min_samples,
            max_resources=min(50000, len(X_train)),
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=1,
            scoring='f1'
        )

        halving_search.fit(X_train_nb, y_train)
        best_nb = halving_search.best_estimator_
        elapsed_time = time.time() - start_time
        
        print(f"\nMultinomial Naive Bayes optimization completed in {elapsed_time:.2f} seconds")
        print(f"Best parameters: {halving_search.best_params_}")
        print(f"Best F1 score: {halving_search.best_score_:.4f}")
        
        self.models['naive_bayes'] = best_nb
        self.validation_scores['naive_bayes'] = halving_search.best_score_

        return best_nb

    def optimize_svm(self, X_train, y_train):
        """
        Optimize an SVM classifier using HalvingRandomSearchCV.

        Args:
            X_train: Training feature matrix
            y_train: Training target vector

        Returns:
            Optimized SVM classifier
        """
        print("\nOptimizing SVM...")
        start_time = time.time()

        if self.debug_mode:
            param_dist = {
                'C': [0.1, 1.0, 10.0],
                'gamma': ['scale', 'auto'],
                'kernel': ['linear', 'rbf'],
                'probability': [True],
                'class_weight': ['balanced', None]
            }
        else:
            param_dist = svm_param_dist

        base_svm = SVC(random_state=self.random_state)
        min_samples = 1500

        halving_search = HalvingRandomSearchCV(
            base_svm,
            param_distributions=param_dist,
            factor=3,
            resource='n_samples',
            min_resources=min_samples,
            max_resources=min(50000, len(X_train)),
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=1,
            scoring='f1'
        )

        halving_search.fit(X_train, y_train)
        best_svm = halving_search.best_estimator_
        elapsed_time = time.time() - start_time

        print(f"\nSVM optimization completed in {elapsed_time:.2f} seconds")
        print(f"Best parameters: {halving_search.best_params_}")
        print(f"Best F1 score: {halving_search.best_score_:.4f}")

        self.models['svm'] = best_svm
        self.validation_scores['svm'] = halving_search.best_score_

        return best_svm

    def optimize_xgboost(self, X_train, y_train):
        """
        Optimize an XGBoost classifier using HalvingRandomSearchCV.

        Args:
            X_train: Training feature matrix
            y_train: Training target vector

        Returns:
            Optimized XGBoost classifier
        """
        print("\nOptimizing XGBoost...")
        start_time = time.time()

        if self.debug_mode:
            param_dist = {
                'n_estimators': [50, 100],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0],
                'gamma': [0, 1]
            }
        else:
            param_dist = xgb_param_dist

        base_xgb = XGBClassifier(random_state=self.random_state)
        min_samples = 1500

        halving_search = HalvingRandomSearchCV(
            base_xgb,
            param_distributions=param_dist,
            factor=3,
            resource='n_samples',
            min_resources=min_samples,
            max_resources=min(50000, len(X_train)),
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=1,
            scoring='f1'
        )

        halving_search.fit(X_train, y_train)
        best_xgb = halving_search.best_estimator_
        elapsed_time = time.time() - start_time

        print(f"\nXGBoost optimization completed in {elapsed_time:.2f} seconds")
        print(f"Best parameters: {halving_search.best_params_}")
        print(f"Best F1 score: {halving_search.best_score_:.4f}")

        self.models['xgboost'] = best_xgb
        self.validation_scores['xgboost'] = halving_search.best_score_

        if hasattr(best_xgb, 'feature_importances_'):
            self.feature_importances['xgboost'] = best_xgb.feature_importances_

        return best_xgb

    def optimize_lightgbm(self, X_train, y_train):
        """
        Optimize a LightGBM classifier using HalvingRandomSearchCV.

        Args:
            X_train: Training feature matrix
            y_train: Training target vector

        Returns:
            Optimized LightGBM classifier
        """
        print("\nOptimizing LightGBM...")
        start_time = time.time()

        if self.debug_mode:
            param_dist = {
                'n_estimators': [50, 100],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0],
                'num_leaves': [31, 63, 127]
            }
        else:
            param_dist = lgbm_param_dist

        base_lgbm = LGBMClassifier(random_state=self.random_state)
        min_samples = 1500

        halving_search = HalvingRandomSearchCV(
            base_lgbm,
            param_distributions=param_dist,
            factor=3,
            resource='n_samples',
            min_resources=min_samples,
            max_resources=min(50000, len(X_train)),
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=1,
            scoring='f1'
        )

        halving_search.fit(X_train, y_train)
        best_lgbm = halving_search.best_estimator_
        elapsed_time = time.time() - start_time

        print(f"\nLightGBM optimization completed in {elapsed_time:.2f} seconds")
        print(f"Best parameters: {halving_search.best_params_}")
        print(f"Best F1 score: {halving_search.best_score_:.4f}")

        self.models['lightgbm'] = best_lgbm
        self.validation_scores['lightgbm'] = halving_search.best_score_

        if hasattr(best_lgbm, 'feature_importances_'):
            self.feature_importances['lightgbm'] = best_lgbm.feature_importances_

        return best_lgbm

    def optimize_extra_trees(self, X_train, y_train):
        """
        Optimize an Extra Trees classifier using HalvingRandomSearchCV.

        Args:
            X_train: Training feature matrix
            y_train: Training target vector

        Returns:
            Optimized Extra Trees classifier
        """
        print("\nOptimizing Extra Trees...")
        start_time = time.time()

        if self.debug_mode:
            param_dist = {
                'n_estimators': [50, 100],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'max_features': ['sqrt'],
                'bootstrap': [True],
                'class_weight': ['balanced', None]
            }
        else:
            param_dist = et_param_dist

        base_et = ExtraTreesClassifier(random_state=self.random_state)
        min_samples = 1500

        halving_search = HalvingRandomSearchCV(
            base_et,
            param_distributions=param_dist,
            factor=3,
            resource='n_samples',
            min_resources=min_samples,
            max_resources=min(50000, len(X_train)),
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=1,
            scoring='f1'
        )

        halving_search.fit(X_train, y_train)
        best_et = halving_search.best_estimator_
        elapsed_time = time.time() - start_time

        print(f"\nExtra Trees optimization completed in {elapsed_time:.2f} seconds")
        print(f"Best parameters: {halving_search.best_params_}")
        print(f"Best F1 score: {halving_search.best_score_:.4f}")

        self.models['extra_trees'] = best_et
        self.validation_scores['extra_trees'] = halving_search.best_score_

        if hasattr(best_et, 'feature_importances_'):
            self.feature_importances['extra_trees'] = best_et.feature_importances_

        return best_et

    def optimize_adaboost(self, X_train, y_train):
        """
        Optimize an AdaBoost classifier using HalvingRandomSearchCV.

        Args:
            X_train: Training feature matrix
            y_train: Training target vector

        Returns:
            Optimized AdaBoost classifier
        """
        print("\nOptimizing AdaBoost...")
        start_time = time.time()

        if self.debug_mode:
            param_dist = {
                'n_estimators': [50, 100],
                'learning_rate': [0.01, 0.1, 1.0],
                'algorithm': ['SAMME', 'SAMME.R']
            }
        else:
            param_dist = ada_param_dist

        base_ada = AdaBoostClassifier(random_state=self.random_state)
        min_samples = 1500

        halving_search = HalvingRandomSearchCV(
            base_ada,
            param_distributions=param_dist,
            factor=3,
            resource='n_samples',
            min_resources=min_samples,
            max_resources=min(50000, len(X_train)),
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=1,
            scoring='f1'
        )

        halving_search.fit(X_train, y_train)
        best_ada = halving_search.best_estimator_
        elapsed_time = time.time() - start_time

        print(f"\nAdaBoost optimization completed in {elapsed_time:.2f} seconds")
        print(f"Best parameters: {halving_search.best_params_}")
        print(f"Best F1 score: {halving_search.best_score_:.4f}")

        self.models['adaboost'] = best_ada
        self.validation_scores['adaboost'] = halving_search.best_score_

        if hasattr(best_ada, 'feature_importances_'):
            self.feature_importances['adaboost'] = best_ada.feature_importances_

        return best_ada

    def create_voting_classifier(self):
        """Create a voting classifier from optimized models."""
        if len(self.models) < 2:
            raise ValueError("At least two optimized models are required to create a voting classifier")

        print("\nCreating Voting Classifier...")

        # Handle Naive Bayes separately - don't include in voting classifier because of negative values
        has_naive_bayes = 'naive_bayes' in self.models
        nb_model = self.models.pop('naive_bayes') if has_naive_bayes else None

        # Create a list of tuples (name, model) for the voting classifier
        estimators = [(name, model) for name, model in self.models.items()]

        voting_clf = VotingClassifier(
            estimators=estimators,
            voting='soft',
            n_jobs=self.n_jobs
        )

        # Add Naive Bayes back to models if it was present
        if has_naive_bayes:
            self.models['naive_bayes'] = nb_model

        print(f"Voting Classifier created with models: {[name for name, _ in estimators]}")

        return voting_clf

    def evaluate_model(self, model, X_test, y_test, model_name='model'):
        """Evaluate a model on test data."""
        print(f"\nEvaluating {model_name}...")

        # Handle Naive Bayes - ensure no negative values
        if model_name == 'naive_bayes':
            X_test_eval = X_test.copy()
            X_test_eval[X_test_eval < 0] = 0
        else:
            X_test_eval = X_test

        y_pred = model.predict(X_test_eval)
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        # Add ROC AUC
        if hasattr(model, "predict_proba"):
            try:
                y_prob = model.predict_proba(X_test_eval)[:, 1]
                metrics['roc_auc'] = roc_auc_score(y_test, y_prob)
            except:
                print(f"Warning: Could not calculate ROC AUC for {model_name}")

        print(f"{model_name} - Accuracy: {metrics['accuracy']:.4f}")
        print(f"{model_name} - F1 Score: {metrics['f1']:.4f}")
        if 'roc_auc' in metrics:
            print(f"{model_name} - ROC AUC: {metrics['roc_auc']:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        self.results[model_name] = metrics
        
        return metrics
    
    def plot_confusion_matrices(self):
        """Plot confusion matrices for all evaluated models."""
        if not self.results:
            print("No evaluation results available to plot")
            return
        
        n_models = len(self.results)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 5))
        
        # Handle case with only one model
        if n_models == 1:
            axes = [axes]
        
        for ax, (model_name, metrics) in zip(axes, self.results.items()):
            cm = metrics['confusion_matrix']
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            # Plot the confusion matrix
            sns.heatmap(
                cm_normalized, 
                annot=cm,
                fmt='d',
                cmap='Blues', 
                cbar=False,
                ax=ax
            )

            ax.set_title(
                f"{model_name.replace('_', ' ').title()}\nAccuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            ax.set_xticklabels(['Not Toxic', 'Toxic'])
            ax.set_yticklabels(['Not Toxic', 'Toxic'])
        
        plt.tight_layout()
        plt.savefig(SAVE_PATH / "confusion_matrices.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importances(self, feature_names=None, top_n=20):
        """
        Plot feature importances for models that support it.
        
        Args:
            feature_names: List of feature names
            top_n: Number of top features to display
        """
        if not self.feature_importances:
            print("No feature importance data available to plot")
            return
        
        n_models = len(self.feature_importances)
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 10))

        if n_models == 1:
            axes = [axes]
        
        for ax, (model_name, importances) in zip(axes, self.feature_importances.items()):
            # Create feature names if not provided
            if feature_names is None:
                feature_names = [f"Feature {i}" for i in range(len(importances))]
            
            # Create a DataFrame for sorting
            importance_df = pd.DataFrame({
                'Feature': feature_names[:len(importances)],
                'Importance': importances
            })
            
            # Sort and get top features
            importance_df = importance_df.sort_values('Importance', ascending=False).head(top_n)
            
            # Plot horizontal bar chart
            sns.barplot(
                x='Importance',
                y='Feature',
                data=importance_df,
                ax=ax
            )
            
            ax.set_title(f"{model_name.replace('_', ' ').title()}\nTop {top_n} Feature Importances")
            ax.set_xlabel('Importance')
            ax.set_ylabel('Feature')
        
        plt.tight_layout()
        plt.savefig(SAVE_PATH / "feature_importances.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_comparison_bar(self):
        """Plot performance comparison between models."""
        if not self.results:
            print("No evaluation results available to plot")
            return
        
        # Extract metrics for comparison
        metrics_df = pd.DataFrame({
            'Model': [],
            'Metric': [],
            'Value': []
        })
        
        for model_name, metrics in self.results.items():
            # Include accuracy and F1
            metrics_df = pd.concat([metrics_df, pd.DataFrame({
                'Model': [model_name.replace('_', ' ').title()] * 2,
                'Metric': ['Accuracy', 'F1 Score'],
                'Value': [metrics['accuracy'], metrics['f1']]
            })], ignore_index=True)
            
            # ROC AUC
            if 'roc_auc' in metrics:
                metrics_df = pd.concat([metrics_df, pd.DataFrame({
                    'Model': [model_name.replace('_', ' ').title()],
                    'Metric': ['ROC AUC'],
                    'Value': [metrics['roc_auc']]
                })], ignore_index=True)

        plt.figure(figsize=(12, 6))
        ax = sns.barplot(
            x='Model',
            y='Value',
            hue='Metric',
            data=metrics_df
        )
        
        # Add value labels
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f')
        
        plt.title('Model Performance Comparison')
        plt.ylabel('Score')
        plt.ylim(0, 1.1)  # Set y-axis limit to accommodate labels
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(title='Metric')
        plt.tight_layout()
        
        plt.savefig(SAVE_PATH / "model_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_models(self):
        """Save optimized models and evaluation metrics to disk."""
        if not os.path.exists(SAVE_PATH):
            os.makedirs(SAVE_PATH)
        
        print(f"\nSaving models to {SAVE_PATH}...")
        
        # Save individual models
        for model_name, model in self.models.items():
            # Skip voting classifier as it will be saved separately
            if model_name == 'voting':
                continue
                
            model_path = SAVE_PATH / f"{model_name}.joblib"
            joblib.dump(model, model_path)
            print(f"Saved {model_name} to {model_path}")
        
        # Save voting classifier if it exists
        if 'voting' in self.models:
            voting_path = SAVE_PATH / "voting.joblib"
            joblib.dump(self.models['voting'], voting_path)
            print(f"Saved voting classifier to {voting_path}")
        
        # Save evaluation results separately for each model
        for model_name, metrics in self.results.items():
            metrics_path = SAVE_PATH / f"{model_name}_metrics.joblib"
            joblib.dump(metrics, metrics_path)
            print(f"Saved {model_name} evaluation metrics to {metrics_path}")
            
        # Also save all results together
        all_results_path = SAVE_PATH / "all_metrics.joblib"
        joblib.dump(self.results, all_results_path)
        print(f"Saved all evaluation results to {all_results_path}")
        
        print("Model saving complete.")

    def train_and_evaluate(self, models_to_train=None):
        """Complete pipeline for training and evaluating models."""
        # Available models
        all_models = {
            'random_forest': self.optimize_random_forest,
            'gradient_boosting': self.optimize_gradient_boosting,
            'naive_bayes': self.optimize_multinomial_nb,
            'svm': self.optimize_svm,
            'xgboost': self.optimize_xgboost,
            'lightgbm': self.optimize_lightgbm,
            'extra_trees': self.optimize_extra_trees,
            'adaboost': self.optimize_adaboost
        }

        # Determine which models to train
        if models_to_train is None:
            # Train all models by default
            models_to_train = list(all_models.keys())

        # Load data
        X, y = self.load_data()
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y)

        # Preprocess data: dimensionality reduction and resampling
        X_train, X_test, y_train, y_test = self.preprocess_data(X_train, X_test, y_train, y_test)

        # Train selected models
        for model_name in models_to_train:
            if model_name in all_models:
                try:
                    print(f"\nTraining {model_name}...")
                    model_trainer = all_models[model_name]

                    # Handle Naive Bayes separately (replace negatives with zeros)
                    if model_name == 'naive_bayes':
                        X_train_nb = X_train.copy()
                        X_train_nb[X_train_nb < 0] = 0
                        model = model_trainer(X_train_nb, y_train)
                    else:
                        model = model_trainer(X_train, y_train)

                    # Evaluate the model
                    self.evaluate_model(model, X_test, y_test, model_name)
                except Exception as e:
                    print(f"Error training {model_name}: {e}")
            else:
                print(f"Warning: Model {model_name} not recognized. Skipping.")

        # Create and evaluate ensemble classifiers
        if len(self.models) >= 2:
            try:
                # Standard voting classifier
                voting_clf = self.create_voting_classifier()
                voting_clf.fit(X_train, y_train)
                self.models['voting'] = voting_clf
                self.evaluate_model(voting_clf, X_test, y_test, 'voting')

                # Weighted voting classifier based on validation scores
                if len(self.validation_scores) >= 2:
                    weighted_voting_clf = create_weighted_voting_ensemble(
                        self.models, self.validation_scores, voting='soft', n_jobs=self.n_jobs
                    )
                    weighted_voting_clf.fit(X_train, y_train)
                    self.models['weighted_voting'] = weighted_voting_clf
                    self.evaluate_model(weighted_voting_clf, X_test, y_test, 'weighted_voting')

                # Stacking classifier
                stacking_clf = create_stacking_ensemble(
                    self.models, meta_learner=None, cv=5, n_jobs=self.n_jobs
                )
                stacking_clf.fit(X_train, y_train)
                self.models['stacking'] = stacking_clf
                self.evaluate_model(stacking_clf, X_test, y_test, 'stacking')
            except Exception as e:
                print(f"Error creating ensemble models: {e}")

        # Plot results
        self.plot_confusion_matrices()
        
        if self.feature_importances:
            # Get feature names if possible
            if isinstance(X, pd.DataFrame) and X.columns is not None:
                feature_names = X.columns.tolist()
            else:
                feature_names = None
                
            self.plot_feature_importances(feature_names)
        
        self.plot_comparison_bar()
        
        # Save models
        self.save_models()
        
        return self.models, self.results


if __name__ == "__main__":
    print("=== Optimized Classifier Training ===")
    
    trainer = OptimizedClassifierTrainer(
        sample_fraction=SAMPLE_FRACTION,
        debug_mode=DEBUG_MODE,
        n_jobs=N_JOBS,
        random_state=RANDOM_STATE
    )
    
    models, results = trainer.train_and_evaluate()
    
    print("\nTraining and evaluation complete!")