import os
import sys
from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Configuration
USE_FULL_DATA = True  # Set to False for faster testing
DEBUG_MODE = False    # Set to True for reduced parameter search 
SAMPLE_FRACTION = 0.5 if USE_FULL_DATA else 0.1  # Use 50% of data for full training, 10% for testing
N_JOBS = -1          # Use all available cores
RANDOM_STATE = 42    # For reproducibility
SAVE_PATH = project_root / "model" / "saved_models"

# Create the save directory if it doesn't exist
os.makedirs(SAVE_PATH, exist_ok=True)

class OptimizedClassifierTrainer:
    """Class to train and optimize classifiers for toxic comment detection."""
    
    def __init__(self, sample_fraction=SAMPLE_FRACTION, debug_mode=DEBUG_MODE, n_jobs=N_JOBS, random_state=RANDOM_STATE):
        """
        Initialize the trainer.
        
        Args:
            sample_fraction: Fraction of data to use (0.0-1.0)
            debug_mode: If True, use smaller parameter spaces for faster testing
            n_jobs: Number of CPU cores to use (-1 for all)
            random_state: Random seed for reproducibility
        """
        self.sample_fraction = sample_fraction
        self.debug_mode = debug_mode
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.models = {}
        self.feature_importances = {}
        self.results = {}
    
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
        
        # Define parameter space
        if self.debug_mode:
            param_dist = {
                'n_estimators': [50, 100],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'max_features': ['sqrt'],
                'bootstrap': [True],
                'class_weight': ['balanced', 'balanced_subsample', None]
            }
        else:
            param_dist = {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [None, 10, 20, 30, 40],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None],
                'bootstrap': [True, False],
                'class_weight': ['balanced', 'balanced_subsample', None]
            }
        
        base_rf = RandomForestClassifier(random_state=self.random_state)

        # Make sure there are enough samples
        min_samples = max(1000, int(len(y_train) * 0.1))  # At least 1000 samples or 10% of data
        
        halving_search = HalvingRandomSearchCV(
            base_rf,
            param_distributions=param_dist,
            factor=3,           # Reduce candidates by a factor of 3 at each iteration
            resource='n_samples',  # Use fraction of samples for efficiency
            min_resources=min_samples,  # Start with enough samples for reliable CV
            max_resources='auto',  # Automatically determine max resources
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
            param_dist = {
                'n_estimators': [50, 100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7, 9],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'subsample': [0.7, 0.8, 0.9, 1.0],
                'max_features': ['sqrt', 'log2', None]
            }
        
        base_gb = GradientBoostingClassifier(random_state=self.random_state)
        min_samples = max(1000, int(len(y_train) * 0.1))
        
        halving_search = HalvingRandomSearchCV(
            base_gb,
            param_distributions=param_dist,
            factor=3,           # Reduce candidates by a factor of 3 at each iteration
            resource='n_samples',  # Use fraction of samples for efficiency
            min_resources=min_samples,  # Start with enough samples for reliable CV
            max_resources='auto',  # Automatically determine max resources
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
        
        # Negative values -> 0
        X_train_nb = X_train.copy()
        X_train_nb[X_train_nb < 0] = 0

        param_dist = {
            'alpha': np.logspace(-3, 1, 10),
            'fit_prior': [True, False]
        }
        
        base_nb = MultinomialNB()
        min_samples = max(1000, int(len(y_train) * 0.1))  # At least 1000 samples or 10% of data
        
        halving_search = HalvingRandomSearchCV(
            base_nb,
            param_distributions=param_dist,
            factor=3,           # Reduce candidates by a factor of 3 at each iteration
            resource='n_samples',  # Use fraction of samples for efficiency
            min_resources=min_samples,  # Start with enough samples for reliable CV
            max_resources='auto',  # Automatically determine max resources
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
        
        return best_nb
    
    def create_voting_classifier(self):
        """
        Create a voting classifier from optimized models.
        
        Returns:
            Voting classifier
        """
        if len(self.models) < 2:
            raise ValueError("At least two optimized models are required to create a voting classifier")
        
        print("\nCreating Voting Classifier...")
        
        # Dont use naive bayes since it gives error for some reason
        filtered_models = {name: model for name, model in self.models.items() 
                          if 'naive_bayes' not in name}

        # Create a list of tuples (name, model) for the voting classifier
        estimators = [(name, model) for name, model in filtered_models.items()]

        voting_clf = VotingClassifier(
            estimators=estimators,
            voting='soft',
            n_jobs=self.n_jobs
        )
        
        print(f"Voting Classifier created with models: {[name for name, _ in estimators]}")
        
        return voting_clf
    
    def evaluate_model(self, model, X_test, y_test, model_name='model'):
        """
        Evaluate a model on test data.
        
        Args:
            model: Trained model to evaluate
            X_test: Test feature matrix
            y_test: Test target vector
            model_name: Name of the model for reporting
            
        Returns:
            Dictionary of evaluation metrics
        """
        print(f"\nEvaluating {model_name}...")
        
        # Handle Naive Bayes
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
            
            ax.set_title(f"{model_name.replace('_', ' ').title()}\nAccuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
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
    
    def train_and_evaluate(self):
        """Complete pipeline for training and evaluating models."""
        # Load data
        X, y = self.load_data()
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        # Optimize and evaluate Random Forest
        rf_model = self.optimize_random_forest(X_train, y_train)
        self.evaluate_model(rf_model, X_test, y_test, 'random_forest')
        
        # Optimize and evaluate Gradient Boosting
        gb_model = self.optimize_gradient_boosting(X_train, y_train)
        self.evaluate_model(gb_model, X_test, y_test, 'gradient_boosting')
        
        # Optimize and evaluate Naive Bayes
        nb_model = self.optimize_multinomial_nb(X_train, y_train)
        self.evaluate_model(nb_model, X_test, y_test, 'naive_bayes')
        
        # Create and evaluate voting classifier
        voting_clf = self.create_voting_classifier()
        voting_clf.fit(X_train, y_train)
        self.models['voting'] = voting_clf
        self.evaluate_model(voting_clf, X_test, y_test, 'voting')
        
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
