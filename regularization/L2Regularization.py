import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from typing import Optional

class Regularization:
    """
    A class to perform regularization and feature selection on machine learning data.
    
    This class loads feature and target datasets, performs correlation analysis,
    tests for normality, plots feature distributions, and applies an ElasticNet
    model (with grid search) for feature selection by culling near-zero coefficients.
    """
    
    def __init__(self, features_path, target_path, threshold_value=0.01, correlation_threshold=0.4, random_state=42):
        """
        Initialize the Regularization class.
        
        Args:
            features_path (str): Path to the CSV file containing the feature data.
            target_path (str): Path to the CSV file containing the target values.
            threshold_value (float): Coefficient threshold below which features are culled.
            correlation_threshold (float): Absolute correlation threshold for filtering.
            random_state (int): Random seed for reproducibility.
        """
        self.features_path: str = features_path
        self.target_path: str = target_path
        self.threshold_value: float = threshold_value
        self.correlation_threshold: float = correlation_threshold
        self.random_state: int = random_state
        self.data: Optional[pd.DataFrame] = None
        self.target: Optional[pd.Series] = None
        self.best_model: Optional[ElasticNet] = None
        
        # Set up logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logging.info("Regularization class initialized.")

    def load_data(self):
        """Load features and target data from CSV files. 
        TODO: Currently only loads targets with toxic comments"""
        logging.info("Loading feature data from %s", self.features_path)
        self.data = pd.read_csv(self.features_path)
        logging.info("Loading target data from %s", self.target_path)
        targets = pd.read_csv(self.target_path)
        self.target = targets["toxic"]
        logging.info("Data loaded successfully. Features shape: %s, Target shape: %s", self.data.shape, self.target.shape)
        print("Features shape:", self.data.shape)  # Debug: Print shape of features
        print("Target shape:", self.target.shape)  # Debug: Print shape of targets

    
    def correlation_heatmap(self):
        """Compute the correlation matrix, filter it by a threshold, and display a heatmap."""
        if self.data is None:
            raise ValueError("Data is not loaded. Please call load_data() before running correlation_heatmap.")
        logging.info("Computing correlation matrix.")
        corr_matrix = self.data.corr()
        
        # Only keep correlations with an absolute value above the threshold
        mask = np.abs(corr_matrix) < self.correlation_threshold
        filtered_corr = corr_matrix.where(~mask, np.nan)
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(filtered_corr, cmap="coolwarm", fmt=".2f", linewidths=0.5)
        plt.title(f"Filtered Correlation Heatmap (|r| > {self.correlation_threshold})")
        plt.show()
        logging.info("Correlation heatmap plotted.")
    
    def normality_tests(self):
        """
        Perform Shapiro-Wilk normality tests on all features.
        
        Returns:
            dict: A dictionary with feature names as keys and test results as values.
        """
        if self.data is None:
            raise ValueError("Data is not loaded. Please call load_data() before running normality_tests.")
        logging.info("Performing normality tests on features using the Shapiro-Wilk test.")
        results = {}
        for feature in self.data.columns:
            stat, p = shapiro(self.data[feature])
            if p > 0.05:
                result = "normally distributed"
            else:
                result = f"NOT normally distributed (p={p:.4f})"
            results[feature] = result
            logging.info("Feature %s: %s", feature, result)
        return results
    
    def plot_feature_distributions(self):
        """Plot the KDE distribution of each feature by target class."""
        logging.info("Plotting feature distributions with KDE by target class.")
        if self.data is None:
            raise ValueError("Data is not loaded. Please call load_data() before using plot_feature_distributions.")
        num_features = len(self.data.columns)
        cols = 5
        rows = int(np.ceil(num_features / cols))
        fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(15, rows * 3))
        axes = axes.flatten()
        
        for i, feature in enumerate(self.data.columns):
            sns.kdeplot(data=self.data, x=feature, hue=self.target, common_norm=False, ax=axes[i])
            axes[i].set_title(f"Distribution of {feature}")
        
        # Remove any extra axes
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
            
        plt.tight_layout()
        plt.show()
        logging.info("Feature distribution plots generated.")

    def feature_selection(self, sample_fraction=1.0):
        """
        Perform feature selection using ElasticNet with GridSearchCV.

        Splits the data, searches for the best hyperparameters, evaluates performance,
        and culls features with near-zero coefficients.

        Returns:
            tuple: (new_data, mse, best_alpha, best_l1_ratio)
        """
        logging.info("Starting feature selection using ElasticNet and GridSearchCV.")
        if self.data is None:
            raise ValueError("Data is not loaded. Please call load_data() before running feature_selection.")

        # Subsample the data for faster processing
        if sample_fraction < 1.0:
            data_sample, _, target_sample, _ = train_test_split(
                self.data, self.target,
                train_size=sample_fraction,
                random_state=self.random_state,
                stratify=self.target
            )
            logging.info(
                f"Using {sample_fraction * 100:.1f}% of data ({len(data_sample)} samples) for feature selection")
        else:
            data_sample, target_sample = self.data, self.target

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(data_sample, target_sample, test_size=0.2, random_state=self.random_state)
        logging.info("Data split into training and testing sets.")

        # Define the parameter grid for hyperparameter tuning
        param_grid = {
            'alpha': [0.01, 0.1, 1, 10, 100],
            'l1_ratio': [0.1, 0.5, 0.7, 1.0]
        }
        logging.info("Parameter grid: %s", param_grid)

        # Initialize the ElasticNet model and grid search
        elasticnet = ElasticNet(max_iter=10000, random_state=self.random_state)
        grid_search = GridSearchCV(estimator=elasticnet, param_grid=param_grid,
                                   cv=5, scoring='neg_mean_squared_error',
                                   verbose=1, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        logging.info("Grid search completed.")

        best_alpha = grid_search.best_params_['alpha']
        best_l1_ratio = grid_search.best_params_['l1_ratio']
        self.best_model = grid_search.best_estimator_
        logging.info("Best parameters found: alpha=%s, l1_ratio=%s", best_alpha, best_l1_ratio)

        # Evaluate the best model on the test set
        y_pred = self.best_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        logging.info("Test MSE: %s", mse)
        logging.info("Best model coefficients: %s", self.best_model.coef_)

        # Cull features whose coefficients are below the threshold value
        cull_mask = np.abs(self.best_model.coef_) < self.threshold_value
        logging.info("Culling features with coefficient absolute value below %s", self.threshold_value)
        new_data = self.data.loc[:, ~cull_mask]
        logging.info("Features before culling: %d, after culling: %d", self.data.shape[1], new_data.shape[1])

        return new_data, mse, best_alpha, best_l1_ratio
    
    def run_all(self):
        """
        Run the complete regularization and feature selection pipeline.
        
        Returns:
            tuple: (new_data, mse, best_alpha, best_l1_ratio)
        """
        logging.info("Running the full regularization pipeline.")
        self.load_data()
        self.correlation_heatmap()
        self.normality_tests()
        self.plot_feature_distributions()
        new_data, mse, best_alpha, best_l1_ratio = self.feature_selection()
        logging.info("Regularization and feature selection pipeline completed.")
        return new_data, mse, best_alpha, best_l1_ratio

if __name__ == "__main__":
    # Initialize the Regularization object with file paths and parameters
    reg = Regularization(features_path="features_DEMO.csv", target_path="train.csv", 
                         threshold_value=0.01, correlation_threshold=0.4)
    
    # Run the pipeline
    new_data, mse, best_alpha, best_l1_ratio = reg.run_all()
    
    # Final logging and output
    logging.info("Feature selection complete.")
    print("Feature selection complete.")
    print("Test MSE:", mse)
    print("Best alpha:", best_alpha)
    print("Best l1_ratio:", best_l1_ratio)
    print("New data shape:", new_data.shape)
