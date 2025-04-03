import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import classification_report, f1_score, precision_recall_curve, auc, roc_curve

os.environ['LOKY_MAX_CPU_COUNT'] = '8'

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Configuration
SAVE_PATH = project_root / "model" / "saved_models"


def load_models_and_metrics():
    """Load saved models and their evaluation metrics."""
    print("=== Loading Models and Metrics ===")

    SAVE_PATH.mkdir(exist_ok=True, parents=True)

    model_files = list(SAVE_PATH.glob("*.joblib"))
    model_files = [f for f in model_files if "_metrics" not in f.name and "all_metrics" not in f.name]
    
    if not model_files:
        print("No saved models found. Please run optimized_classifiers.py first.")
        return None, None
    
    # Try to load metrics
    try:
        metrics_path = SAVE_PATH / "all_metrics.joblib"
        
        if metrics_path.exists():
            all_metrics = joblib.load(metrics_path)
            print(f"Loaded evaluation metrics from {metrics_path.name}")
        else:
            print("No metrics file found. Please run optimized_classifiers.py first.")
            return None, None
    except Exception as e:
        print(f"Error loading metrics: {e}")
        return None, None
    
    # Load models
    models = {}
    for model_file in model_files:
        model_name = model_file.stem
        try:
            print(f"Loading model: {model_name}")
            models[model_name] = joblib.load(model_file)
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
    
    if not models:
        print("No models could be loaded.")
        return None, None
    
    return models, all_metrics


def plot_evaluation_results(metrics):
    """Plot evaluation results for all models."""
    if not metrics:
        print("No metrics available to plot")
        return

    plot_data = {
        'Model': [],
        'Metric': [],
        'Value': []
    }
    
    for model_name, model_metrics in metrics.items():
        for metric_name in ['accuracy', 'f1', 'roc_auc']:
            if metric_name in model_metrics:
                plot_data['Model'].append(model_name.replace('_', ' ').title())
                plot_data['Metric'].append(metric_name.replace('_', ' ').title())
                plot_data['Value'].append(model_metrics[metric_name])
    
    if not plot_data['Model']:
        print("No metrics data available for plotting")
        return
        
    df = pd.DataFrame(plot_data)
    
    # Create a bar plot
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x='Model', y='Value', hue='Metric', data=df)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f')
    
    plt.title('Model Performance Comparison')
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.ylim(0, 1.1)  # Add space for labels
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(SAVE_PATH / "model_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    plot_confusion_matrices(metrics)


def plot_confusion_matrices(metrics):
    """Plot confusion matrices for all models."""
    if not metrics:
        return
    
    # Count models with confusion matrices
    models_with_cm = [name for name, model_metrics in metrics.items() 
                     if 'confusion_matrix' in model_metrics]
    
    if not models_with_cm:
        print("No confusion matrices available to plot")
        return

    fig, axes = plt.subplots(1, len(models_with_cm), figsize=(5*len(models_with_cm), 5))
    if len(models_with_cm) == 1:
        axes = [axes]

    for ax, model_name in zip(axes, models_with_cm):
        cm = metrics[model_name]['confusion_matrix']
        
        # Create heatmap
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues', 
            cbar=False,
            ax=ax
        )
        
        ax.set_title(f"{model_name.replace('_', ' ').title()}\nF1: {metrics[model_name]['f1']:.4f}")
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_xticklabels(['Not Toxic', 'Toxic'])
        ax.set_yticklabels(['Not Toxic', 'Toxic'])
    
    plt.tight_layout()
    plt.savefig(SAVE_PATH / "confusion_matrices.png", dpi=300, bbox_inches='tight')
    plt.show()


def optimize_threshold(models, metrics, X_test, y_test):
    """Find optimal threshold to maximize F1 score for each model and return the best."""
    best_threshold = 0.5
    best_f1 = 0.0
    best_model_name = None
    best_model = None
    
    # Try each model to find the one with the best optimized threshold
    for model_name, model in models.items():
        try:
            # Get probability predictions
            y_probs = model.predict_proba(X_test)[:, 1]

            # Calculate precision, recall, and F1 for different thresholds
            precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)

            # Calculate F1 scores for each threshold
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)

            # Find threshold with max F1
            optimal_idx = np.argmax(f1_scores)
            optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
            model_best_f1 = f1_scores[optimal_idx]
            
            print(f"Model {model_name}: Optimal threshold: {optimal_threshold:.3f} with F1 score: {model_best_f1:.3f}")
            
            # Update if this is the best model+threshold combination
            if model_best_f1 > best_f1:
                best_f1 = model_best_f1
                best_threshold = optimal_threshold
                best_model_name = model_name
                best_model = model
                
        except Exception as e:
            print(f"Error optimizing threshold for {model_name}: {e}")
            continue
    
    if best_model is not None:
        print(f"\nBest model found: {best_model_name} with optimal threshold: {best_threshold:.3f} and F1 score: {best_f1:.3f}")
        
        # Calculate error counts for best model
        y_probs = best_model.predict_proba(X_test)[:, 1]
        y_pred = (y_probs >= best_threshold).astype(int)
        fp = np.sum((y_test == 0) & (y_pred == 1))
        fn = np.sum((y_test == 1) & (y_pred == 0))
        print(f"At threshold={best_threshold:.3f}: False Positives: {fp}, False Negatives: {fn}")
        
        # Plot precision-recall curve for best model
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
        optimal_idx = np.argmax(f1_scores)
        
        plt.figure(figsize=(10, 6))
        plt.plot(recalls, precisions, 'b-', label='Precision-Recall curve')
        plt.plot(recalls[optimal_idx], precisions[optimal_idx], 'ro',
                 label=f'Optimal threshold: {best_threshold:.3f}, F1: {best_f1:.3f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {best_model_name}')
        plt.legend(loc='best')
        plt.grid(True)
        plt.savefig(SAVE_PATH / "threshold_optimization.png", dpi=300, bbox_inches='tight')
        plt.show()

        # Plot ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, 'b-', label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {best_model_name}')
        plt.legend(loc='best')
        plt.grid(True)
        plt.savefig(SAVE_PATH / "roc_curve.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        return best_threshold, best_f1, best_model_name, best_model
    else:
        print("Could not find optimal threshold for any model")
        return 0.5, None, None, None


def demo_model_usage(threshold=0.5, best_model_name=None, best_f1=None):
    """Demonstrate how to use the best model for prediction with adjustable threshold."""
    print(f"\n=== Model Usage Demo (Threshold: {threshold:.3f}) ===")

    # Load models and metrics
    models, metrics = load_models_and_metrics()
    if not models or not metrics:
        return
    
    # Find best model based on F1 score
    try:
        # If we have a best model name from threshold optimization, use it
        if best_model_name and best_model_name in models:
            best_model = models.get(best_model_name)
            best_score = best_f1 if best_f1 else metrics[best_model_name]['f1']
        else:
            # Fall back to the original method
            best_model_name = max(metrics.items(), key=lambda x: x[1]['f1'])[0]
            best_model = models.get(best_model_name)
            best_score = metrics[best_model_name]['f1']
        
        print(f"Best model found: {best_model_name} with F1 score: {best_score:.4f}")
    except (KeyError, ValueError) as e:
        print(f"Error finding best model: {e}")
        return
    
    # Load features and examples for prediction
    try:
        features_path = project_root / "data" / "features_train.csv"
        if not features_path.exists():
            print(f"Features file not found at {features_path}")
            return
            
        # Load a small sample of the data
        features_df = pd.read_csv(features_path, nrows=1000)
        df_train = pd.read_csv(project_root / "data" / "train.csv", nrows=1000)
        
        # Ensure the number of samples match
        min_samples = min(len(features_df), len(df_train))
        features_df = features_df.iloc[:min_samples]
        df_train = df_train.iloc[:min_samples]
        
        # Get a mix of toxic and non-toxic examples for demonstration
        toxic_indices = df_train[df_train['toxic'] == 1].index[:3]
        non_toxic_indices = df_train[df_train['toxic'] == 0].index[:3]
        sample_indices = list(toxic_indices) + list(non_toxic_indices)
        
        # Show a few examples
        print("\nSample predictions:")
        for idx in sample_indices:
            if idx >= len(features_df):
                continue
                
            features = features_df.iloc[[idx]]
            text = df_train.iloc[idx]['comment_text']
            
            # Truncate long texta
            if len(text) > 100:
                display_text = text[:100] + "..."
            else:
                display_text = text

            # Get probability and apply custom threshold
            probability = best_model.predict_proba(features)[0][1]
            prediction = 1 if probability >= threshold else 0

            # Get true label
            actual_label = df_train.iloc[idx]['toxic']

            print(f"\nExample {sample_indices.index(idx) + 1}:")
            print(f"Text: {display_text}")
            print(f"Prediction: {'Toxic' if prediction == 1 else 'Not Toxic'}")
            print(f"Probability of being toxic: {probability:.4f}")
            print(f"Actual label: {'Toxic' if actual_label == 1 else 'Not Toxic'}")
            
    except Exception as e:
        print(f"Error in model demo: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nModel usage demo complete.")


def threshold_evaluation(model, X_test, y_test):
    """Evaluate model performance across different thresholds."""
    try:
        y_probs = model.predict_proba(X_test)[:, 1]

        thresholds = np.arange(0.1, 0.7, 0.05)
        results = []

        for threshold in thresholds:
            y_pred = (y_probs >= threshold).astype(int)
            f1 = f1_score(y_test, y_pred)

            # Calculate metrics
            tn = np.sum((y_test == 0) & (y_pred == 0))
            fp = np.sum((y_test == 0) & (y_pred == 1))
            fn = np.sum((y_test == 1) & (y_pred == 0))
            tp = np.sum((y_test == 1) & (y_pred == 1))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0

            results.append({
                'threshold': threshold,
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'false_positives': fp,
                'false_negatives': fn
            })

        # Convert to DataFrame for plotting
        df = pd.DataFrame(results)

        # Plot metrics vs threshold
        plt.figure(figsize=(12, 6))
        plt.plot(df['threshold'], df['f1'], 'b-', marker='o', label='F1 Score')
        plt.plot(df['threshold'], df['precision'], 'g-', marker='s', label='Precision')
        plt.plot(df['threshold'], df['recall'], 'r-', marker='^', label='Recall')
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title('Performance Metrics vs Classification Threshold')
        plt.legend()
        plt.grid(True)
        plt.savefig(SAVE_PATH / "threshold_metrics.png", dpi=300, bbox_inches='tight')
        plt.show()

        # Plot false positives and negatives
        plt.figure(figsize=(12, 6))
        plt.plot(df['threshold'], df['false_positives'], 'r-', marker='o', label='False Positives')
        plt.plot(df['threshold'], df['false_negatives'], 'b-', marker='s', label='False Negatives')
        plt.xlabel('Threshold')
        plt.ylabel('Count')
        plt.title('Error Types vs Classification Threshold')
        plt.legend()
        plt.grid(True)
        plt.savefig(SAVE_PATH / "threshold_errors.png", dpi=300, bbox_inches='tight')
        plt.show()

        # Print optimal threshold for different metrics
        max_f1_idx = df['f1'].idxmax()
        balance_idx = np.abs(df['false_positives'] - df['false_negatives']).idxmin()

        print(
            f"Optimal threshold for F1 score: {df.iloc[max_f1_idx]['threshold']:.3f} (F1: {df.iloc[max_f1_idx]['f1']:.3f})")
        print(f"Threshold for balanced errors: {df.iloc[balance_idx]['threshold']:.3f}")
        print(f"Threshold for minimizing false negatives: {df['threshold'].min():.3f}")

        return df

    except Exception as e:
        print(f"Error in threshold evaluation: {e}")
        return None


def main(custom_threshold=None):
    """Main function for evaluating optimized models."""
    print("=== Optimized Model Evaluation ===")
    
    # Load models and their evaluation metrics
    models, metrics = load_models_and_metrics()
    if not models or not metrics:
        return
    
    # Plot evaluation results
    print("\nPlotting evaluation results...")
    plot_evaluation_results(metrics)

    # Load test data for threshold optimization
    features_path = project_root / "data" / "features_train.csv"
    target_path = project_root / "data" / "train.csv"

    if features_path.exists() and target_path.exists():
        try:
            print("\nLoading test data for threshold optimization...")
            X_test = pd.read_csv(features_path, nrows=5000)
            y_df = pd.read_csv(target_path, nrows=5000)
            y_test = y_df['toxic']

            print("\nFinding optimal threshold...")
            optimal_threshold, best_f1, best_model_name, best_model = optimize_threshold(models, metrics, X_test, y_test)
            
            # Use custom threshold if provided, otherwise use the optimized one
            threshold = custom_threshold if custom_threshold is not None else optimal_threshold

            # Demo with optimal threshold and best model
            demo_model_usage(threshold=threshold, best_model_name=best_model_name, best_f1=best_f1)
        except Exception as e:
            print(f"Error during threshold optimization: {e}")
            # Fall back to default threshold
            demo_model_usage()
    else:
        # Use default threshold if data isn't available
        demo_model_usage()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate toxicity detection models')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Custom classification threshold (default: auto-optimize)')
    args = parser.parse_args()

    main(custom_threshold=args.threshold)