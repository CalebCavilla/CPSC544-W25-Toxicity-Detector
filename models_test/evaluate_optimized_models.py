import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Configuration
SAVE_PATH = project_root / "models_test" / "saved_models"

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

def demo_model_usage():
    """Demonstrate how to use the best model for prediction."""
    print("\n=== Model Usage Demo ===")
    
    # Load models and metrics
    models, metrics = load_models_and_metrics()
    if not models or not metrics:
        return
    
    # Find best model based on F1 score
    try:
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

            prediction = best_model.predict(features)[0]
            probability = best_model.predict_proba(features)[0][1]
            
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

def main():
    """Main function for evaluating optimized models."""
    print("=== Optimized Model Evaluation ===")
    
    # Load models and their evaluation metrics
    models, metrics = load_models_and_metrics()
    if not models or not metrics:
        return
    
    # Plot evaluation results
    print("\nPlotting evaluation results...")
    plot_evaluation_results(metrics)
    
    # Demonstrate model usage
    demo_model_usage()

if __name__ == "__main__":
    main()
