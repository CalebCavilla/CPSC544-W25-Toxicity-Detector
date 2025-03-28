import sys
from pathlib import Path
import logging
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import the Regularization class from the package
from regularization.L2Regularization import Regularization

def main():
    print("=== Regularization Execution Test ===")
    
    # Set file paths for the features and target CSV files
    features_file = project_root / "data" / "features_DEMO.csv"
    target_file = project_root / "data" / "train.csv"
    
    # Print file paths for confirmation
    print(f"Using features file: {features_file}")
    print(f"Using target file: {target_file}")
    
    # Initialize the Regularization object with file paths and parameters
    reg = Regularization(
        features_path=str(features_file),
        target_path=str(target_file),
        threshold_value=0.01,
        correlation_threshold=0.4
    )
    
    # Run the full regularization pipeline
    try:
        new_data, mse, best_alpha, best_l1_ratio = reg.run_all()
        print("\nFeature selection complete.")
        print("Test MSE:", mse)
        print("Best alpha:", best_alpha)
        print("Best l1_ratio:", best_l1_ratio)
        print("New data shape:", new_data.shape)
    except Exception as e:
        print(f"Error during regularization execution: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
