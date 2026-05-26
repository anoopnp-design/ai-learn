import numpy as np
import pandas as pd
from typing import Dict, Any, List

class DataPreprocessingPipeline:
    """
    Domain service responsible for vector data transformation.
    Handles data validation, shape manipulation, and feature scaling.
    """
    def __init__(self):
        # Hardcoded scaling parameters for our dataset (simulating a saved scaler object)
        # Means and Standard Deviations for: [sepal_length, sepal_width, petal_length, petal_width]
        self.feature_means = np.array([5.84, 3.05, 3.75, 1.19])
        self.feature_stds = np.array([0.83, 0.43, 1.76, 0.76])

    def payload_to_dataframe(self, data: Dict[str, Any]) -> pd.DataFrame:
        """
        Converts an incoming raw schema dictionary into an isolated 
        in-memory, column-oriented Pandas DataFrame.
        """
        # Equivalent to instantiating an internal data table collection
        df = pd.DataFrame([data])
        return df

    def scale_features_vectorized(self, df: pd.DataFrame) -> np.ndarray:
        """
        Applies mathematical Z-score standardization: (X - mean) / std.
        Executes across contiguous C-arrays using vectorization instead of loops.
        """
        # Ensure correct feature alignment order
        ordered_features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        raw_matrix = df[ordered_features].to_numpy() # Extracts data into a pure NumPy array
        
        # Vectorized scaling calculation: Executes across the entire matrix at once
        scaled_matrix = (raw_matrix - self.feature_means) / self.feature_stds
        return scaled_matrix

    def process_batch(self, batch_data: List[Dict[str, Any]]) -> np.ndarray:
        """
        Demonstrates matrix scaling across a large block of entities simultaneously.
        """
        df = pd.DataFrame(batch_data)
        return self.scale_features_vectorized(df)

# Self-executing structural unit test when run directly via the CLI
if __name__ == "__main__":
    print("[Pipeline Test] Initializing pipeline serialization...")
    pipeline = DataPreprocessingPipeline()
    
    # Mocking an unstructured payload input
    mock_payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    
    df = pipeline.payload_to_dataframe(mock_payload)
    print(f"[Pipeline Test] Pandas DataFrame Structure:\n{df}\n")
    
    scaled_vector = pipeline.scale_features_vectorized(df)
    print(f"[Pipeline Test] Resulting Normalized NumPy Vector Matrix:\n{scaled_vector}")