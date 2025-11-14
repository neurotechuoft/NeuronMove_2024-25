import sys
from pathlib import Path
import pandas as pd # Essential for DataFrame operations
import pickle

# --- Module Setup ---
# Add project root to the Python path to allow importing 'config'
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config as cfg

# --- Configuration ---
# Assuming ANNOTATION_DATA is correctly defined in your config.py
ANNOTATION_DIR = cfg.ANNOTATION_DATA 

def load_and_view_annotations(filename: str):
    """
    Loads a pickled annotation file, converts the structured array into a 
    Pandas DataFrame, prints key structural information, and saves it as a CSV file.
    """
    pkl_filename = filename
    pkl_file_path = ANNOTATION_DIR / pkl_filename
    
    # Define CSV output path
    csv_filename = pkl_filename.replace('.pkl', '.csv')
    csv_file_path = ANNOTATION_DIR / csv_filename
    
    print("="*70)
    print(f"LOADING AND VIEWING: {filename}")
    
    if not pkl_file_path.exists():
        print(f"ERROR: Pickle file not found at: {pkl_file_path}. Please run src/extract_annotations.py first.")
        return

    try:
        # 1. Load the data from the pickle file
        with open(pkl_file_path, 'rb') as f:
            loaded_data = pickle.load(f)
            
        # The 'labels' key holds the structured NumPy array
        labels_array = loaded_data['labels']
        
        # 2. Convert the structured NumPy array into a Pandas DataFrame
        df = pd.DataFrame(labels_array)
        
        # 3. Save the DataFrame to a CSV file (index=False prevents adding an extra column)
        df.to_csv(csv_file_path, index=False)
        
        # 4. Display the DataFrame structure in the console
        print("\n--- DataFrame Head (First 5 Participants) ---")
        print(df.head())
        
        print("\n--- DataFrame Information (Data Types and Non-Null Counts) ---")
        df.info()

        print(f"\n✅ Annotation data saved as scrollable CSV file to: {csv_file_path}")

    except Exception as e:
        print(f"❌ An error occurred during loading or processing: {e}")
    
# --- Main Execution ---
if __name__ == "__main__":
    
    # We will demonstrate with the PD annotations as they have the most fields
    target_file = 'labels_PD_phys_tremor.pkl'
    
    # NOTE: You can change the target_file to 'labels_HC_phys.pkl' to view the controls data.
    
    load_and_view_annotations(target_file)
    print("="*70)