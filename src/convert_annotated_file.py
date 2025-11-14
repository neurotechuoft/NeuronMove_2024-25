import sys
from pathlib import Path
import scipy.io as sio 
import numpy as np
import pickle
from typing import Dict, Any, List

# --- Module Setup ---
# Add project root to the Python path to allow importing 'config'
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config as cfg

# --- Configuration ---
ANNOTATION_DIR = cfg.ANNOTATION_DATA 

def extract_and_save_annotations(filename: str) -> None:
    """
    Loads annotations from a .mat file, extracts the structured array 
    under the 'labels' key, and saves it as a .pkl file.
    """
    mat_file_path = ANNOTATION_DIR / filename
    
    # Generate output file name
    pkl_filename = filename.replace('.mat', '.pkl')
    pkl_file_path = ANNOTATION_DIR / pkl_filename

    print("="*70)
    print(f"STARTING EXTRACTION: {filename}")
    
    if not mat_file_path.exists():
        print(f"WARNING: Input file not found: {mat_file_path}. Skipping.")
        return

    try:
        # 1. Load the data using scipy.io.loadmat
        data = sio.loadmat(mat_file_path)
        
        # 2. Extract the main structured array. 
        # The exploration step confirmed this is always keyed as 'labels'.
        if 'labels' in data:
            labels_struct = data['labels']
            
            # The structured array is usually wrapped in a (1, N) array in MATLAB imports.
            # We access the first and only element which contains the struct array.
            
            # 3. Convert the structured array into a list of dictionaries for easier use
            # We must convert the inner objects to NumPy arrays if they are still MATLAB references
            
            # Flatten the (1, N) array to a 1D iterable of participant structures
            # And convert the structured NumPy array to a standard list of Python dicts
            
            # We use value.flat[0] for the HC/PD structure, then we can convert it to a list of dicts.
            
            # Since the outer shape is (1, N) and Dtype is complex, we flatten it
            # and then convert the structured numpy array (void type) into a list of dictionaries.
            
            # NOTE: If the data structure is a simple struct array:
            # annotation_list = [labels_struct[0, i] for i in range(labels_struct.shape[1])]
            
            # Given the (1, N) structure, we can usually just read the entire array and process.
            
            # Let's extract the array and save the raw structured NumPy array, which is clean enough
            # and preserves the dtype information.
            
            # The variable we want to save is the structured array itself:
            # We squeeze it to remove the single-dimension of size 1 if present.
            labels_array = np.squeeze(labels_struct)
            
            # The final data to save is just the structured NumPy array
            data_to_save: Dict[str, Any] = {
                'labels': labels_array,
                'num_participants': labels_array.size
            }
            
            print(f"  - Successfully extracted '{filename}'.")
            print(f"  - Array Shape (squeezed): {labels_array.shape}")
            print(f"  - Participant Count: {labels_array.size}")
            
            # 4. Save the data
            with open(pkl_file_path, 'wb') as f:
                pickle.dump(data_to_save, f)

            print(f"  Data successfully saved to: {pkl_file_path}")
            
        else:
            print("  ERROR: Key 'labels' not found in the .mat file. Skipping save.")
            
    except Exception as e:
        print(f"   An error occurred during processing: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    
    # 1. Define the input files to process
    filenames = [
        'labels_HC_phys.mat',
        'labels_PD_phys_tremor.mat'
    ]
    
    # 2. Process and save each file separately
    for filename in filenames:
        extract_and_save_annotations(filename)

    print("\nProcessing complete for all annotation files.")
    print("="*70)