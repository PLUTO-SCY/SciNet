import json
import os
import numpy as np
from tqdm import tqdm
from scipy.stats import percentileofscore

# --- 1. Configuration ---

# The directory containing your reference disruption score files
REFERENCE_SCORES_DIR = "results/disruption"

# A cache file to store the loaded distribution for much faster subsequent runs
REFERENCE_CACHE_FILE = "results/disruption_distribution.npy"

# Your specific list of scores to be transformed
SCORES_TO_TRANSFORM = [-0.389, -0.489, -0.071, -0.153, 0.045, 0.062, 0.119, 0.135]


# --- 2. Helper Functions ---

def build_reference_distribution(directory_path, cache_file):
    """
    Loads disruption scores from all JSON files in a directory.
    Uses a cache file to avoid reloading on subsequent runs.
    """
    if os.path.exists(cache_file):
        print(f"Loading reference score distribution from cache: {cache_file}")
        return np.load(cache_file)

    print(f"Building reference score distribution from directory: {directory_path}")
    all_scores = []
    
    # Ensure the directory exists
    if not os.path.isdir(directory_path):
        raise FileNotFoundError(f"The specified directory does not exist: {directory_path}")
        
    file_list = [f for f in os.listdir(directory_path) if f.endswith('.json')]
    
    for filename in tqdm(file_list, desc="Reading reference files"):
        filepath = os.path.join(directory_path, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    # IMPORTANT: Reading 'disruption_score' as specified
                    score = item.get("disruption_score")
                    if score is not None:
                        all_scores.append(score)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not read or parse {filename}. Error: {e}")
            
    reference_scores = np.array(all_scores)
    if reference_scores.size == 0:
        raise ValueError("No disruption scores were found in the specified directory.")
        
    print(f"Built distribution with {len(reference_scores)} scores. Caching to {cache_file}...")
    np.save(cache_file, reference_scores)
    return reference_scores


def transform_score_by_percentile(raw_score, reference_scores):
    """
    Transforms a raw score to a 0-10 scale using percentile ranking.
    For disruption, higher is better, so we rank the score directly.
    """
    if raw_score is None:
        return None
    # `percentileofscore` calculates the percentage of scores in the reference set
    # that are less than the given raw_score.
    percentile = percentileofscore(reference_scores, raw_score)
    # Scale from 0-100 to 0-10
    return percentile / 10.0


# --- 3. Main Calculation ---

def main():
    """
    Main function to execute the transformation.
    """
    # Step 1: Build or load the reference distribution from the 'disruption' folder
    try:
        reference_scores = build_reference_distribution(REFERENCE_SCORES_DIR, REFERENCE_CACHE_FILE)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: Could not build reference distribution. {e}")
        return

    # Step 2: Transform your list of scores
    transformed_scores = []
    print("\n--- Transforming Your Scores (0-10 Scale) ---")
    
    # Sort the scores for a cleaner presentation
    for raw_score in sorted(SCORES_TO_TRANSFORM):
        transformed = transform_score_by_percentile(raw_score, reference_scores)
        transformed_scores.append(transformed)
        print(f"Raw Score: {raw_score:<10.4f} => Transformed Score: {transformed:.3f}")

    # Step 3: Calculate the final average of the transformed scores
    if transformed_scores:
        final_average = np.mean(transformed_scores)
        print("\n" + "="*50)
        print(f"📊 Final Average of Transformed Disruption Scores: {final_average:.4f}")
        print("="*50)
    else:
        print("Could not calculate the final average.")

if __name__ == "__main__":
    main()