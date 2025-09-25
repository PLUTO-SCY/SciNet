import json
import os
import numpy as np
from tqdm import tqdm
from scipy.stats import percentileofscore

# --- 1. Configuration ---

# Directory containing the reference novelty scores to build the distribution
REFERENCE_SCORES_DIR = "/data5/shaochenyang/AI_Scientist/OpenAlex/task1Result/CNovelty_multiprocess_5000_final"

# Cache file for the reference distribution to speed up subsequent runs
REFERENCE_CACHE_FILE = "novelty_distribution.npy"

# Your provided list of average novelty scores
YOUR_AVERAGE_SCORES = [-9.7092, -12.0309, -15.3292, -22.1099, -31.7734, -33.1668, -39.5253, -42.3264]


# --- 2. Helper Functions ---

def build_reference_distribution(directory_path, cache_file):
    """
    Loads novelty scores from all JSON files in a directory.
    Uses a cache file to avoid reloading on subsequent runs.
    """
    if os.path.exists(cache_file):
        print(f"Loading reference score distribution from cache: {cache_file}")
        return np.load(cache_file)

    print(f"Building reference score distribution from: {directory_path}")
    all_scores = []
    file_list = [f for f in os.listdir(directory_path) if f.endswith('.json')]
    
    for filename in tqdm(file_list, desc="Reading reference files"):
        filepath = os.path.join(directory_path, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    score = item.get("novelty_score")
                    if score is not None:
                        all_scores.append(score)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not read or parse {filename}. Error: {e}")
            
    reference_scores = np.array(all_scores)
    print(f"Built distribution with {len(reference_scores)} scores. Caching to {cache_file}...")
    np.save(cache_file, reference_scores)
    return reference_scores


def transform_score_by_percentile(raw_score, reference_scores_negated):
    """Transforms a raw score to a 0-10 scale using percentile ranking."""
    if raw_score is None:
        return None
    # By using negated scores, we correctly rank lower (more negative) raw scores as better.
    percentile = percentileofscore(reference_scores_negated, -raw_score)
    return percentile / 10.0


# --- 3. Main Calculation ---

def main():
    # Step 1: Build or load the reference distribution
    # This is necessary to understand the context of your scores.
    reference_scores = build_reference_distribution(REFERENCE_SCORES_DIR, REFERENCE_CACHE_FILE)
    # Negate once for efficiency in the transformation function
    reference_scores_negated = reference_scores * -1

    # Step 2: Transform your list of scores
    transformed_scores = []
    print("\n--- Transforming Your Scores (0-10 Scale) ---")
    for raw_score in YOUR_AVERAGE_SCORES:
        transformed = transform_score_by_percentile(raw_score, reference_scores_negated)
        transformed_scores.append(transformed)
        print(f"Raw Score: {raw_score:<10.4f} => Transformed Score: {transformed:.4f}")

    # Step 3: Calculate the final average of the transformed scores
    if transformed_scores:
        final_average = np.mean(transformed_scores)
        print("\n" + "="*50)
        print(f"📊 Final Average of Transformed Novelty Scores: {final_average:.4f}")
        print("="*50)
    else:
        print("Could not calculate the final average.")

if __name__ == "__main__":
    main()