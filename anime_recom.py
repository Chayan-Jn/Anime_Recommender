import pandas as pd
import numpy as np
import re
import sys
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

# Load the dataset (fulldata) using the relative path to the same directory
fulldata = pd.read_csv('./fulldata.csv')  # Use the relative path to the CSV file

# Function to clean anime titles by removing unwanted keywords (like season numbers, etc.)
def clean_title(title):
    """
    Cleans up anime titles by removing known sequel/season/variant indicators.
    Uses a keyword list for easy customization.
    """
    title = title.lower()

   
    keywords_to_remove = [
        "season", "movie", "part", "ova", "special", "final", "complete",
        "collection", "edition", "chapter", "arc", "episode", "ep", "vol",
        "volume", "tv", "the", "1st", "2nd", "3rd", "4th", "5th", "6th", "7th",
        "8th", "9th", "0th", "first", "second", "third", "fourth", "fifth",
        "sixth", "seventh", "eighth", "ninth", "tenth"
    ]

    # Build regex pattern: \b(keyword)\b to match whole words
    pattern = r'\b(?:' + '|'.join(map(re.escape, keywords_to_remove)) + r')\b'
    title = re.sub(pattern, '', title)

    # Remove "S1", "S2", ..., "S9" type season indicators
    title = re.sub(r'\bs\d+\b', '', title)

    # Remove extra whitespace
    title = re.sub(r'\s+', ' ', title).strip()

    return title

# Function to check if the candidate anime is similar to any anime already in the list
def check_similar(candidate, current_list):
    """
    Check if the candidate anime is similar to any anime already in the list.
    Uses case-insensitive substring matching as a proxy for similarity.
    
    Returns:
        bool: True if similar anime found in list, False otherwise
    """
    candidate_lower = clean_title(candidate)
    
    for existing in current_list:
        existing_lower = clean_title(existing)
        if candidate_lower == existing_lower or \
           candidate_lower in existing_lower or \
           existing_lower in candidate_lower:
            return True
    return False

# Main function for getting anime recommendations
def three(anime_name, n=10):
    """
    1. Filters by genre similarity (cosine)
    2. Refines with KNN on user ratings
    """
    try:
        # === 1. Prepare Data ===
        # Get unique anime with stats
        anime_stats = fulldata.groupby(['anime_id', 'name', 'genre']).agg(
            avg_rating=('user_rating', 'mean'),
            rating_count=('user_rating', 'count')
        ).reset_index()
        
        # Create user-item rating matrix
        rating_matrix = fulldata.pivot_table(
            index='user_id',
            columns='name',
            values='user_rating',
            fill_value=0
        ).T

        # === 2. Find Target Anime ===
        # Case-insensitive exact match first
        target_mask = anime_stats['name'].str.lower() == anime_name.lower()
        if not target_mask.any():
            # Fallback to partial match
            target_mask = anime_stats['name'].str.contains(anime_name, case=False)
            if not target_mask.any():
                return pd.DataFrame({"Error": [f"'{anime_name}' not found"]})
        
        target_idx = target_mask.idxmax()
        target_name = anime_stats.loc[target_idx, 'name']
        
        # === 3. Genre Similarity Filter ===
        # One-hot encode genres
        genres = anime_stats['genre'].str.get_dummies(', ')
        genre_sim = cosine_similarity(genres)
        
        # Get top 100 genre-similar anime (excluding target)
        sim_indices = np.argsort(genre_sim[target_idx])[::-1][1:1001]
        candidate_df = anime_stats.iloc[sim_indices]
        
        # === 4. KNN Refinement ===
        # Filter to candidates available in rating matrix
        valid_candidates = [name for name in candidate_df['name'] 
                          if name in rating_matrix.index]
        
        if not valid_candidates:
            return pd.DataFrame({"Error": ["No valid candidates for KNN"]})
        
        # Fit KNN with cosine distance
        knn = NearestNeighbors(n_neighbors=min(n+1, len(valid_candidates)), 
                             metric='correlation')
        knn.fit(rating_matrix.loc[valid_candidates])
        
        # Get neighbors (use proxy if target not in matrix)
        if target_name not in rating_matrix.index:
            distances, indices = knn.kneighbors(
                rating_matrix.loc[[valid_candidates[0]]],
                n_neighbors=n
            )
        else:
            distances, indices = knn.kneighbors(
                rating_matrix.loc[[target_name]],
                n_neighbors=n+1
            )
        
        # === 5. Strict Recommendation Filtering ===
        rec_names = []
        target_lower = target_name.lower()
        
        for idx in indices[0]:
            candidate = valid_candidates[idx]
            candidate_lower = candidate.lower()
            
            # Original strict checks against the input anime
            if (candidate_lower == target_lower) or \
               (target_lower in candidate_lower) or \
               (candidate_lower in target_lower):
                continue
            
            # New check: skip if similar to anything already in rec_names
            if check_similar(candidate, rec_names):
                continue
            
            rec_names.append(candidate)
            
            if len(rec_names) >= n:
                break

        
        # === 6. Prepare Output ===
        if not rec_names:
            return pd.DataFrame({"Error": ["No valid recommendations after filtering"]})
        
        results = anime_stats[anime_stats['name'].isin(rec_names)][
            ['name', 'genre', 'avg_rating']
        ].sort_values('avg_rating', ascending=False)
        
        # Add genre similarity percentage
        results['Genre Match'] = [
            f"{genre_sim[target_idx][anime_stats[anime_stats['name']==name].index[0]]:.0%}"
            for name in results['name']
        ]
        
        # Format columns and index
        results.columns = ['Anime', 'Genres', 'Rating', 'Genre Match']
        results.index = np.arange(1, len(results)+1)
        results.index.name = 'Rank'
        
        # Return results
        return results

    except Exception as e:
        return pd.DataFrame({
            "Error": ["System error"],
            "Details": [str(e)]
        })


if __name__ == "__main__":
    # Get the anime name from command-line arguments
    if len(sys.argv) < 2:
        print("Usage: python script.py <anime_name>")
        sys.exit(1)
    
    anime_name = sys.argv[1]  # First command-line argument
    
    # Call the `three` function with the anime name and print results
    recommendations = three(anime_name)
    
    # Print the recommendations or error
    if isinstance(recommendations, pd.DataFrame):
        if "Error" in recommendations.columns:
            print(f"Error: {recommendations['Error'].iloc[0]}")
        else:
            print(recommendations)
    else:
        print("Error generating recommendations")
