from svd_from_scratch import SVD
import numpy as np






# 1. Define Raw Dataset: 6 Users and 5 Movies
# Columns: [The Matrix, Interstellar, Star Wars, Toy Story, Coco]
# 0 represents an unrated movie (missing data)
ratings_matrix = np.array([
    [5, 4, 5, 0, 0],  # User 0: Hardcore Sci-Fi Fan
    [0, 5, 4, 1, 0],  # User 1: Sci-Fi Fan who dislikes Animation
    [0, 0, 1, 4, 5],  # User 2: Animation Fan
    [1, 0, 0, 5, 4],  # User 3: Animation Fan who dislikes Sci-Fi
    [4, 0, 5, 4, 0],  # User 4: Likes both, missing some ratings
    [0, 2, 0, 0, 5],  # User 5: Outlier / Unique taste
])

movies = ["The Matrix", "Interstellar", "Star Wars", "Toy Story", "Coco"]
num_users, num_movies = ratings_matrix.shape

print("--- Original Sparse Ratings Matrix ---")
print(ratings_matrix)

# 2. Matrix Normalization & Missing Value Imputation from Scratch
# We calculate the average rating for each movie, ignoring the zeros.
filled_matrix = ratings_matrix.copy().astype(float)
movie_means = np.zeros(num_movies)

for j in range(num_movies):
    valid_ratings = ratings_matrix[:, j][ratings_matrix[:, j] > 0]
    if len(valid_ratings) > 0:
        movie_means[j] = np.mean(valid_ratings)
    else:
        movie_means[j] = 2.5 # Default fallback if movie has no ratings at all
        
    # Replace the 0s in this column with the movie's average rating
    zero_indices = (ratings_matrix[:, j] == 0)
    filled_matrix[zero_indices, j] = movie_means[j]

print("\n--- Imputed Ratings Matrix (Gaps Filled with Column Means) ---")
print(np.round(filled_matrix, 2))

# 3. Fit the Custom SVD (Using k=2 Latent Tastes)
k_latent_features = 2
svd_rec = SVD(n_components=k_latent_features)
svd_rec.fit(filled_matrix)

# 4. Reconstruct the Matrix to Generate Predictions
# This computes U_k @ diag(S_k) @ VT_k under the hood
predicted_matrix = svd_rec.reconstruct()

print("\n--- Complete Predicted Ratings Matrix (Continuous Space) ---")
print(np.round(predicted_matrix, 2))

# 5. Recommendation Generator Engine
def get_top_recommendations(user_id, num_rec=1):
    print(f"\n=========================================")
    print(f"Generating Recommendations for User #{user_id}")
    print(f"=========================================")
    
    # Extract movies this specific user has already seen in reality
    user_original_profile = ratings_matrix[user_id, :]
    user_predicted_profile = predicted_matrix[user_id, :]
    
    already_watched_indices = np.where(user_original_profile > 0)[0]
    unwatched_indices = np.where(user_original_profile == 0)[0]
    
    print("Already Watched:")
    for idx in already_watched_indices:
        print(f" - {movies[idx]}: Rated {user_original_profile[idx]}/5")
        
    if len(unwatched_indices) == 0:
        print("User has watched everything! No new recommendations available.")
        return
        
    # Gather predictions only for unseen content
    predictions_for_unseen = [(idx, user_predicted_profile[idx]) for idx in unwatched_indices]
    
    # Sort by highest predicted rating value descending
    predictions_for_unseen.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop Predicted Recommendations:")
    for rank, (movie_idx, pred_score) in enumerate(predictions_for_unseen[:num_rec]):
        # Clip score mathematically to remain within normal bounds [1.0, 5.0]
        final_score = np.clip(pred_score, 1.0, 5.0)
        print(f" Rank {rank+1}: {movies[movie_idx]} (Predicted Rating: {final_score:.2f}/5)")

# 6. Test the Recommendation Results on specific target profiles
get_top_recommendations(user_id=0, num_rec=2) # Sci-Fi fan who hasn't seen animations
get_top_recommendations(user_id=2, num_rec=2) # Animation fan who hasn't seen Sci-Fi