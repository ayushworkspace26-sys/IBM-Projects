

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

np.random.seed(42)
pd.set_option("display.max_colwidth", None)


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------
def load_data(data_dir="data"):
    
    movies_path = os.path.join(data_dir, "movies.csv")
    ratings_path = os.path.join(data_dir, "ratings.csv")

    if os.path.exists(movies_path) and os.path.exists(ratings_path):
        movies_raw = pd.read_csv(movies_path)
        ratings_raw = pd.read_csv(ratings_path)

        movies = movies_raw.rename(columns={"movieId": "movie_id"})[
            ["movie_id", "title", "genres"]
        ].copy()
        ratings = ratings_raw.rename(
            columns={"movieId": "movie_id", "userId": "user_id"}
        )[["user_id", "movie_id", "rating"]].copy()

        # Keep runtime reasonable for a minor project if the real dataset is large
        max_users = 300
        if ratings["user_id"].nunique() > max_users:
            keep_users = np.random.choice(ratings["user_id"].unique(), max_users, replace=False)
            ratings = ratings[ratings["user_id"].isin(keep_users)]
            movies = movies[movies["movie_id"].isin(ratings["movie_id"])].reset_index(drop=True)

        print(f"Loaded real dataset from '{data_dir}/' "
              f"({len(movies)} movies, {len(ratings)} ratings, "
              f"{ratings['user_id'].nunique()} users).")
        return movies, ratings, "Kaggle: grouplens/movielens-latest-small"

    print(f"No movies.csv/ratings.csv found in '{data_dir}/' — generating a synthetic demo dataset.")
    return _generate_synthetic_data()


def _generate_synthetic_data(n_users=60):
    
    movies_data = [
        ("The Great Heist", "Action|Crime|Thriller", "A group of skilled thieves plan the ultimate bank heist in a race against time."),
        ("Starlight Voyage", "Sci-Fi|Adventure", "A crew of explorers travels beyond the solar system to find a new home for humanity."),
        ("Whispering Woods", "Horror|Mystery", "A small town is haunted by a legend that begins to come true."),
        ("Love in Autumn", "Romance|Drama", "Two strangers meet during a rainy autumn and slowly fall in love."),
        ("Laugh Riot", "Comedy", "A chaotic office party spirals out of control in one unforgettable night."),
        ("Iron Legacy", "Action|Sci-Fi", "A retired super-soldier is pulled back into action to stop a global threat."),
        ("The Silent Witness", "Thriller|Mystery|Crime", "A witness to a murder must stay hidden while helping solve the case."),
        ("Galaxy Quest Reborn", "Sci-Fi|Comedy|Adventure", "A washed-up space crew gets a second chance to save the galaxy."),
        ("Hearts Unspoken", "Romance|Drama", "A long-distance couple fights to keep their relationship alive."),
        ("Dead of Night", "Horror", "A group of friends are trapped in a cabin as something sinister hunts them."),
        ("The Last Sprint", "Drama|Sports", "An aging athlete trains for one final chance at Olympic gold."),
        ("Kingdom of Ash", "Fantasy|Adventure|Action", "A young warrior must reclaim her kingdom from an ancient evil."),
        ("Office Chaos", "Comedy|Drama", "A group of mismatched coworkers try to survive a company merger."),
        ("Deep Blue Mystery", "Mystery|Thriller|Adventure", "A marine biologist uncovers a conspiracy beneath the ocean's surface."),
        ("Neon Nights", "Action|Crime|Sci-Fi", "In a cyberpunk city, a detective hunts a rogue AI mastermind."),
        ("The Wedding Plan", "Romance|Comedy", "A wedding planner falls for the groom's best man days before the big event."),
        ("Shadow Realm", "Fantasy|Horror", "A cursed forest hides a doorway to a realm of nightmares."),
        ("Rocket Summer", "Adventure|Comedy|Family", "Two kids build a homemade rocket to prove aliens are real."),
        ("The Boardroom", "Drama|Thriller", "A corporate takeover reveals secrets that could destroy an empire."),
        ("Midnight Symphony", "Drama|Romance|Music", "A struggling violinist gets one shot to perform at a prestigious concert hall."),
        ("Frozen Pursuit", "Action|Thriller", "An arctic expedition turns deadly when a team is hunted across the ice."),
        ("Comic Relief", "Comedy", "A stand-up comedian's disastrous week becomes his best material yet."),
        ("The Vanishing Point", "Mystery|Sci-Fi|Thriller", "People start disappearing without a trace in a quiet coastal town."),
        ("Warrior's Path", "Action|Adventure|Fantasy", "A lone warrior journeys across a war-torn land to fulfil an ancient prophecy."),
        ("Sunset Boulevard Dreams", "Drama|Romance", "An aspiring actress navigates love and ambition in the city of dreams."),
        ("The Prank War", "Comedy|Family", "Two rival families escalate a prank war that gets hilariously out of hand."),
        ("Echoes of War", "Drama|War|History", "A soldier's letters home reveal the true cost of a forgotten battle."),
        ("Alien Horizon", "Sci-Fi|Horror|Thriller", "A research station on a distant moon loses contact after an alien signal."),
        ("The Getaway Car", "Action|Comedy|Crime", "A getaway driver and a rookie thief must survive their first job together."),
        ("Paper Hearts", "Romance|Drama|Comedy", "A journaling app connects two strangers who fall for each other's words."),
    ]
    movies = pd.DataFrame(movies_data, columns=["title", "genres", "description"])
    movies.insert(0, "movie_id", range(1, len(movies) + 1))

    all_genres = sorted(set(g for gl in movies["genres"] for g in gl.split("|")))
    user_favorite_genre = {u: np.random.choice(all_genres) for u in range(1, n_users + 1)}

    rows = []
    for user_id in range(1, n_users + 1):
        fav_genre = user_favorite_genre[user_id]
        n_ratings = np.random.randint(6, 16)
        rated_movies = np.random.choice(movies["movie_id"], size=n_ratings, replace=False)
        for movie_id in rated_movies:
            genres = movies.loc[movies.movie_id == movie_id, "genres"].values[0]
            base = 4.0 if fav_genre in genres.split("|") else 3.0
            rating = np.clip(np.round(np.random.normal(base, 0.8) * 2) / 2, 0.5, 5.0)
            rows.append((user_id, movie_id, rating))
    ratings = pd.DataFrame(rows, columns=["user_id", "movie_id", "rating"])

    print(f"Generated synthetic dataset ({len(movies)} movies, {len(ratings)} ratings, {n_users} users).")
    return movies, ratings, "Synthetic (fallback)"


# ---------------------------------------------------------------------------
# 2. Content-based filtering
# ---------------------------------------------------------------------------
def build_content_similarity(movies):
    """TF-IDF over genres (+ description if available) -> cosine similarity matrix."""
    genre_text = movies["genres"].str.replace("|", " ", regex=False)
    if "description" in movies.columns:
        movies["combined_features"] = genre_text + " " + movies["description"]
    else:
        movies["combined_features"] = genre_text

    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies["combined_features"])
    content_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return content_sim


def recommend_content_based(movies, content_sim, title, top_n=5):
    """Recommend movies similar in content to `title`."""
    if title not in movies["title"].values:
        return f"'{title}' not found in the dataset."

    idx = movies.index[movies["title"] == title][0]
    sim_scores = list(enumerate(content_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [s for s in sim_scores if s[0] != idx][:top_n]

    movie_indices = [i for i, _ in sim_scores]
    scores = [round(s, 3) for _, s in sim_scores]

    result = movies.iloc[movie_indices][["title", "genres"]].copy()
    result["similarity_score"] = scores
    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Collaborative filtering (user-based)
# ---------------------------------------------------------------------------
def build_user_item_matrix(ratings):
    return ratings.pivot_table(index="user_id", columns="movie_id", values="rating").fillna(0)


def build_user_similarity(user_item_matrix):
    user_sim = cosine_similarity(user_item_matrix)
    return pd.DataFrame(user_sim, index=user_item_matrix.index, columns=user_item_matrix.index)


def recommend_collaborative(movies, ratings, user_item_matrix, user_sim_df, user_id, top_n=5, n_neighbors=10):
    """Recommend movies for `user_id` based on similar users' ratings."""
    if user_id not in user_item_matrix.index:
        return f"User {user_id} not found."

    similar_users = user_sim_df[user_id].drop(user_id).sort_values(ascending=False).head(n_neighbors)
    already_rated = set(ratings.loc[ratings.user_id == user_id, "movie_id"])

    weighted_scores = pd.Series(dtype=float)
    for other_user, sim_score in similar_users.items():
        if sim_score <= 0:
            continue
        other_ratings = user_item_matrix.loc[other_user]
        weighted_scores = weighted_scores.add(other_ratings * sim_score, fill_value=0)

    weighted_scores = weighted_scores.drop(labels=already_rated, errors="ignore")
    weighted_scores = weighted_scores.sort_values(ascending=False).head(top_n)

    result = movies.set_index("movie_id").loc[weighted_scores.index, ["title", "genres"]].copy()
    result["predicted_score"] = weighted_scores.values.round(3)
    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4. Hybrid recommendation
# ---------------------------------------------------------------------------
def recommend_hybrid(movies, ratings, content_sim, user_item_matrix, user_sim_df,
                      user_id, liked_title, top_n=5, alpha=0.5):
    """
    Blend content-based and collaborative scores.
    alpha=1.0 -> pure content-based, alpha=0.0 -> pure collaborative.
    """
    if liked_title not in movies["title"].values:
        return f"'{liked_title}' not found in the dataset."
    if user_id not in user_item_matrix.index:
        return f"User {user_id} not found."

    idx = movies.index[movies["title"] == liked_title][0]
    content_scores = pd.Series(content_sim[idx], index=movies["movie_id"])
    content_scores = (content_scores - content_scores.min()) / (content_scores.max() - content_scores.min() + 1e-9)

    similar_users = user_sim_df[user_id].drop(user_id).sort_values(ascending=False).head(10)
    collab_scores = pd.Series(0.0, index=movies["movie_id"])
    for other_user, sim_score in similar_users.items():
        if sim_score > 0:
            collab_scores = collab_scores.add(user_item_matrix.loc[other_user] * sim_score, fill_value=0)
    if collab_scores.max() > 0:
        collab_scores = (collab_scores - collab_scores.min()) / (collab_scores.max() - collab_scores.min() + 1e-9)

    already_rated = set(ratings.loc[ratings.user_id == user_id, "movie_id"])
    hybrid_scores = (alpha * content_scores + (1 - alpha) * collab_scores).drop(labels=already_rated, errors="ignore")
    hybrid_scores = hybrid_scores.drop(labels=[movies.loc[idx, "movie_id"]], errors="ignore")
    hybrid_scores = hybrid_scores.sort_values(ascending=False).head(top_n)

    result = movies.set_index("movie_id").loc[hybrid_scores.index, ["title", "genres"]].copy()
    result["hybrid_score"] = hybrid_scores.values.round(3)
    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 5. Evaluation
# ---------------------------------------------------------------------------
def evaluate_recall_at_n(ratings, user_item_matrix, n=10, min_ratings_per_user=5):
    """Recall@N: was a held-out rating recovered in the user's top-N recommendations?"""
    hits, total = 0, 0
    eval_users = ratings.groupby("user_id").filter(lambda x: len(x) >= min_ratings_per_user)["user_id"].unique()

    for user_id in eval_users:
        user_ratings = ratings[ratings.user_id == user_id]
        test_row = user_ratings.sample(1, random_state=int(user_id))
        test_movie_id = test_row["movie_id"].values[0]

        train_matrix = user_item_matrix.copy()
        train_matrix.loc[user_id, test_movie_id] = 0

        sims = cosine_similarity(train_matrix)
        sims_df = pd.DataFrame(sims, index=train_matrix.index, columns=train_matrix.index)
        neighbors = sims_df[user_id].drop(user_id).sort_values(ascending=False).head(10)

        scores = pd.Series(0.0, index=train_matrix.columns)
        for other_user, sim_score in neighbors.items():
            if sim_score > 0:
                scores = scores.add(train_matrix.loc[other_user] * sim_score, fill_value=0)

        already_rated = set(user_ratings.movie_id) - {test_movie_id}
        scores = scores.drop(labels=already_rated, errors="ignore")
        top_n_movies = scores.sort_values(ascending=False).head(n).index

        total += 1
        if test_movie_id in top_n_movies:
            hits += 1

    return hits / total if total else 0.0


# ---------------------------------------------------------------------------
# 6. Main / demo
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Movie Recommendation System (content-based + collaborative + hybrid)")
    parser.add_argument("--data-dir", default="data", help="Folder containing movies.csv and ratings.csv (default: ./data)")
    parser.add_argument("--movie", default=None, help="Movie title for content-based / hybrid recommendations")
    parser.add_argument("--user", type=int, default=None, help="User id for collaborative / hybrid recommendations")
    parser.add_argument("--top-n", type=int, default=5, help="Number of recommendations to return")
    parser.add_argument("--alpha", type=float, default=0.5, help="Hybrid blend weight (1=content-only, 0=collaborative-only)")
    args = parser.parse_args()

    movies, ratings, dataset_source = load_data(args.data_dir)
    print(f"\nDataset source: {dataset_source}")
    print(f"Movies: {len(movies)} | Ratings: {len(ratings)} | Users: {ratings.user_id.nunique()}\n")

    content_sim = build_content_similarity(movies)
    user_item_matrix = build_user_item_matrix(ratings)
    user_sim_df = build_user_similarity(user_item_matrix)

    movie_title = args.movie or movies["title"].iloc[0]
    user_id = args.user if args.user is not None else int(user_item_matrix.index[0])

    print(f"=== Content-based recommendations for fans of '{movie_title}' ===")
    print(recommend_content_based(movies, content_sim, movie_title, top_n=args.top_n))

    print(f"\n=== Collaborative recommendations for user {user_id} ===")
    print(recommend_collaborative(movies, ratings, user_item_matrix, user_sim_df, user_id, top_n=args.top_n))

    print(f"\n=== Hybrid recommendations (alpha={args.alpha}) ===")
    print(recommend_hybrid(movies, ratings, content_sim, user_item_matrix, user_sim_df,
                            user_id, movie_title, top_n=args.top_n, alpha=args.alpha))

    print("\n=== Evaluation ===")
    recall = evaluate_recall_at_n(ratings, user_item_matrix, n=10)
    print(f"Recall@10 on held-out ratings: {recall:.2%}")


if __name__ == "__main__":
    main()
