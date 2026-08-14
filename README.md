 # Movie Recommender System (Content-Based)

Recommends similar movies based on plot, genre, cast, and crew — using the 
TMDB 5000 Movies dataset. Given a movie title, returns the top 5 most 
similar movies.

## Problem
Recommend movies similar to a given title, using only movie metadata 
(no user ratings/behavior data) — a content-based filtering approach.

## Data & Preprocessing
- TMDB 5000 Movies + Credits datasets, merged on title
- Kept: movie_id, title, overview, genres, keywords, cast (top 3), crew (director)
- Cleaned nulls, parsed nested JSON-like columns (genres/keywords/cast/crew) 
  into plain lists
- Combined all fields into a single "tags" string per movie, with weighted 
  importance: genres ×3, keywords ×2, cast ×2, crew ×2, overview ×1 — so 
  genre/cast/crew matter more than plot-word overlap
- Applied stemming (PorterStemmer) to merge word variants (e.g. "loved"/"loving" → "love")

## Approach
1. **Baseline:** Bag-of-words via CountVectorizer (top 5000 words, stop words 
   removed) → cosine similarity between movie vectors
2. **Upgrade:** Replaced bag-of-words with SBERT sentence embeddings 
   (all-MiniLM-L6-v2) → cosine similarity on 384-dim vectors, capturing 
   sentence-level meaning instead of just word overlap

## Note on ML Type
This is **unsupervised learning** — there's no labeled target (no "correct" 
recommendation to predict). Cosine similarity just measures distance between 
movie vectors; it's not trained against ground-truth labels the way a 
classifier or regressor would be. Supervised methods (e.g. XGBoost, logistic 
regression) don't apply here, since there's no y variable to predict.

## Result
`recommend("movie title")` returns the top 5 most similar movies by cosine 
similarity. Verified qualitatively — e.g. genre overlap checked between 
query movie and its recommendations.

## Future Improvements
- Collaborative filtering (SVD/ALS) if user rating data becomes available — 
  captures "people who liked X also liked Y," which content-based similarity 
  can't learn on its own
- Hybrid system combining content-based + collaborative filtering, as 
  production systems (e.g. Netflix) do
- Quantitative evaluation (e.g. genre-overlap score across many queries) 
  instead of spot-checking individual recommendations

## Scope
This is a content-based recommender only — it has no user data, so it can't 
personalize to individual taste, only find movies with similar attributes. 
It's a learning/portfolio project, not a production recommendation system.
