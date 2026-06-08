# ─────────────────────────────────────────────────────────────────
# WHAT THIS FILE DOES:
# Reads the two CSV files, cleans them, combines movie features
# into a single "tags" text, converts to numbers, computes
# similarity between all movies, then saves everything to disk.
# You only run this file ONCE.
# ─────────────────────────────────────────────────────────────────

# ── STEP A: Import libraries ──────────────────────────────────────
# pandas  → for working with tables (DataFrames)
# numpy   → for numerical operations
# ast     → to safely convert text that looks like Python lists
# nltk    → Natural Language Toolkit (for stemming words)
# pickle  → to save Python objects to disk as files
# sklearn → machine learning tools (vectorizer + similarity)

import pandas as pd
import numpy as np
import ast
import nltk
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import PorterStemmer

# ── STEP B: Download NLTK data ────────────────────────────────────
# PorterStemmer needs this downloaded once
# It reduces words to their root: "running" → "run", "loves" → "love"
nltk.download('punkt')

# Initialize the stemmer
ps = PorterStemmer()


# ══════════════════════════════════════════════════════════════════
# STEP 1: LOAD THE DATA
# ══════════════════════════════════════════════════════════════════
# We have two CSV files:
#   movies CSV  → has plot, genres, keywords, budget, revenue etc.
#   credits CSV → has cast and crew info
# We need to MERGE them into one table using the movie title

print("📂 Loading data...")

movies  = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")

print(f"   Movies file  : {movies.shape[0]} rows, {movies.shape[1]} columns")
print(f"   Credits file : {credits.shape[0]} rows, {credits.shape[1]} columns")

# Merge both DataFrames on the 'title' column
# This adds cast/crew columns to the movies table
movies = movies.merge(credits, on="title")
print(f"   After merge  : {movies.shape[0]} rows, {movies.shape[1]} columns")


# ══════════════════════════════════════════════════════════════════
# STEP 2: KEEP ONLY THE COLUMNS WE NEED
# ══════════════════════════════════════════════════════════════════
# We only need these 6 columns to build recommendations:
#   movie_id  → unique ID (used to fetch posters from TMDB API later)
#   title     → movie name (shown to users)
#   overview  → plot summary (tells us what the movie is about)
#   genres    → action, comedy, drama etc.
#   keywords  → specific themes like "based on novel", "revenge" etc.
#   cast      → actors
#   crew      → director, producer etc.

movies = movies[[
    "movie_id", "title", "overview",
    "genres", "keywords", "cast", "crew"
]]

# Remove rows where any of these are missing/empty
movies.dropna(inplace=True)

print(f"\n✅ After cleanup: {movies.shape[0]} movies remaining")
print("\nSample row:")
print(movies.iloc[0]["title"], "→", movies.iloc[0]["overview"][:80], "...")


# ══════════════════════════════════════════════════════════════════
# STEP 3: UNDERSTAND THE RAW DATA FORMAT
# ══════════════════════════════════════════════════════════════════
# The genres, keywords, cast, crew columns look like this in the CSV:
#
#   genres = '[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}]'
#
# This is a STRING that looks like a Python list of dictionaries.
# We need to CONVERT it into an actual Python list: ["Action", "Adventure"]
# We use ast.literal_eval() to safely parse it.

print("\n📋 Raw genres example:")
print(movies.iloc[0]["genres"])   # shows the raw string format


# ── HELPER FUNCTIONS ──────────────────────────────────────────────

def convert(text):
    """
    Converts a JSON-like string of name-id pairs into a list of names.
    Input:  '[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}]'
    Output: ["Action", "Adventure"]
    """
    names = []
    for item in ast.literal_eval(text):
        names.append(item["name"])
    return names


def convert_cast(text):
    """
    Like convert(), but only takes the FIRST 3 actors.
    We don't want 50 actors — just the leads matter for matching.
    Input:  long list of cast members
    Output: ["Tom Hanks", "Robin Wright", "Gary Sinise"]  (top 3 only)
    """
    names = []
    count = 0
    for item in ast.literal_eval(text):
        if count < 3:
            names.append(item["name"])
            count += 1
        else:
            break
    return names


def get_director(text):
    """
    Searches the crew list for the person whose job is "Director".
    Returns a list with just that one name.
    Input:  long list of crew members with different jobs
    Output: ["Christopher Nolan"]
    """
    for item in ast.literal_eval(text):
        if item["job"] == "Director":
            return [item["name"]]
    return []   # return empty list if no director found


def collapse(name_list):
    """
    Removes spaces from names so multi-word names stay as one token.
    Why? Because CountVectorizer splits on spaces.
    "Sam Worthington" → two separate words: "Sam" and "Worthington"
    "SamWorthington"  → one token that means this specific actor

    Input:  ["Sam Worthington", "Zoe Saldana"]
    Output: ["SamWorthington", "ZoeSaldana"]
    """
    return [name.replace(" ", "") for name in name_list]


def stem(text):
    """
    Reduces each word in a text string to its root form.
    "loved loving loves" all become "love"
    This makes matching better — "action" and "actions" are treated as same.

    Input:  "an adventurous story about running heroes"
    Output: "adventur stori run hero"  (root forms)
    """
    stemmed_words = []
    for word in text.split():
        stemmed_words.append(ps.stem(word))
    return " ".join(stemmed_words)


# ══════════════════════════════════════════════════════════════════
# STEP 4: APPLY THE HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════

print("\n⚙️  Processing features...")

# Convert genres from string → list of genre names
movies["genres"] = movies["genres"].apply(convert)
# Example result: ["Action", "Adventure", "Fantasy"]

# Convert keywords from string → list of keyword names
movies["keywords"] = movies["keywords"].apply(convert)
# Example result: ["based on novel", "hero", "magic"]

# Convert cast from string → list of TOP 3 actor names
movies["cast"] = movies["cast"].apply(convert_cast)
# Example result: ["SamWorthington", "ZoeSaldana", "SigourneyWeaver"]

# Extract just the director from crew
movies["crew"] = movies["crew"].apply(get_director)
# Example result: ["JamesCameron"]

# Split the overview (plot text) into a list of words
movies["overview"] = movies["overview"].apply(
    lambda x: x.split() if isinstance(x, str) else []
)
# Example result: ["A", "paraplegic", "marine", "dispatched", ...]

# Remove spaces from multi-word names in all list columns
movies["genres"]   = movies["genres"].apply(collapse)
movies["keywords"] = movies["keywords"].apply(collapse)
movies["cast"]     = movies["cast"].apply(collapse)
movies["crew"]     = movies["crew"].apply(collapse)

print("   ✅ genres    → done")
print("   ✅ keywords  → done")
print("   ✅ cast      → done")
print("   ✅ crew      → done")
print("   ✅ overview  → done")


# ══════════════════════════════════════════════════════════════════
# STEP 5: CREATE THE "TAGS" COLUMN
# ══════════════════════════════════════════════════════════════════
# We combine ALL our feature lists into one single text string per movie.
# This is called the "tags" — it represents the movie's identity.
#
# For Avatar, tags would look like:
# "Sam dispatch planet resources unobtanium Action Adventure Fantasy
#  SamWorthington ZoeSaldana JamesCameron culture marine soldier ..."
#
# Two movies with similar tags will have similar content → good recommendations.

print("\n🔗 Creating tags column...")

movies["tags"] = (
    movies["overview"] +    # plot words
    movies["genres"] +      # genre names
    movies["keywords"] +    # theme keywords
    movies["cast"] +        # top 3 actors
    movies["crew"]          # director
)

# Join the list into a single lowercase string
movies["tags"] = movies["tags"].apply(
    lambda word_list: " ".join(word_list).lower()
)

# Apply stemming to reduce words to roots
movies["tags"] = movies["tags"].apply(stem)

print("   Sample tags for first movie:")
print("  ", movies.iloc[0]["tags"][:200], "...")


# ══════════════════════════════════════════════════════════════════
# STEP 6: KEEP ONLY WHAT WE NEED FOR THE APP
# ══════════════════════════════════════════════════════════════════
# The app only needs: movie_id, title, tags
# movie_id → to fetch poster from TMDB API
# title    → to display to user
# tags     → for similarity calculation

new_df = movies[["movie_id", "title", "tags"]].reset_index(drop=True)

print(f"\n📊 Final dataset: {len(new_df)} movies")
print(new_df.head(3))


# ══════════════════════════════════════════════════════════════════
# STEP 7: VECTORIZATION — Convert Text to Numbers
# ══════════════════════════════════════════════════════════════════
# Computers can't compare text directly — they need numbers.
# CountVectorizer converts each movie's tags into a vector (list of numbers).
#
# Think of it like this:
# We build a vocabulary of the 5000 most common words across all movies.
# Each movie gets a vector of 5000 numbers:
#   - If the word appears in that movie's tags → the number goes up
#   - If the word doesn't appear → 0
#
# So Avatar's vector might have high numbers for "marine", "planet", "alien"
# And Titanic's vector has high numbers for "ship", "ocean", "love"
#
# max_features=5000 → use only top 5000 words (avoids noise)
# stop_words="english" → ignore common words like "the", "a", "is"

print("\n🔢 Vectorizing tags...")

cv = CountVectorizer(max_features=5000, stop_words="english")
vectors = cv.fit_transform(new_df["tags"]).toarray()

print(f"   Vector shape: {vectors.shape}")
# This will show something like (4800, 5000)
# 4800 movies, each represented by 5000 numbers


# ══════════════════════════════════════════════════════════════════
# STEP 8: COSINE SIMILARITY — Find How Similar Movies Are
# ══════════════════════════════════════════════════════════════════
# Cosine similarity measures the angle between two vectors.
# - Score of 1.0 → identical (same movie)
# - Score of 0.8 → very similar
# - Score of 0.0 → completely different
#
# We compute this for EVERY pair of movies.
# Result: a 4800 × 4800 matrix where:
#   similarity[0][1] = how similar movie 0 is to movie 1
#   similarity[0][100] = how similar movie 0 is to movie 100
#
# When a user picks a movie, we look at its row in this matrix
# and find the highest scores → those are the recommendations.

print("\n📐 Computing cosine similarity...")
print("   (This may take 30-60 seconds for ~4800 movies...)")

similarity = cosine_similarity(vectors)

print(f"   Similarity matrix shape: {similarity.shape}")
print(f"   Sample scores for movie 0: {similarity[0][:5]}")
# First score will be 1.0 (movie compared to itself)


# ══════════════════════════════════════════════════════════════════
# STEP 9: SAVE EVERYTHING TO DISK
# ══════════════════════════════════════════════════════════════════
# We save two files using pickle:
#   movies.pkl      → the cleaned DataFrame (movie_id, title, tags)
#   similarity.pkl  → the 4800×4800 similarity matrix
#
# The Streamlit app will load these files instead of recomputing
# everything each time someone opens the app. Much faster!

print("\n💾 Saving model files...")

pickle.dump(new_df,     open("movies.pkl",     "wb"))
pickle.dump(similarity, open("similarity.pkl", "wb"))

print("   ✅ Saved movies.pkl")
print("   ✅ Saved similarity.pkl")
print("\n🎉 Model building complete! Now run: streamlit run app.py")