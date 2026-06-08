import os
import pandas as pd
import ast
import nltk
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import PorterStemmer

nltk.download('punkt')
nltk.download('punkt_tab')

ps = PorterStemmer()

# ── PATHS — works on Windows and Linux (Streamlit Cloud) ──
BASE           = os.path.dirname(os.path.abspath(__file__))
MOVIES_CSV     = os.path.join(BASE, "tmdb_5000_movies.csv")
CREDITS_CSV    = os.path.join(BASE, "tmdb_5000_credits.csv")
MOVIES_PKL     = os.path.join(BASE, "movies.pkl")
SIMILARITY_PKL = os.path.join(BASE, "similarity.pkl")

print(f"Working folder : {BASE}")
print(f"Movies CSV     : {MOVIES_CSV}")
print(f"Credits CSV    : {CREDITS_CSV}")

# ── CHECK FILES EXIST ─────────────────────────────────────
if not os.path.exists(MOVIES_CSV):
    print(f"❌ Not found: {MOVIES_CSV}")
    exit()
if not os.path.exists(CREDITS_CSV):
    print(f"❌ Not found: {CREDITS_CSV}")
    exit()

print("✅ Both CSV files found")

# ── LOAD ──────────────────────────────────────────────────
print("\nLoading CSV files...")
movies  = pd.read_csv(MOVIES_CSV)
credits = pd.read_csv(CREDITS_CSV)
print(f"  movies  : {movies.shape}")
print(f"  credits : {credits.shape}")

# ── MERGE ─────────────────────────────────────────────────
print("\nMerging...")
movies = movies.merge(credits, on="title")
print(f"  merged  : {movies.shape}")

# ── SELECT COLUMNS ────────────────────────────────────────
# Use 'id' from movies CSV — correct TMDB ID (Avatar = 19995)
# Do NOT use 'movie_id' from credits — has wrong values
movies = movies[["id", "title", "overview", "genres", "keywords", "cast", "crew"]]
movies = movies.rename(columns={"id": "movie_id"})
movies = movies.dropna().reset_index(drop=True)

print(f"\n  Movies after cleanup : {len(movies)}")
print(f"  Sample IDs:")
print(movies[["movie_id", "title"]].head(5).to_string())

avatar = movies[movies["title"] == "Avatar"]["movie_id"].values
print(f"\n  Avatar ID : {avatar}  (should be [19995])")

# ── HELPER FUNCTIONS ──────────────────────────────────────
def convert(text):
    result = []
    try:
        for item in ast.literal_eval(text):
            result.append(item["name"])
    except:
        pass
    return result

def convert_cast(text):
    result = []
    count  = 0
    try:
        for item in ast.literal_eval(text):
            if count < 3:
                result.append(item["name"])
                count += 1
            else:
                break
    except:
        pass
    return result

def get_director(text):
    try:
        for item in ast.literal_eval(text):
            if item["job"] == "Director":
                return [item["name"]]
    except:
        pass
    return []

def collapse(lst):
    return [i.replace(" ", "") for i in lst]

def stem(text):
    return " ".join([ps.stem(w) for w in text.split()])

# ── PROCESS ───────────────────────────────────────────────
print("\nProcessing features...")
movies["genres"]   = movies["genres"].apply(convert)
movies["keywords"] = movies["keywords"].apply(convert)
movies["cast"]     = movies["cast"].apply(convert_cast)
movies["crew"]     = movies["crew"].apply(get_director)
movies["overview"] = movies["overview"].apply(
    lambda x: x.split() if isinstance(x, str) else []
)
movies["genres"]   = movies["genres"].apply(collapse)
movies["keywords"] = movies["keywords"].apply(collapse)
movies["cast"]     = movies["cast"].apply(collapse)
movies["crew"]     = movies["crew"].apply(collapse)
print("  ✅ Done")

# ── TAGS ──────────────────────────────────────────────────
print("\nCreating tags...")
movies["tags"] = (
    movies["overview"] +
    movies["genres"]   +
    movies["keywords"] +
    movies["cast"]     +
    movies["crew"]
)
movies["tags"] = movies["tags"].apply(lambda x: " ".join(x).lower())
movies["tags"] = movies["tags"].apply(stem)
print("  ✅ Done")

# ── FINAL DATAFRAME ───────────────────────────────────────
new_df = movies[["movie_id", "title", "tags"]].reset_index(drop=True)
print(f"\n  Final shape : {new_df.shape}")

# ── VECTORIZE ─────────────────────────────────────────────
print("\nVectorizing...")
cv      = CountVectorizer(max_features=5000, stop_words="english")
vectors = cv.fit_transform(new_df["tags"]).toarray()
print(f"  Vector shape : {vectors.shape}")

# ── COSINE SIMILARITY ─────────────────────────────────────
print("\nComputing similarity (30-60 seconds)...")
similarity = cosine_similarity(vectors)
print(f"  Similarity shape : {similarity.shape}")

# ── SAVE ──────────────────────────────────────────────────
print("\nSaving...")
pickle.dump(new_df,     open(MOVIES_PKL,     "wb"))
pickle.dump(similarity, open(SIMILARITY_PKL, "wb"))
print(f"  ✅ Saved : {MOVIES_PKL}")
print(f"  ✅ Saved : {SIMILARITY_PKL}")

# ── VERIFY ────────────────────────────────────────────────
print("\n--- VERIFICATION ---")
check      = pickle.load(open(MOVIES_PKL, "rb"))
avatar_id  = check[check["title"] == "Avatar"]["movie_id"].values
spectre_id = check[check["title"] == "Spectre"]["movie_id"].values
print(f"  Avatar ID  : {avatar_id}   → expected [19995]")
print(f"  Spectre ID : {spectre_id}  → expected [206647]")

if len(avatar_id) > 0 and int(avatar_id[0]) == 19995:
    print("\n  ✅ ALL CORRECT — run: streamlit run app.py")
else:
    print("\n  ❌ IDs wrong — screenshot and share")