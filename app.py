# ─────────────────────────────────────────────────────────────────
# WHAT THIS FILE DOES:
# This is the Streamlit web application.
# It loads the saved model files (movies.pkl, similarity.pkl),
# shows a search box, and displays movie recommendations with
# posters fetched live from the TMDB API.
# ─────────────────────────────────────────────────────────────────

import streamlit as st   # builds the web UI
import pickle            # loads our saved model files
import requests          # makes HTTP calls to TMDB API for posters


# ══════════════════════════════════════════════════════════════════
# STEP 1: PAGE CONFIGURATION
# ══════════════════════════════════════════════════════════════════
# This must be the FIRST streamlit command in the file.
# Sets the browser tab title, icon, and layout.

st.set_page_config(
    page_title="CineMatch — Movie Recommender",
    page_icon="🎬",
    layout="wide"      # uses full browser width
)


# ══════════════════════════════════════════════════════════════════
# STEP 2: PASTE YOUR TMDB API KEY HERE
# ══════════════════════════════════════════════════════════════════
# Get it from: themoviedb.org → Settings → API
# It looks like: "a1b2c3d4e5f6..."
# Replace the placeholder below with your actual key.

TMDB_API_KEY = "d261bd3fbed3ebd2c0bc38075839b3e5"


# ══════════════════════════════════════════════════════════════════
# STEP 3: LOAD THE MODEL FILES
# ══════════════════════════════════════════════════════════════════
# @st.cache_resource means: load these files ONCE and keep in memory.
# Without this, Streamlit reloads them on every user interaction — slow!

@st.cache_resource
def load_model():
    """Load saved pickle files from disk."""
    movies_df  = pickle.load(open("C:\\DA projects\Movie Recommendation System with Web App\\movie_recommender\\movies.pkl",     "rb"))
    similarity = pickle.load(open("C:\\DA projects\Movie Recommendation System with Web App\\movie_recommender\\similarity.pkl", "rb"))
    return movies_df, similarity

# Call the function — loads movies DataFrame and similarity matrix
movies_df, similarity = load_model()


# ══════════════════════════════════════════════════════════════════
# STEP 4: TMDB API FUNCTIONS
# ══════════════════════════════════════════════════════════════════

def fetch_poster(movie_id):
    """
    Given a TMDB movie_id, returns the URL of the movie poster image.

    How it works:
    1. We call the TMDB API endpoint for that movie
    2. The response contains a 'poster_path' like "/1E5baAaEse26fej7uHcjOgEE2t2.jpg"
    3. We prepend the base URL to get the full image URL
    4. If anything fails, we return a placeholder image

    Example API response:
    {
      "title": "Avatar",
      "poster_path": "/jRXYjXNq0Cs2TcJjLkki24MLp7u.jpg",
      "vote_average": 7.2,
      ...
    }
    """
    try:
        url = (
            f"https://api.themoviedb.org/3/movie/{movie_id}"
            f"?api_key={TMDB_API_KEY}&language=en-US"
        )
        response = requests.get(url, timeout=5)
        data     = response.json()

        poster_path = data.get("poster_path", "")

        if poster_path:
            # TMDB stores images at this base URL
            return f"https://image.tmdb.org/t/p/w500{poster_path}"
        else:
            return "https://via.placeholder.com/500x750?text=No+Poster"

    except Exception:
        # If API call fails for any reason, show placeholder
        return "https://via.placeholder.com/500x750?text=Error"


def fetch_details(movie_id):
    """
    Fetches extra details about a movie: rating, overview, release year.
    Returns a dictionary with these fields (or empty dict if API fails).
    """
    try:
        url = (
            f"https://api.themoviedb.org/3/movie/{movie_id}"
            f"?api_key={TMDB_API_KEY}&language=en-US"
        )
        response = requests.get(url, timeout=5)
        return response.json()
    except Exception:
        return {}


# ══════════════════════════════════════════════════════════════════
# STEP 5: THE RECOMMENDATION FUNCTION
# ══════════════════════════════════════════════════════════════════

def recommend(movie_title, num_recommendations=6):
    """
    Given a movie title, returns the top-N most similar movies.

    How it works:
    1. Find the index (row number) of the selected movie in our DataFrame
    2. Look up that row in the similarity matrix → get similarity scores
       for ALL other movies
    3. Sort those scores from highest to lowest
    4. Skip index 0 (that's the movie itself, similarity = 1.0)
    5. Return the top-N movie names and their TMDB IDs

    Example:
      Input : "Avatar"
      Output: ["Guardians of the Galaxy", "Star Trek", "Interstellar", ...]

    Parameters:
      movie_title        : string, the movie the user selected
      num_recommendations: how many similar movies to return (default 6)

    Returns:
      names : list of recommended movie titles
      ids   : list of corresponding TMDB movie IDs (for poster fetching)
    """

    # Find the row index of the selected movie
    # .index[0] gets the first (and only) match
    try:
        movie_index = movies_df[
            movies_df["title"] == movie_title
        ].index[0]
    except IndexError:
        # Movie not found in our dataset
        return [], []

    # Get similarity scores for this movie against ALL movies
    # This is one row of the similarity matrix: 4800 numbers
    # Each number = how similar that movie is to our selected movie
    similarity_scores = list(enumerate(similarity[movie_index]))
    # Format: [(0, 1.0), (1, 0.23), (2, 0.18), (3, 0.45), ...]
    # (index, similarity_score)

    # Sort by similarity score, highest first
    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],   # sort by the score (second element)
        reverse=True           # highest first
    )

    # Skip index 0 (the movie itself has score 1.0 — perfect match)
    # Take the next num_recommendations movies
    top_movies = similarity_scores[1 : num_recommendations + 1]

    # Extract movie names and IDs
    names = []
    ids   = []
    for idx, score in top_movies:
        names.append(movies_df.iloc[idx]["title"])
        ids.append(movies_df.iloc[idx]["movie_id"])

    return names, ids


# ══════════════════════════════════════════════════════════════════
# STEP 6: BUILD THE USER INTERFACE
# ══════════════════════════════════════════════════════════════════
# Everything below is what the user sees in the browser.
# Streamlit runs this code top-to-bottom every time a user interacts.

# ── Header ────────────────────────────────────────────────────────
st.markdown(
    """
    <h1 style='text-align:center; color:#E50914; font-size:3rem;'>
        🎬 CineMatch
    </h1>
    <p style='text-align:center; color:gray; font-size:1.1rem;'>
        A content-based movie recommendation engine trained on 4,800+ movies
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# ── Search Box ────────────────────────────────────────────────────
# Dropdown with all movie titles for the user to pick from
selected_movie = st.selectbox(
    label="🔍 Search for a movie you liked:",
    options=movies_df["title"].values,    # all 4800+ movie titles
    index=None,                           # nothing selected by default
    placeholder="Start typing a movie name..."
)

# ── Settings Row ──────────────────────────────────────────────────
col_left, col_right = st.columns([3, 1])

with col_right:
    num_recs = st.slider(
        "Number of recommendations",
        min_value=3,
        max_value=10,
        value=6
    )

# ── Recommend Button ──────────────────────────────────────────────
if st.button("🎯 Find Similar Movies", type="primary", use_container_width=True):

    if not selected_movie:
        # User clicked without selecting a movie
        st.warning("⚠️ Please select a movie from the dropdown first.")

    else:
        # Show a loading spinner while computing
        with st.spinner(f"Finding movies similar to **{selected_movie}**..."):
            names, ids = recommend(selected_movie, num_recommendations=num_recs)

        if not names:
            st.error("Could not find recommendations. Try a different movie.")

        else:
            st.markdown(f"### 🎬 Because you liked **{selected_movie}**, you might enjoy:")
            st.markdown("")

            # ── Display Results in Rows of 3 ──────────────────────
            # We loop through results and display 3 movies per row
            # using st.columns(3)

            for row_start in range(0, len(names), 3):
                # Create 3 columns for this row
                cols = st.columns(3)

                for col_position in range(3):
                    movie_index = row_start + col_position

                    # Check we haven't gone past the end of our list
                    if movie_index < len(names):
                        movie_name = names[movie_index]
                        movie_id   = ids[movie_index]

                        # Fetch poster and details from TMDB API
                        poster_url = fetch_poster(movie_id)
                        print(poster_url)
                        details    = fetch_details(movie_id)

                        # Extract details safely with .get() fallbacks
                        rating      = details.get("vote_average", "N/A")
                        overview    = details.get("overview", "No description available.")
                        release     = details.get("release_date", "")
                        year        = release[:4] if release else "N/A"

                        # Display inside the column
                        with cols[col_position]:
                            st.image(poster_url, width=250)                            
                            st.markdown(f"**{movie_name}** ({year})")
                            st.markdown(f"⭐ {rating}/10")

                            # Collapsible synopsis section
                            with st.expander("📖 Synopsis"):
                                st.write(
                                    overview[:300] + "..."
                                    if len(overview) > 300
                                    else overview
                                )

                st.markdown("")   # spacing between rows


# ══════════════════════════════════════════════════════════════════
# STEP 7: HOW IT WORKS — EXPLANATION SECTION
# ══════════════════════════════════════════════════════════════════
# This section explains the tech to visitors.
# Good for your portfolio — shows you understand what you built.

st.markdown("---")

with st.expander("🔧 How does CineMatch work?"):
    st.markdown("""
    **CineMatch uses Content-Based Filtering** — it recommends movies
    with similar *content*, not based on what other users watched.

    ### The 5-Step Process:

    **Step 1 — Feature Extraction**
    For every movie, we combine:
    - Plot overview (what the movie is about)
    - Genres (Action, Drama, Comedy...)
    - Keywords (specific themes and topics)
    - Top 3 cast members
    - Director name

    **Step 2 — Text Preprocessing**
    - Multi-word names collapse: "Sam Worthington" → "SamWorthington"
    - Stemming reduces words to roots: "running" → "run"
    - All text lowercased for consistency

    **Step 3 — Bag of Words Vectorization**
    CountVectorizer builds a vocabulary of 5,000 words.
    Each movie becomes a vector of 5,000 numbers — how often
    each vocabulary word appears in that movie's tags.

    **Step 4 — Cosine Similarity**
    We compute a 4,800 × 4,800 matrix of similarity scores.
    Cosine similarity measures the angle between two vectors:
    - Score 1.0 = identical content
    - Score 0.0 = nothing in common

    **Step 5 — Ranking**
    When you select a movie, we look at its row in the matrix,
    sort all other movies by their similarity score, and return
    the top results.

    **Posters & Ratings:** Fetched live from [The Movie Database API](https://www.themoviedb.org/)
    """)

# ── Footer Stats ──────────────────────────────────────────────────
st.markdown("---")
c1, c2, c3, c4 = st.columns(4)
c1.metric("🎬 Movies in Database", f"{len(movies_df):,}")
c2.metric("🔢 Vocabulary Size",    "5,000 words")
c3.metric("📐 Algorithm",          "Cosine Similarity")
c4.metric("🌐 Poster Source",      "TMDB API")