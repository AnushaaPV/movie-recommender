import os
import sys
import pickle
import requests
import streamlit as st

# ── PAGE CONFIG ────────────────────────────────────────────
st.set_page_config(
    page_title="CineMatch — Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# ── PATHS — works on Windows AND Streamlit Cloud (Linux) ──
BASE           = os.path.dirname(os.path.abspath(__file__))
MOVIES_PKL     = os.path.join(BASE, "movies.pkl")
SIMILARITY_PKL = os.path.join(BASE, "similarity.pkl")

# ── TMDB API KEY ───────────────────────────────────────────
# For local use: paste your key directly below
# For Streamlit Cloud: add it in App Settings → Secrets as:
#   TMDB_API_KEY = "your_key_here"
try:
    TMDB_API_KEY = st.secrets["TMDB_API_KEY"]
except Exception:
    TMDB_API_KEY = "d261bd3fbed3ebd2c0bc38075839b3e5"

# ── AUTO BUILD MODEL IF PKL FILES MISSING ─────────────────
# This runs automatically on Streamlit Cloud first deployment
if not os.path.exists(MOVIES_PKL) or not os.path.exists(SIMILARITY_PKL):
    with st.spinner("⚙️ Building model for first time — please wait 2-3 minutes..."):
        import subprocess
        result = subprocess.run(
            [sys.executable, os.path.join(BASE, "model_builder.py")],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            st.error("❌ Model build failed. See error below:")
            st.code(result.stderr)
            st.stop()
        else:
            st.success("✅ Model built successfully!")
            st.rerun()

# ── LOAD MODEL ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    movies_df  = pickle.load(open(MOVIES_PKL,     "rb"))
    similarity = pickle.load(open(SIMILARITY_PKL, "rb"))
    return movies_df, similarity

movies_df, similarity = load_model()

# ── FETCH POSTER AND DETAILS ───────────────────────────────
def fetch_info(movie_id):
    fallback = {
        "poster_url" : "https://via.placeholder.com/300x450?text=No+Poster",
        "rating"     : "N/A",
        "overview"   : "No description available.",
        "year"       : "N/A"
    }

    if TMDB_API_KEY == "d261bd3fbed3ebd2c0bc38075839b3e5" or not TMDB_API_KEY:
        fallback["poster_url"] = "https://via.placeholder.com/300x450?text=Add+API+Key"
        return fallback

    try:
        url      = f"https://api.themoviedb.org/3/movie/{int(movie_id)}?api_key={TMDB_API_KEY}&language=en-US"
        response = requests.get(url, timeout=10)

        if response.status_code == 401:
            fallback["poster_url"] = "https://via.placeholder.com/300x450?text=Invalid+Key"
            return fallback

        if response.status_code != 200:
            return fallback

        data         = response.json()
        poster_path  = data.get("poster_path", "")
        release_date = data.get("release_date", "")
        raw_rating   = data.get("vote_average", None)

        return {
            "poster_url" : f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else fallback["poster_url"],
            "rating"     : round(raw_rating, 1) if raw_rating else "N/A",
            "overview"   : data.get("overview", "No description available."),
            "year"       : release_date[:4] if release_date else "N/A"
        }
    except Exception:
        return fallback

# ── RECOMMENDATION FUNCTION ────────────────────────────────
def recommend(movie_title, n=6):
    try:
        idx = movies_df[movies_df["title"] == movie_title].index[0]
    except IndexError:
        return [], []

    scores = sorted(
        list(enumerate(similarity[idx])),
        key=lambda x: x[1],
        reverse=True
    )[1 : n + 1]

    names = [movies_df.iloc[i]["title"]    for i, _ in scores]
    ids   = [movies_df.iloc[i]["movie_id"] for i, _ in scores]
    return names, ids

# ══════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════

# ── Header ─────────────────────────────────────────────────
st.markdown("""
    <h1 style='text-align:center; color:#E50914; font-size:3rem;'>
        🎬 CineMatch
    </h1>
    <p style='text-align:center; color:gray; font-size:1.1rem;'>
        Content-based movie recommendation engine · 4,800+ movies
    </p>
""", unsafe_allow_html=True)

st.markdown("---")

# ── API Key Warning ────────────────────────────────────────
if TMDB_API_KEY == "d261bd3fbed3ebd2c0bc38075839b3e5":
    st.warning(
        "⚠️ TMDB API key not set — posters and ratings won't load. "
        "Get your free key at themoviedb.org → Settings → API"
    )

# ── Search Box ─────────────────────────────────────────────
selected_movie = st.selectbox(
    "🔍 Search for a movie you liked:",
    options=movies_df["title"].values,
    index=None,
    placeholder="Start typing a movie name..."
)

# ── Slider ─────────────────────────────────────────────────
num_recs = st.slider("Number of recommendations", 3, 10, 6)

# ── Button ─────────────────────────────────────────────────
if st.button("🎯 Find Similar Movies", type="primary", use_container_width=True):

    if not selected_movie:
        st.warning("Please select a movie first.")
    else:
        with st.spinner("Finding recommendations..."):
            names, ids = recommend(selected_movie, n=num_recs)

        if not names:
            st.error("No recommendations found. Try another movie.")
        else:
            st.markdown(f"### 🎬 Because you liked **{selected_movie}**:")
            st.markdown("")

            for row_start in range(0, len(names), 3):
                cols = st.columns(3)
                for col_pos in range(3):
                    idx = row_start + col_pos
                    if idx < len(names):
                        info = fetch_info(ids[idx])
                        with cols[col_pos]:
                            st.image(
                                info["poster_url"],
                                use_container_width=True
                            )
                            st.markdown(f"**{names[idx]}**")
                            st.markdown(f"📅 {info['year']}  ⭐ {info['rating']}/10")
                            with st.expander("📖 Synopsis"):
                                overview = info["overview"]
                                st.write(
                                    overview[:300] + "..."
                                    if len(overview) > 300
                                    else overview
                                )
                st.markdown("")

# ── API Test Tool ──────────────────────────────────────────
st.markdown("---")
with st.expander("🔑 Test TMDB API Key"):
    if st.button("▶ Run Test"):
        if TMDB_API_KEY == "d261bd3fbed3ebd2c0bc38075839b3e5":
            st.error("❌ API key not set yet.")
        else:
            try:
                r = requests.get(
                    f"https://api.themoviedb.org/3/movie/19995?api_key={TMDB_API_KEY}",
                    timeout=10
                )
                if r.status_code == 200:
                    d = r.json()
                    st.success(f"✅ API Key works! Avatar rating: {d.get('vote_average')}/10")
                    if d.get("poster_path"):
                        st.image(
                            f"https://image.tmdb.org/t/p/w200{d['poster_path']}",
                            width=150
                        )
                elif r.status_code == 401:
                    st.error("❌ API key INVALID — re-copy from themoviedb.org → Settings → API")
                else:
                    st.warning(f"Unexpected status: {r.status_code}")
            except Exception as e:
                st.error(f"Connection error: {e}")

# ── How It Works ───────────────────────────────────────────
st.markdown("---")
with st.expander("🔧 How does CineMatch work?"):
    st.markdown("""
    **Content-Based Filtering** — recommends movies with similar content.

    | Step | What happens |
    |---|---|
    | Feature Extraction | Combine plot, genres, keywords, cast, director into one text |
    | Preprocessing | Collapse names, apply stemming to root words |
    | Vectorization | CountVectorizer → 5,000 dimension vectors |
    | Similarity | Cosine Similarity across 4,800 × 4,800 matrix |
    | Ranking | Sort scores, return top-N results |
    """)

# ── Footer ─────────────────────────────────────────────────
st.markdown("---")
c1, c2, c3, c4 = st.columns(4)
c1.metric("🎬 Movies",    f"{len(movies_df):,}")
c2.metric("🔢 Vocabulary", "5,000 words")
c3.metric("📐 Algorithm",  "Cosine Similarity")
c4.metric("🌐 Posters",    "TMDB API")