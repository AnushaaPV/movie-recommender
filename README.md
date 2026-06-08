# movie-recommender
Content-based movie recommendation system built with Python and Streamlit

# 🎬 CineMatch — Movie Recommendation System

>A Machine Learning-powered movie recommendation web application that suggests similar movies based on content similarity. Built using Python, Scikit-learn, and Streamlit,trained on 4,800+ movies from the TMDB dataset. This project provides personalized movie recommendations along with posters, ratings, release years, and movie overviews.
---

## 🚀 Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://anushaapv-movie-recommender-app.streamlit.app)

> 👆 Click the badge to try the live app — no installation needed!

---

## 📌 What It Does

- 🔍 **Search** from 4,800+ movies using a smart dropdown
- 🎯 **Get recommendations** based on content similarity — not popularity
- 🖼️ **See movie posters** fetched live from TMDB API
- ⭐ **View ratings and synopses** for every recommended movie
- 🎛️ **Control** how many recommendations you want (3 to 10)

---

## 🧠 How It Works

```
You pick a movie
       ↓
System looks at its content:
plot + genres + keywords + top 3 cast + director
       ↓
Compares it against all 4,800 movies
using Cosine Similarity
       ↓
Returns the most similar movies
       ↓
Fetches posters & ratings from TMDB API
       ↓
Displays results in the web app
```

### Algorithm: Content-Based Filtering

| Step | What Happens |
|---|---|
| **1. Feature Extraction** | Combine plot overview, genres, keywords, top 3 cast, and director into one "tags" string per movie |
| **2. Text Preprocessing** | Collapse multi-word names (`Sam Worthington` → `SamWorthington`). Apply Porter Stemming (`running` → `run`) |
| **3. Vectorization** | CountVectorizer converts each movie's tags into a 5,000-dimension numerical vector |
| **4. Cosine Similarity** | Compute a 4,800 × 4,800 matrix of similarity scores between every pair of movies |
| **5. Recommendation** | Sort the selected movie's similarity row from highest to lowest — return top N results |

### Why Content-Based Filtering?

Content-based filtering works purely on **movie attributes** — no user history or ratings needed:
- ✅ Works for brand new users immediately
- ✅ No cold start problem
- ✅ Recommendations are explainable — you can see exactly why two movies are similar

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.11 |
| **Web Framework** | Streamlit |
| **Machine Learning** | Scikit-learn (CountVectorizer, Cosine Similarity) |
| **NLP** | NLTK (Porter Stemmer) |
| **Data Processing** | Pandas, NumPy |
| **Poster & Ratings API** | TMDB REST API (live calls) |
| **Dataset** | TMDB 5000 Movie Dataset (Kaggle) |
| **Deployment** | Streamlit Cloud (free hosting) |
| **Version Control** | Git + GitHub |

---

## 📁 Project Structure

```
movie-recommender/
│
├── app.py                    ← Streamlit web app (UI + recommendation logic + API)
├── model_builder.py          ← ML pipeline — run once to build the model
├── requirements.txt          ← Python dependencies for deployment
├── README.md                 ← This file
├── .gitignore                ← Files excluded from Git
│
├── tmdb_5000_movies.csv      ← Movie metadata (title, genres, overview, keywords)
├── tmdb_5000_credits.csv     ← Cast and crew data
│
├── movies.pkl                ← Generated: cleaned movie DataFrame (4,806 rows)
└── similarity.pkl            ← Generated: 4,806 × 4,806 cosine similarity matrix
```

---

## ⚙️ Run Locally


## Run it yourself

```bash
# clone the repo
git clone https://github.com/AnushaaPV/movie-recommender.git
cd movie-recommender

# set up a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# install libraries
pip install -r requirements.txt
```

Get a free TMDB API key from [themoviedb.org](https://www.themoviedb.org/) (Settings → API) and paste it into `app.py` where it says `TMDB_API_KEY`.

```bash
# build the model (takes about 1-2 minutes)
python model_builder.py

# run the app
streamlit run app.py
```
---

## 🌐 Deploy to Streamlit Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub → click **New app**
4. Select repository: `AnushaaPV/movie-recommender`, branch: `main`, file: `app.py`
5. Click **Advanced settings → Secrets** and add:
```toml
TMDB_API_KEY = "your_actual_key_here"
```
6. Click **Deploy** — live in 3-5 minutes ✅

---

## 🔑 Environment Variables

| Variable | Description | Where to Get It |
|---|---|---|
| `TMDB_API_KEY` | Fetches movie posters and ratings | [themoviedb.org](https://www.themoviedb.org/) → Settings → API |

---

## 📊 Dataset Details

| Property | Detail |
|---|---|
| **Source** | [TMDB 5000 Movie Dataset — Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) |
| **Total Movies** | 4,806 |
| **Files** | `tmdb_5000_movies.csv` + `tmdb_5000_credits.csv` |
| **Features Used** | title, overview, genres, keywords, cast (top 3), crew (director) |
| **Vocabulary Size** | 5,000 most common words |
| **Similarity Matrix** | 4,806 × 4,806 |

---

## 🗂️ Key Files Explained

### `model_builder.py` — Run Once
- Loads and merges both CSV files on the `title` column
- Uses `id` column from movies CSV as the correct TMDB movie ID
- Extracts genres, keywords, top 3 cast, director from JSON-like strings using `ast.literal_eval`
- Removes spaces from multi-word names so they stay as one token
- Creates unified "tags" column by combining all features
- Applies Porter Stemming via NLTK
- Vectorizes using CountVectorizer with 5,000 max features
- Computes cosine similarity matrix
- Saves `movies.pkl` and `similarity.pkl`

### `app.py` — The Web App
- Loads saved pkl files (auto-builds on first run if missing)
- Provides a searchable dropdown with all 4,800+ titles
- Recommendation function: finds movie index → sorts similarity row → returns top-N
- Fetches live poster, rating, release year, and overview from TMDB API
- Displays results in responsive 3-column grid
- Includes API key test tool and "How It Works" explainer

---

## 🌱 Future Improvements

- [ ] Add collaborative filtering using user ratings data
- [ ] User login and personal watchlist/favourites
- [ ] Filter by genre, release year, or minimum rating
- [ ] TV show recommendations
- [ ] Trailer links via YouTube Data API
- [ ] Recommend based on multiple movies at once

---

## 💡 What I Learned Building This

- Engineering features from raw JSON-like nested text in real datasets
- The difference between content-based and collaborative filtering approaches
- How CountVectorizer and cosine similarity work mathematically
- Building and deploying a full end-to-end ML web application
- Integrating live REST APIs (TMDB) to enrich ML output with real data
- Git workflow for version control, collaboration, and cloud deployment
- Debugging cross-platform path issues between Windows and Linux (Streamlit Cloud)

---


