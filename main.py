from fastapi import FastAPI, Query
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# Load the dataset
file_path = "kdrama_list.csv"
df = pd.read_csv(file_path)

def preprocess_text(text):
    return text.lower() if isinstance(text, str) else ""

def recommend_movie(genre=None, year=None, actor=None, platform=None, content_rating=None):
    filtered_df = df.copy()
    filtered_df["Combined"] = filtered_df[["Genre", "Main Cast", "Network", "Content Rating"]].fillna("").agg(" ".join, axis=1).apply(preprocess_text)
    
    user_input = f"{genre or ''} {actor or ''} {platform or ''} {content_rating or ''}".strip().lower()
    
    if not user_input:
        return {"error": "Please provide at least one filter criteria."}
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(filtered_df["Combined"])
    user_vector = vectorizer.transform([user_input])
    
    similarities = cosine_similarity(user_vector, tfidf_matrix).flatten()
    filtered_df["Similarity"] = similarities
    filtered_df = filtered_df.sort_values(by="Similarity", ascending=False)

    if year:
        filtered_df = filtered_df[filtered_df["Year"].astype(str) == str(year)]

    if filtered_df.empty:
        return {"error": "No matching movies found. Try adjusting your filters."}

    recommended = filtered_df.iloc[0]
    return {
        "Name": str(recommended["Name"]),
        "Year": int(recommended["Year"]) if pd.notna(recommended["Year"]) else None,
        "Genre": str(recommended["Genre"]),
        "Main Cast": str(recommended["Main Cast"]),
        "Content Rating": str(recommended["Content Rating"]),
        "Platform": str(recommended["Network"]),
        "Synopsis": str(recommended["Sinopsis"]),
        "Score": float(recommended["Score"]) if pd.notna(recommended["Score"]) else None,
        "Image URL": str(recommended["img url"])
    }

@app.get("/")
def home():
    return {"message": "Welcome to the KDrama Recommendation API"}

@app.get("/monitor")
def monitor():
    return {"message": "Monitoring the API"}

@app.get("/recommend")
def get_recommendation(
    genre: str = Query(None),
    year: int = Query(None),
    actor: str = Query(None),
    platform: str = Query(None),
    content_rating: str = Query(None),
):
    return recommend_movie(genre, year, actor, platform, content_rating)
