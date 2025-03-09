import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def preprocess_text(text):
    return text.lower() if isinstance(text, str) else ""

def recommend_movie(df, genre=None, year=None, actor=None, platform=None, content_rating=None):
    df = df.copy()
    df['Combined'] = df[['Genre', 'Main Cast', 'Network', 'Content Rating']].fillna('').agg(' '.join, axis=1).apply(preprocess_text)
    
    user_input = f"{genre or ''} {actor or ''} {platform or ''} {content_rating or ''}".strip().lower()
    
    if not user_input:
        return "Please provide at least one filter criteria."
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['Combined'])
    user_vector = vectorizer.transform([user_input])
    
    similarities = cosine_similarity(user_vector, tfidf_matrix).flatten()
    df['Similarity'] = similarities
    df = df.sort_values(by='Similarity', ascending=False)
    
    if year:
        df = df[df['Year'] == year]
    
    if df.empty:
        return "No matching movies found. Try adjusting your filters."
    
    recommended = df.iloc[0]
    return {
        "Name": recommended['Name'],
        "Year": recommended['Year'],
        "Genre": recommended['Genre'],
        "Main Cast": recommended['Main Cast'],
        "Content Rating": recommended['Content Rating'],
        "Platform": recommended['Network'],
        "Synopsis": recommended['Sinopsis'],
        "Score": recommended['Score'],
        "Image URL": recommended['img url']
    }

if __name__ == "__main__":
    file_path = "kdrama_list.csv"
    df = load_data(file_path)
    
    # Example usage
    user_genre = input("Enter genre: ")
    user_year = input("Enter year (optional): ")
    user_year = int(user_year) if user_year.isdigit() else None
    user_actor = input("Enter actor name (optional): ")
    user_platform = input("Enter platform (optional): ")
    user_rating = input("Enter content rating (optional): ")
    
    suggestion = recommend_movie(df, user_genre, user_year, user_actor, user_platform, user_rating)
    print("\nMovie Recommendation:")
    print(suggestion)
