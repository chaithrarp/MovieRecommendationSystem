from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
import ast
import re
import os
from datetime import datetime
import json
import random

app = Flask(__name__, 
            template_folder='../frontend',  # Point to frontend folder
            static_folder='../frontend')    # Serve CSS/JS from frontend folder
CORS(app)

# Global variables for our datasets and models
movies_df = None
credits_df = None
tfidf_matrix = None
cosine_sim = None
genre_matrix = None
mlb = None

def safe_literal_eval(val):
    """Safely evaluate string representations of Python literals"""
    if pd.isna(val):
        return []
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            return []
    return val

def extract_names(data_list, limit=None):
    """Extract names from a list of dictionaries"""
    if not data_list:
        return []
    names = [item['name'] for item in data_list if isinstance(item, dict) and 'name' in item]
    return names[:limit] if limit else names

def load_and_preprocess_data():
    """Load and preprocess the TMDb dataset"""
    global movies_df, credits_df, tfidf_matrix, cosine_sim, genre_matrix, mlb
    
    print("Loading TMDb dataset...")
    
    # Try different possible paths for the dataset
    possible_paths = [
        ('movies_metadata.csv', 'credits.csv'),  # Current directory
        ('data/movies_metadata.csv', 'data/credits.csv'),  # data folder
        ('../data/movies_metadata.csv', '../data/credits.csv'),  # parent data folder
        ('backend/movies_metadata.csv', 'backend/credits.csv'),  # backend folder
        ('dataset/movies_metadata.csv', 'dataset/credits.csv'),  # dataset folder
    ]
    
    movies_loaded = False
    credits_loaded = False
    
    for movies_path, credits_path in possible_paths:
        try:
            if os.path.exists(movies_path):
                print(f"Found movies dataset at: {movies_path}")
                movies_df = pd.read_csv(movies_path, low_memory=False)
                movies_loaded = True
                
                if os.path.exists(credits_path):
                    print(f"Found credits dataset at: {credits_path}")
                    credits_df = pd.read_csv(credits_path)
                    credits_loaded = True
                else:
                    print(f"Credits file not found at: {credits_path}")
                    credits_df = None
                break
        except Exception as e:
            print(f"Failed to load from {movies_path}: {e}")
            continue
    
    if not movies_loaded:
        print("ERROR: Could not find TMDb dataset files!")
        print("Please ensure you have:")
        print("1. movies_metadata.csv")
        print("2. credits.csv")
        print("In one of these locations:")
        for movies_path, credits_path in possible_paths:
            print(f"   - {movies_path} and {credits_path}")
        create_sample_data()
        return
    
    try:
        print(f"Initial dataset size: {len(movies_df)} movies")
        
        # Clean movies dataset
        print("Cleaning dataset...")
        
        # Remove duplicates
        movies_df = movies_df.drop_duplicates(subset=['title'])
        movies_df = movies_df.dropna(subset=['title', 'overview'])
        
        # Handle invalid IDs - be more careful with this
        movies_df = movies_df[movies_df['id'].notna()]
        movies_df['id'] = pd.to_numeric(movies_df['id'], errors='coerce')
        movies_df = movies_df[movies_df['id'].notna()]
        movies_df['id'] = movies_df['id'].astype(int)
        
        # Process release date and year
        movies_df['release_date'] = pd.to_datetime(movies_df['release_date'], errors='coerce')
        movies_df['year'] = movies_df['release_date'].dt.year
        movies_df = movies_df[movies_df['year'].notna()]
        movies_df['year'] = movies_df['year'].astype(int)
        
        # Process rating
        movies_df['vote_average'] = pd.to_numeric(movies_df['vote_average'], errors='coerce')
        movies_df = movies_df[movies_df['vote_average'].notna()]
        movies_df['rating'] = movies_df['vote_average']
        
        # Process vote count
        movies_df['vote_count'] = pd.to_numeric(movies_df['vote_count'], errors='coerce')
        movies_df = movies_df[movies_df['vote_count'].notna()]
        
        # Process genres
        movies_df['genres_list'] = movies_df['genres'].apply(safe_literal_eval)
        movies_df['genres_names'] = movies_df['genres_list'].apply(lambda x: extract_names(x))
        
        # Filter out movies without genres
        movies_df = movies_df[movies_df['genres_names'].apply(len) > 0]
        
        print(f"After basic cleaning: {len(movies_df)} movies")
        
        # Merge with credits if available
        if credits_loaded and credits_df is not None:
            print("Processing credits data...")
            credits_df['id'] = pd.to_numeric(credits_df['id'], errors='coerce')
            credits_df = credits_df[credits_df['id'].notna()]
            credits_df['id'] = credits_df['id'].astype(int)
            
            credits_df['cast_list'] = credits_df['cast'].apply(safe_literal_eval)
            credits_df['cast_names'] = credits_df['cast_list'].apply(lambda x: extract_names(x, limit=5))
            credits_df['crew_list'] = credits_df['crew'].apply(safe_literal_eval)
            
            # Merge datasets
            movies_df = movies_df.merge(credits_df[['id', 'cast_names', 'crew_list']], on='id', how='left')
            print("Credits data merged successfully")
        else:
            print("No credits data available, proceeding without cast information")
            movies_df['cast_names'] = [[] for _ in range(len(movies_df))]
        
        # Fill missing cast
        movies_df['cast_names'] = movies_df['cast_names'].apply(lambda x: x if isinstance(x, list) and len(x) > 0 else [])
        
        # Create content for similarity calculation
        movies_df['content'] = (
            movies_df['overview'].fillna('') + ' ' +
            movies_df['genres_names'].apply(lambda x: ' '.join(x) if x else '') + ' ' +
            movies_df['cast_names'].apply(lambda x: ' '.join(x[:3]) if x else '')
        )
        
        # Filter for quality movies
        movies_df = movies_df[
            (movies_df['year'] >= 1970) & 
            (movies_df['year'] <= 2024) &
            (movies_df['rating'] >= 1.0) &
            (movies_df['vote_count'] >= 5)
        ].copy()
        
        print(f"After quality filtering: {len(movies_df)} movies")
        
        # Sample for performance if dataset is too large
        if len(movies_df) > 10000:
            movies_df = movies_df.nlargest(10000, 'vote_count').reset_index(drop=True)
            print(f"Sampled to {len(movies_df)} movies for performance")
        
        # Create TF-IDF matrix
        print("Creating TF-IDF matrix...")
        tfidf = TfidfVectorizer(max_features=3000, stop_words='english', lowercase=True)
        tfidf_matrix = tfidf.fit_transform(movies_df['content'])
        
        # Create genre matrix for genre-based filtering
        print("Creating genre matrix...")
        mlb = MultiLabelBinarizer()
        genre_matrix = mlb.fit_transform(movies_df['genres_names'])
        
        print(f"Dataset loaded successfully! {len(movies_df)} movies available.")
        print(f"Available genres: {list(mlb.classes_)}")
        
        # Print some sample movie titles to verify
        print("\nSample movie titles:")
        sample_titles = movies_df['title'].head(10).tolist()
        for i, title in enumerate(sample_titles, 1):
            print(f"{i}. {title}")
        
    except Exception as e:
        print(f"Error processing dataset: {e}")
        import traceback
        traceback.print_exc()
        create_sample_data()

def create_sample_data():
    """Create sample data if real dataset fails to load"""
    global movies_df, tfidf_matrix, cosine_sim, genre_matrix, mlb
    
    print("Creating sample movie dataset...")
    
    # Create more realistic sample data
    sample_movies = [
        {
            'id': 1, 'title': 'The Shawshank Redemption', 'year': 1994, 'rating': 9.3, 'vote_count': 2000000,
            'overview': 'Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.',
            'genres_names': ['Drama'], 'cast_names': ['Tim Robbins', 'Morgan Freeman', 'Bob Gunton']
        },
        {
            'id': 2, 'title': 'The Godfather', 'year': 1972, 'rating': 9.2, 'vote_count': 1500000,
            'overview': 'The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant son.',
            'genres_names': ['Crime', 'Drama'], 'cast_names': ['Marlon Brando', 'Al Pacino', 'James Caan']
        },
        {
            'id': 3, 'title': 'The Dark Knight', 'year': 2008, 'rating': 9.0, 'vote_count': 2200000,
            'overview': 'When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests.',
            'genres_names': ['Action', 'Crime', 'Drama'], 'cast_names': ['Christian Bale', 'Heath Ledger', 'Aaron Eckhart']
        },
        {
            'id': 4, 'title': 'Pulp Fiction', 'year': 1994, 'rating': 8.9, 'vote_count': 1800000,
            'overview': 'The lives of two mob hitmen, a boxer, a gangster and his wife intertwine in four tales of violence and redemption.',
            'genres_names': ['Crime', 'Drama'], 'cast_names': ['John Travolta', 'Uma Thurman', 'Samuel L. Jackson']
        },
        {
            'id': 5, 'title': 'Forrest Gump', 'year': 1994, 'rating': 8.8, 'vote_count': 1700000,
            'overview': 'The presidencies of Kennedy and Johnson, Vietnam, Watergate, and other history unfold through the perspective of an Alabama man.',
            'genres_names': ['Drama', 'Romance'], 'cast_names': ['Tom Hanks', 'Robin Wright', 'Gary Sinise']
        }
    ]
    
    # Extend with more sample movies
    genres_list = ['Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary',
                   'Drama', 'Family', 'Fantasy', 'History', 'Horror', 'Music', 'Mystery',
                   'Romance', 'Science Fiction', 'Thriller', 'War', 'Western']
    
    actors_list = ['Leonardo DiCaprio', 'Scarlett Johansson', 'Tom Hanks', 'Meryl Streep',
                   'Robert Downey Jr.', 'Jennifer Lawrence', 'Brad Pitt', 'Emma Stone',
                   'Ryan Gosling', 'Anne Hathaway', 'Christian Bale', 'Natalie Portman',
                   'Matt Damon', 'Sandra Bullock', 'Will Smith', 'Angelina Jolie']
    
    # Add more sample movies
    for i in range(6, 101):
        sample_movies.append({
            'id': i,
            'title': f'Sample Movie {i}',
            'year': random.randint(1990, 2023),
            'rating': round(random.uniform(5.0, 9.0), 1),
            'vote_count': random.randint(100, 50000),
            'overview': f'This is a sample movie {i} with an engaging plot and great characters that will keep you entertained.',
            'genres_names': random.sample(genres_list, random.randint(1, 3)),
            'cast_names': random.sample(actors_list, random.randint(2, 4))
        })
    
    movies_df = pd.DataFrame(sample_movies)
    
    # Create content and matrices
    movies_df['content'] = (
        movies_df['overview'] + ' ' +
        movies_df['genres_names'].apply(lambda x: ' '.join(x)) + ' ' +
        movies_df['cast_names'].apply(lambda x: ' '.join(x))
    )
    
    tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies_df['content'])
    
    mlb = MultiLabelBinarizer()
    genre_matrix = mlb.fit_transform(movies_df['genres_names'])
    
    print("Sample dataset created successfully!")
    print("NOTE: This is sample data. Please add your TMDb dataset files to get real movie recommendations.")

def get_recommendations(preferences, num_recommendations=20):
    """Get movie recommendations based on user preferences"""
    try:
        selected_genres = preferences.get('genres', [])
        selected_actors = preferences.get('actors', [])
        min_rating = preferences.get('min_rating', 5.0)
        min_year = preferences.get('min_year', 1990)
        
        # Filter movies based on preferences
        filtered_movies = movies_df[
            (movies_df['rating'] >= min_rating) &
            (movies_df['year'] >= min_year)
        ].copy()
        
        if len(filtered_movies) == 0:
            return []
        
        scores = np.zeros(len(filtered_movies))
        
        # Genre-based scoring
        if selected_genres:
            for idx, movie_genres in enumerate(filtered_movies['genres_names']):
                genre_matches = len(set(selected_genres) & set(movie_genres))
                scores[idx] += genre_matches * 2
        
        # Actor-based scoring
        if selected_actors:
            for idx, movie_cast in enumerate(filtered_movies['cast_names']):
                actor_matches = len(set(selected_actors) & set(movie_cast))
                scores[idx] += actor_matches * 1.5
        
        # Add rating bonus
        scores += filtered_movies['rating'].values * 0.1
        
        # Add popularity bonus
        scores += np.log1p(filtered_movies['vote_count'].values) * 0.05
        
        # Get top recommendations
        top_indices = np.argsort(scores)[::-1][:num_recommendations]
        recommendations = []
        
        for idx in top_indices:
            movie = filtered_movies.iloc[idx]
            match_reasons = []
            
            if selected_genres:
                matching_genres = set(selected_genres) & set(movie['genres_names'])
                if matching_genres:
                    match_reasons.extend([f"Genre: {g}" for g in matching_genres])
            
            if selected_actors:
                matching_actors = set(selected_actors) & set(movie['cast_names'])
                if matching_actors:
                    match_reasons.extend([f"Actor: {a}" for a in matching_actors])
            
            if movie['rating'] >= min_rating + 1:
                match_reasons.append(f"High Rating: {movie['rating']}")
            
            recommendations.append({
                'id': int(movie['id']),
                'title': movie['title'],
                'year': int(movie['year']),
                'rating': float(movie['rating']),
                'genres': movie['genres_names'],
                'cast': movie['cast_names'][:5],
                'overview': movie['overview'][:200] + '...' if len(movie['overview']) > 200 else movie['overview'],
                'match_reasons': match_reasons,
                'score': float(scores[idx])
            })
        
        return recommendations
        
    except Exception as e:
        print(f"Error getting recommendations: {e}")
        import traceback
        traceback.print_exc()
        return []

# Routes
@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_from_directory('../frontend', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files (CSS, JS, etc.)"""
    return send_from_directory('../frontend', filename)

@app.route('/api/movies/genres')
def get_genres():
    """Get all available genres"""
    try:
        if mlb is not None:
            genres = list(mlb.classes_)
        else:
            genres = ['Action', 'Comedy', 'Drama', 'Thriller', 'Romance', 'Sci-Fi']
        
        return jsonify({
            'success': True,
            'genres': sorted(genres)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/movies/actors')
def get_actors():
    """Get all available actors"""
    try:
        all_actors = set()
        for cast_list in movies_df['cast_names']:
            if isinstance(cast_list, list):
                all_actors.update(cast_list)
        
        return jsonify({
            'success': True,
            'actors': sorted(list(all_actors))
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/movies/trending')
def get_trending():
    """Get trending movies"""
    try:
        # Sort by a combination of rating and vote count
        trending = movies_df.nlargest(20, ['rating', 'vote_count'])[['id', 'title', 'year', 'rating', 'genres_names', 'cast_names', 'overview', 'vote_count']]
        
        movies = []
        for _, movie in trending.iterrows():
            movies.append({
                'id': int(movie['id']),
                'title': movie['title'],
                'year': int(movie['year']),
                'rating': float(movie['rating']),
                'genres': movie['genres_names'],
                'cast': movie['cast_names'][:5] if movie['cast_names'] else [],
                'overview': movie['overview'][:200] + '...' if len(movie['overview']) > 200 else movie['overview']
            })
        
        return jsonify({
            'success': True,
            'movies': movies
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/movies/search')
def search_movies():
    """Search for movies"""
    try:
        query = request.args.get('q', '').lower()
        if not query:
            return jsonify({'success': True, 'movies': []})
        
        # Search in title and overview
        matches = movies_df[
            movies_df['title'].str.lower().str.contains(query, na=False) |
            movies_df['overview'].str.lower().str.contains(query, na=False)
        ].head(20)
        
        movies = []
        for _, movie in matches.iterrows():
            movies.append({
                'id': int(movie['id']),
                'title': movie['title'],
                'year': int(movie['year']),
                'rating': float(movie['rating']),
                'genres': movie['genres_names'],
                'cast': movie['cast_names'][:5] if movie['cast_names'] else [],
                'overview': movie['overview'][:200] + '...' if len(movie['overview']) > 200 else movie['overview']
            })
        
        return jsonify({
            'success': True,
            'movies': movies
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/recommend', methods=['POST'])
def recommend():
    """Get movie recommendations"""
    try:
        preferences = request.json
        recommendations = get_recommendations(preferences)
        
        return jsonify({
            'success': True,
            'recommendations': recommendations
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/movies/rate', methods=['POST'])
def rate_movie():
    """Rate a movie (placeholder implementation)"""
    try:
        data = request.json
        movie_id = data.get('movie_id')
        rating = data.get('rating')
        
        # In a real app, you'd save this to a database
        print(f"User rated movie {movie_id} with {rating} stars")
        
        return jsonify({
            'success': True,
            'message': 'Rating saved successfully'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/stats')
def get_stats():
    """Get application statistics"""
    try:
        total_movies = len(movies_df)
        total_genres = len(mlb.classes_) if mlb else 0
        
        all_actors = set()
        for cast_list in movies_df['cast_names']:
            if isinstance(cast_list, list):
                all_actors.update(cast_list)
        total_actors = len(all_actors)
        
        return jsonify({
            'success': True,
            'stats': {
                'total_movies': total_movies,
                'total_genres': total_genres,
                'total_actors': total_actors
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Movie Recommender System...")
    print("Checking for TMDb dataset files...")
    
    # Check if we can find the dataset files
    dataset_found = False
    for movies_path, credits_path in [
        ('movies_metadata.csv', 'credits.csv'),
        ('data/movies_metadata.csv', 'data/credits.csv'),
        ('../data/movies_metadata.csv', '../data/credits.csv'),
        ('backend/movies_metadata.csv', 'backend/credits.csv'),
        ('dataset/movies_metadata.csv', 'dataset/credits.csv'),
    ]:
        if os.path.exists(movies_path):
            dataset_found = True
            print(f"✓ Found dataset at: {movies_path}")
            break
    
    if not dataset_found:
        print("⚠️  TMDb dataset not found! The app will use sample data.")
        print("   To use real movie data, please download the TMDb dataset and place:")
        print("   - movies_metadata.csv")
        print("   - credits.csv")
        print("   In the same directory as this script or in a 'data/' folder.")
    
    load_and_preprocess_data()
    print("Server starting on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
