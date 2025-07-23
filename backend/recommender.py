import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import json
import ast
import re
from datetime import datetime
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

class EnhancedMovieRecommender:
    def __init__(self):
        self.movies_df = None
        self.credits_df = None
        self.processed_movies = None
        self.tfidf_matrix = None
        self.content_similarity = None
        self.svd_model = None
        self.nmf_model = None
        self.clustering_model = None
        self.rating_predictor = None
        self.knn_model = None
        self.movie_features = None
        self.genre_vectors = None
        self.user_profiles = {}
        self.load_and_process_data()
    
    def load_and_process_data(self):
        """Load and preprocess the movie dataset with enhanced processing"""
        try:
            print("Loading movie datasets...")
            self.movies_df = pd.read_csv('backend/movies_metadata.csv', low_memory=False)
            
            try:
                self.credits_df = pd.read_csv('backend/credits.csv')
            except FileNotFoundError:
                print("Credits file not found, proceeding without cast data")
                self.credits_df = None
            
            self.clean_data()
            self.process_features()
            self.create_advanced_features()
            self.train_ml_models()
            
            print(f"Loaded {len(self.processed_movies)} movies with advanced ML models!")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            self.create_sample_data()
    
    def clean_data(self):
        """Enhanced data cleaning with better filtering"""
        # Remove invalid entries
        self.movies_df = self.movies_df[pd.to_numeric(self.movies_df['id'], errors='coerce').notna()]
        self.movies_df['id'] = self.movies_df['id'].astype(int)
        
        # Filter out movies with missing essential data
        essential_cols = ['title', 'overview', 'genres', 'release_date']
        for col in essential_cols:
            if col in self.movies_df.columns:
                self.movies_df = self.movies_df[self.movies_df[col].notna()]
        
        # Enhanced numeric processing
        numeric_cols = ['vote_average', 'vote_count', 'popularity', 'runtime', 'budget', 'revenue']
        for col in numeric_cols:
            if col in self.movies_df.columns:
                self.movies_df[col] = pd.to_numeric(self.movies_df[col], errors='coerce')
        
        # More sophisticated filtering
        self.movies_df = self.movies_df[
            (self.movies_df['vote_average'].notna()) &
            (self.movies_df['vote_count'] >= 20) &  # Lowered threshold for more data
            (self.movies_df['vote_average'] > 0)
        ]
        
        # Extract and validate year
        self.movies_df['year'] = pd.to_datetime(self.movies_df['release_date'], errors='coerce').dt.year
        self.movies_df = self.movies_df[
            (self.movies_df['year'].notna()) &
            (self.movies_df['year'] >= 1950) &  # Filter very old movies
            (self.movies_df['year'] <= datetime.now().year + 2)  # Filter future movies
        ]
        
        print(f"After enhanced cleaning: {len(self.movies_df)} movies remain")
    
    def safe_eval_genres(self, genres_str):
        """Enhanced genre extraction"""
        try:
            if pd.isna(genres_str) or genres_str == '':
                return []
            genres_list = ast.literal_eval(genres_str)
            return [genre['name'] for genre in genres_list if isinstance(genre, dict) and 'name' in genre]
        except:
            return []
    
    def safe_eval_cast_crew(self, cast_str, role_type='cast', limit=10):
        """Enhanced cast and crew extraction"""
        try:
            if pd.isna(cast_str) or cast_str == '':
                return []
            cast_list = ast.literal_eval(cast_str)
            if role_type == 'cast':
                return [person['name'] for person in cast_list[:limit] 
                       if isinstance(person, dict) and 'name' in person]
            else:  # crew
                directors = [person['name'] for person in cast_list 
                           if isinstance(person, dict) and person.get('job') == 'Director']
                return directors[:3]  # Top 3 directors
        except:
            return []
    
    def process_features(self):
        """Enhanced feature processing"""
        # Process genres
        self.movies_df['genres_list'] = self.movies_df['genres'].apply(self.safe_eval_genres)
        self.movies_df['genres_str'] = self.movies_df['genres_list'].apply(lambda x: ' '.join(x))
        
        # Merge with credits data if available
        if self.credits_df is not None:
            self.credits_df['id'] = pd.to_numeric(self.credits_df['id'], errors='coerce')
            self.movies_df = self.movies_df.merge(
                self.credits_df[['id', 'cast', 'crew']], on='id', how='left'
            )
            
            # Process cast and crew
            self.movies_df['cast'] = self.movies_df['cast'].fillna('[]')
            self.movies_df['crew'] = self.movies_df['crew'].fillna('[]')
            
            self.movies_df['cast_list'] = self.movies_df['cast'].apply(
                lambda x: self.safe_eval_cast_crew(x, 'cast', 8)
            )
            self.movies_df['director_list'] = self.movies_df['crew'].apply(
                lambda x: self.safe_eval_cast_crew(x, 'crew', 3)
            )
            
            self.movies_df['cast_str'] = self.movies_df['cast_list'].apply(lambda x: ' '.join(x))
            self.movies_df['director_str'] = self.movies_df['director_list'].apply(lambda x: ' '.join(x))
        else:
            self.movies_df['cast_list'] = [[] for _ in range(len(self.movies_df))]
            self.movies_df['director_list'] = [[] for _ in range(len(self.movies_df))]
            self.movies_df['cast_str'] = ''
            self.movies_df['director_str'] = ''
        
        # Enhanced age categorization
        self.movies_df['age_category'] = self.movies_df.apply(self.enhanced_age_categorization, axis=1)
        
        # Calculate multiple popularity scores
        self.movies_df['popularity_score'] = self.calculate_enhanced_popularity_score()
        self.movies_df['quality_score'] = self.calculate_quality_score()
        
        # Create processed dataset with more features
        self.processed_movies = self.movies_df[[
            'id', 'title', 'overview', 'genres_list', 'cast_list', 'director_list',
            'vote_average', 'vote_count', 'year', 'age_category', 'runtime',
            'budget', 'revenue', 'popularity', 'genres_str', 'cast_str', 'director_str',
            'popularity_score', 'quality_score'
        ]].copy()
        
        # Fill missing values
        numeric_cols = ['runtime', 'budget', 'revenue', 'popularity']
        for col in numeric_cols:
            if col in self.processed_movies.columns:
                self.processed_movies[col] = self.processed_movies[col].fillna(
                    self.processed_movies[col].median()
                )
    
    def enhanced_age_categorization(self, row):
        """Enhanced age rating categorization using multiple signals"""
        genres = row.get('genres_list', [])
        title = str(row.get('title', '')).lower()
        overview = str(row.get('overview', '')).lower()
        
        # Enhanced keyword matching
        family_keywords = ['family', 'children', 'kid', 'disney', 'pixar', 'dreamworks', 'animated']
        teen_keywords = ['high school', 'teenager', 'young adult', 'coming of age']
        adult_keywords = ['violence', 'mature', 'psychological', 'complex']
        
        family_genres = ['Animation', 'Family', 'Adventure']
        teen_genres = ['Comedy', 'Romance', 'Adventure', 'Fantasy']
        adult_genres = ['Horror', 'Thriller', 'Crime', 'War', 'Drama', 'Mystery']
        
        # Scoring system
        family_score = (
            sum(1 for kw in family_keywords if kw in title or kw in overview) +
            sum(1 for g in genres if g in family_genres) * 2
        )
        teen_score = (
            sum(1 for kw in teen_keywords if kw in title or kw in overview) +
            sum(1 for g in genres if g in teen_genres)
        )
        adult_score = (
            sum(1 for kw in adult_keywords if kw in title or kw in overview) +
            sum(1 for g in genres if g in adult_genres) * 1.5
        )
        
        if family_score >= max(teen_score, adult_score):
            return 'family'
        elif adult_score > teen_score:
            return 'adult'
        else:
            return 'teen'
    
    def calculate_enhanced_popularity_score(self):
        """Enhanced popularity calculation using multiple factors"""
        features = ['vote_average', 'vote_count']
        if 'popularity' in self.movies_df.columns:
            features.append('popularity')
        
        # Normalize features
        scaler = MinMaxScaler()
        normalized_features = scaler.fit_transform(self.movies_df[features])
        
        # Weighted combination
        if len(features) == 3:
            weights = [0.4, 0.3, 0.3]  # rating, vote_count, popularity
        else:
            weights = [0.6, 0.4]  # rating, vote_count
        
        popularity = np.average(normalized_features, axis=1, weights=weights)
        return popularity
    
    def calculate_quality_score(self):
        """Calculate a quality score based on multiple factors"""
        # Bayesian average rating
        C = self.movies_df['vote_average'].mean()  # Mean rating across all movies
        m = self.movies_df['vote_count'].quantile(0.7)  # Minimum votes required
        
        v = self.movies_df['vote_count']
        R = self.movies_df['vote_average']
        
        quality_score = (v / (v + m) * R) + (m / (v + m) * C)
        return MinMaxScaler().fit_transform(quality_score.values.reshape(-1, 1)).flatten()
    
    def create_advanced_features(self):
        """Create advanced feature representations"""
        print("Creating advanced features...")
        
        # 1. Enhanced content features
        self.processed_movies['content_features'] = (
            self.processed_movies['overview'].fillna('') + ' ' +
            self.processed_movies['genres_str'] + ' ' +
            self.processed_movies['cast_str'] + ' ' +
            self.processed_movies['director_str']
        )
        
        # 2. Create TF-IDF matrix
        tfidf = TfidfVectorizer(
            max_features=8000,
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.8
        )
        self.tfidf_matrix = tfidf.fit_transform(self.processed_movies['content_features'])
        
        # 3. Create genre vectors using one-hot encoding
        all_genres = set()
        for genres in self.processed_movies['genres_list']:
            all_genres.update(genres)
        all_genres = sorted(list(all_genres))
        
        genre_matrix = np.zeros((len(self.processed_movies), len(all_genres)))
        for i, genres in enumerate(self.processed_movies['genres_list']):
            for genre in genres:
                if genre in all_genres:
                    genre_matrix[i, all_genres.index(genre)] = 1
        
        self.genre_vectors = genre_matrix
        
        # 4. Create numerical feature matrix
        numerical_features = ['vote_average', 'vote_count', 'year', 'runtime', 
                            'budget', 'revenue', 'popularity_score', 'quality_score']
        available_features = [f for f in numerical_features if f in self.processed_movies.columns]
        
        feature_matrix = self.processed_movies[available_features].values
        scaler = StandardScaler()
        self.movie_features = scaler.fit_transform(feature_matrix)
        
        # 5. Combine all features
        combined_features = np.hstack([
            self.tfidf_matrix.toarray()[:, :1000],  # Top 1000 TF-IDF features
            self.genre_vectors,
            self.movie_features
        ])
        
        self.combined_features = combined_features
    
    def train_ml_models(self):
        """Train multiple ML models for different recommendation approaches"""
        print("Training ML models...")
        
        # 1. Content-based similarity with cosine similarity
        n_movies = min(3000, len(self.processed_movies))
        self.content_similarity = cosine_similarity(
            self.combined_features[:n_movies], 
            self.combined_features[:n_movies]
        )
        
        # 2. Dimensionality reduction with SVD
        self.svd_model = TruncatedSVD(n_components=50, random_state=42)
        self.reduced_features = self.svd_model.fit_transform(self.combined_features)
        
        # 3. Matrix factorization with NMF
        self.nmf_model = NMF(n_components=30, random_state=42, max_iter=100)
        self.nmf_features = self.nmf_model.fit_transform(np.abs(self.combined_features))
        
        # 4. Clustering for genre-based recommendations
        self.clustering_model = KMeans(n_clusters=15, random_state=42, n_init=10)
        self.movie_clusters = self.clustering_model.fit_predict(self.reduced_features)
        self.processed_movies['cluster'] = self.movie_clusters
        
        # 5. k-NN model for similarity-based recommendations
        self.knn_model = NearestNeighbors(
            n_neighbors=20, 
            metric='cosine', 
            algorithm='brute'
        )
        self.knn_model.fit(self.reduced_features)
        
        # 6. Rating prediction model
        if len(self.processed_movies) > 100:
            try:
                # Prepare features for rating prediction
                X = self.reduced_features
                y = self.processed_movies['vote_average'].values
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                
                self.rating_predictor = RandomForestRegressor(
                    n_estimators=100, 
                    random_state=42,
                    n_jobs=-1
                )
                self.rating_predictor.fit(X_train, y_train)
                
                # Predict ratings for all movies
                predicted_ratings = self.rating_predictor.predict(X)
                self.processed_movies['predicted_rating'] = predicted_ratings
                
            except Exception as e:
                print(f"Rating predictor training failed: {e}")
                self.processed_movies['predicted_rating'] = self.processed_movies['vote_average']
    
    def get_content_based_recommendations(self, movie_id, n_recommendations=10):
        """Content-based recommendations using multiple similarity measures"""
        try:
            movie_idx = self.processed_movies[self.processed_movies['id'] == movie_id].index[0]
            
            if movie_idx >= len(self.content_similarity):
                # Fallback to k-NN if not in similarity matrix
                return self.get_knn_recommendations(movie_id, n_recommendations)
            
            # Get similarity scores
            sim_scores = list(enumerate(self.content_similarity[movie_idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            
            # Get top similar movies (excluding the movie itself)
            similar_movies = sim_scores[1:n_recommendations+1]
            movie_indices = [i[0] for i in similar_movies]
            
            return self.format_recommendations(movie_indices, similar_movies)
            
        except Exception as e:
            print(f"Content-based recommendation error: {e}")
            return self.get_popular_movies(n_recommendations)
    
    def get_knn_recommendations(self, movie_id, n_recommendations=10):
        """k-NN based recommendations"""
        try:
            movie_idx = self.processed_movies[self.processed_movies['id'] == movie_id].index[0]
            
            # Find similar movies using k-NN
            distances, indices = self.knn_model.kneighbors(
                [self.reduced_features[movie_idx]], 
                n_neighbors=n_recommendations+1
            )
            
            # Exclude the movie itself
            similar_indices = indices[0][1:]
            similarity_scores = 1 - distances[0][1:]  # Convert distance to similarity
            
            similar_movies = list(zip(similar_indices, similarity_scores))
            return self.format_recommendations(similar_indices, similar_movies)
            
        except Exception as e:
            print(f"k-NN recommendation error: {e}")
            return self.get_popular_movies(n_recommendations)
    
    def get_cluster_based_recommendations(self, movie_id, n_recommendations=10):
        """Cluster-based recommendations"""
        try:
            movie_row = self.processed_movies[self.processed_movies['id'] == movie_id].iloc[0]
            movie_cluster = movie_row['cluster']
            
            # Get movies from the same cluster
            cluster_movies = self.processed_movies[
                (self.processed_movies['cluster'] == movie_cluster) &
                (self.processed_movies['id'] != movie_id)
            ]
            
            # Sort by quality score
            cluster_movies = cluster_movies.nlargest(n_recommendations, 'quality_score')
            
            return [{
                'id': int(row['id']),
                'title': row['title'],
                'year': int(row['year']) if pd.notna(row['year']) else None,
                'rating': float(row['vote_average']),
                'predicted_rating': float(row.get('predicted_rating', row['vote_average'])),
                'genres': row['genres_list'],
                'cast': row['cast_list'][:3],
                'directors': row['director_list'],
                'similarity_score': float(row['quality_score']),
                'recommendation_type': 'cluster_based'
            } for _, row in cluster_movies.iterrows()]
            
        except Exception as e:
            print(f"Cluster-based recommendation error: {e}")
            return self.get_popular_movies(n_recommendations)
    
    def get_hybrid_recommendations(self, user_prefs, n_recommendations=15):
        """Hybrid recommendations combining multiple approaches"""
        recommendations = []
        
        # 1. Get content-based recommendations if user has favorite movies
        if user_prefs.get('favorite_movies'):
            for movie_id in user_prefs['favorite_movies'][:2]:  # Use top 2 favorites
                content_recs = self.get_content_based_recommendations(movie_id, 5)
                recommendations.extend(content_recs)
        
        # 2. Get preference-based recommendations
        pref_recs = self.get_preference_based_recommendations(user_prefs, 10)
        recommendations.extend(pref_recs)
        
        # 3. Get cluster-based diversity
        if user_prefs.get('favorite_movies'):
            cluster_recs = self.get_cluster_based_recommendations(
                user_prefs['favorite_movies'][0], 5
            )
            recommendations.extend(cluster_recs)
        
        # 4. Remove duplicates and combine scores
        seen_ids = set()
        unique_recs = []
        
        for rec in recommendations:
            if rec['id'] not in seen_ids:
                seen_ids.add(rec['id'])
                unique_recs.append(rec)
        
        # 5. Re-score and sort
        for rec in unique_recs:
            hybrid_score = self.calculate_hybrid_score(rec, user_prefs)
            rec['hybrid_score'] = hybrid_score
        
        # Sort by hybrid score and return top N
        unique_recs.sort(key=lambda x: x['hybrid_score'], reverse=True)
        return unique_recs[:n_recommendations]
    
    def calculate_hybrid_score(self, recommendation, user_prefs):
        """Calculate hybrid recommendation score"""
        score = 0.0
        
        # Base rating score
        score += recommendation['rating'] * 0.2
        
        # Similarity score
        score += recommendation.get('similarity_score', 0.5) * 0.3
        
        # Predicted rating
        score += recommendation.get('predicted_rating', recommendation['rating']) * 0.2
        
        # User preference matching
        pref_score = self.calculate_preference_match(recommendation, user_prefs)
        score += pref_score * 0.3
        
        return score
    
    def calculate_preference_match(self, movie, user_prefs):
        """Calculate how well a movie matches user preferences"""
        match_score = 0.0
        
        # Genre matching
        movie_genres = set(movie['genres'])
        fav_genres = set(user_prefs.get('favorite_genres', []))
        if fav_genres:
            genre_overlap = len(movie_genres & fav_genres) / len(fav_genres)
            match_score += genre_overlap * 3.0
        
        # Actor matching
        movie_cast = set(movie['cast'])
        fav_actors = set(user_prefs.get('favorite_actors', []))
        if fav_actors:
            actor_overlap = len(movie_cast & fav_actors) / len(fav_actors)
            match_score += actor_overlap * 2.0
        
        # Director matching
        movie_directors = set(movie.get('directors', []))
        fav_directors = set(user_prefs.get('favorite_directors', []))
        if fav_directors:
            director_overlap = len(movie_directors & fav_directors) / len(fav_directors)
            match_score += director_overlap * 1.5
        
        # Year preference
        if user_prefs.get('min_year') and movie['year']:
            if movie['year'] >= user_prefs['min_year']:
                match_score += 0.5
        
        # Rating preference
        if user_prefs.get('min_rating') and movie['rating'] >= user_prefs['min_rating']:
            match_score += 0.5
        
        return min(match_score, 5.0)  # Cap at 5.0
    
    def get_preference_based_recommendations(self, user_prefs, limit=15):
        """Enhanced preference-based recommendations"""
        candidates = self.processed_movies.copy()
        
        # Apply filters
        if user_prefs.get('min_rating', 0) > 0:
            candidates = candidates[candidates['vote_average'] >= user_prefs['min_rating']]
        
        if user_prefs.get('min_year', 0) > 0:
            candidates = candidates[candidates['year'] >= user_prefs['min_year']]
        
        if user_prefs.get('max_year', 0) > 0:
            candidates = candidates[candidates['year'] <= user_prefs['max_year']]
        
        if user_prefs.get('age_group'):
            if user_prefs['age_group'] == 'family':
                candidates = candidates[candidates['age_category'].isin(['family', 'teen'])]
            elif user_prefs['age_group'] == 'teen':
                candidates = candidates[candidates['age_category'].isin(['teen', 'adult'])]
            elif user_prefs['age_group'] == 'adult':
                candidates = candidates[candidates['age_category'] == 'adult']
        
        # Score movies
        candidates = self.score_movies_advanced(candidates, user_prefs)
        
        # Get top recommendations
        recommendations = candidates.nlargest(limit, 'recommendation_score')
        
        return [{
            'id': int(row['id']),
            'title': row['title'],
            'year': int(row['year']) if pd.notna(row['year']) else None,
            'rating': float(row['vote_average']),
            'predicted_rating': float(row.get('predicted_rating', row['vote_average'])),
            'genres': row['genres_list'],
            'cast': row['cast_list'][:3],
            'directors': row['director_list'],
            'similarity_score': float(row['recommendation_score']),
            'match_reasons': self.get_enhanced_match_reasons(row, user_prefs),
            'recommendation_type': 'preference_based'
        } for _, row in recommendations.iterrows()]
    
    def score_movies_advanced(self, movies, user_prefs):
        """Advanced movie scoring algorithm"""
        movies = movies.copy()
        movies['recommendation_score'] = 0.0
        
        # Genre matching with TF-IDF weighting
        fav_genres = set(user_prefs.get('favorite_genres', []))
        if fav_genres:
            movies['genre_score'] = movies['genres_list'].apply(
                lambda x: self.calculate_weighted_genre_score(x, fav_genres)
            )
            movies['recommendation_score'] += movies['genre_score'] * 4.0
        
        # Actor matching with popularity weighting
        fav_actors = set(user_prefs.get('favorite_actors', []))
        if fav_actors:
            movies['actor_score'] = movies['cast_list'].apply(
                lambda x: len(set(x) & fav_actors) / max(len(fav_actors), 1)
            )
            movies['recommendation_score'] += movies['actor_score'] * 3.0
        
        # Director matching
        fav_directors = set(user_prefs.get('favorite_directors', []))
        if fav_directors:
            movies['director_score'] = movies['director_list'].apply(
                lambda x: len(set(x) & fav_directors) / max(len(fav_directors), 1)
            )
            movies['recommendation_score'] += movies['director_score'] * 2.0
        
        # Quality scores
        movies['recommendation_score'] += movies['quality_score'] * 2.0
        movies['recommendation_score'] += movies['popularity_score'] * 1.0
        
        # Predicted rating boost
        if 'predicted_rating' in movies.columns:
            movies['recommendation_score'] += (movies['predicted_rating'] - 5) * 0.5
        
        # Diversity bonus (prefer movies from different years)
        current_year = datetime.now().year
        movies['diversity_score'] = 1 - abs(movies['year'] - current_year/2) / current_year
        movies['recommendation_score'] += movies['diversity_score'] * 0.3
        
        return movies
    
    def calculate_weighted_genre_score(self, movie_genres, fav_genres):
        """Calculate weighted genre score based on genre popularity"""
        if not fav_genres or not movie_genres:
            return 0.0
        
        overlap = set(movie_genres) & fav_genres
        if not overlap:
            return 0.0
        
        # Weight by inverse frequency (rare genres get higher scores)
        total_score = 0.0
        for genre in overlap:
            genre_frequency = sum(1 for genres in self.processed_movies['genres_list'] 
                                if genre in genres) / len(self.processed_movies)
            genre_weight = 1 / (genre_frequency + 0.01)  # Avoid division by zero
            total_score += genre_weight
        
        return total_score / len(fav_genres)
    
    def get_enhanced_match_reasons(self, movie, user_prefs):
        """Get enhanced reasons why a movie matches user preferences"""
        reasons = []
        
        # Genre matching
        movie_genres = set(movie['genres_list'])
        fav_genres = set(user_prefs.get('favorite_genres', []))
        genre_matches = movie_genres & fav_genres
        if genre_matches:
            reasons.append(f"Matches genres: {', '.join(list(genre_matches)[:2])}")
        
        # Cast matching
        movie_cast = set(movie['cast_list'])
        fav_actors = set(user_prefs.get('favorite_actors', []))
        actor_matches = movie_cast & fav_actors
        if actor_matches:
            reasons.append(f"Features: {', '.join(list(actor_matches)[:2])}")
        
        # Director matching
        movie_directors = set(movie['director_list'])
        fav_directors = set(user_prefs.get('favorite_directors', []))
        director_matches = movie_directors & fav_directors
        if director_matches:
            reasons.append(f"Directed by: {', '.join(list(director_matches)[:1])}")
        
        # Quality indicators
        if movie['vote_average'] >= 8.0:
            reasons.append("Highly rated")
        if movie.get('predicted_rating', 0) > movie['vote_average']:
            reasons.append("AI predicts you'll love this")
        
        return reasons[:3]
    
    def format_recommendations(self, movie_indices, similarity_data):
        """Format recommendations with similarity scores"""
        recommendations = []
        
        for i, (idx, similarity) in enumerate(similarity_data):
            if idx < len(self.processed_movies):
                movie = self.processed_movies.iloc[idx]
                recommendations.append({
                    'id': int(movie['id']),
                    'title': movie['title'],
                    'year': int(movie['year']) if pd.notna(movie['year']) else None,
                    'rating': float(movie['vote_average']),
                    'predicted_rating': float(movie.get('predicted_rating', movie['vote_average'])),
                    'genres': movie['genres_list'],
                    'cast': movie['cast_list'][:3],
                    'directors': movie['director_list'],
                    'similarity_score': float(similarity),
                    'recommendation_type': 'content_based'
                })
        
        return recommendations
    
    def search_movies(self, query, limit=10):
        """Enhanced movie search with fuzzy matching"""
        # Direct title match
        mask = self.processed_movies['title'].str.contains(query, case=False, na=False)
        direct_matches = self.processed_movies[mask]
        
        # If we have enough direct matches, return them
        if len(direct_matches) >= limit:
            results = direct_matches.nlargest(limit, 'popularity_score')
        else:
            # Add fuzzy matching for cast, director, and overview
            cast_mask = self.processed_movies['cast_str'].str.contains(query, case=False, na=False)
            director_mask = self.processed_movies['director_str'].str.contains(query, case=False, na=False)
            overview_mask = self.processed_movies['overview'].str.contains(query, case=False, na=False)
            
            # Combine all matches
            all_matches = self.processed_movies[mask | cast_mask | director_mask | overview_mask]
            
            # Score matches based on relevance
            all_matches = all_matches.copy()
            all_matches['search_score'] = 0.0
            
            # Higher score for title matches
            all_matches.loc[mask, 'search_score'] += 3.0
            all_matches.loc[cast_mask, 'search_score'] += 2.0
            all_matches.loc[director_mask, 'search_score'] += 2.0
            all_matches.loc[overview_mask, 'search_score'] += 1.0
            
            # Add popularity boost
            all_matches['search_score'] += all_matches['popularity_score']
            
            results = all_matches.nlargest(limit, 'search_score')
        
        return [{
            'id': int(row['id']),
            'title': row['title'],
            'year': int(row['year']) if pd.notna(row['year']) else None,
            'rating': float(row['vote_average']),
            'genres': row['genres_list'],
            'cast': row['cast_list'][:3],
            'directors': row['director_list'],
            'overview': row['overview'][:200] + '...' if len(str(row['overview'])) > 200 else row['overview']
        } for _, row in results.iterrows()]
    
    def get_all_genres(self):
        """Get all unique genres with counts"""
        genre_counts = Counter()
        for genres_list in self.processed_movies['genres_list']:
            genre_counts.update(genres_list)
        
        return [{'name': genre, 'count': count} 
                for genre, count in genre_counts.most_common()]
    
    def get_popular_actors(self, limit=100):
        """Get popular actors with movie counts"""
        actor_counts = Counter()
        for cast_list in self.processed_movies['cast_list']:
            actor_counts.update(cast_list)
        
        return [{'name': actor, 'movie_count': count} 
                for actor, count in actor_counts.most_common(limit)]
    
    def get_popular_directors(self, limit=50):
        """Get popular directors with movie counts"""
        director_counts = Counter()
        for director_list in self.processed_movies['director_list']:
            director_counts.update(director_list)
        
        return [{'name': director, 'movie_count': count} 
                for director, count in director_counts.most_common(limit)]
    
    def get_trending_movies(self, limit=20, time_window='recent'):
        """Get trending movies with different criteria"""
        if time_window == 'recent':
            # Recent popular movies (last 10 years)
            current_year = datetime.now().year
            recent_movies = self.processed_movies[
                self.processed_movies['year'] >= current_year - 10
            ]
            trending = recent_movies.nlargest(limit, 'popularity_score')
        elif time_window == 'all_time':
            # All-time popular movies
            trending = self.processed_movies.nlargest(limit, 'quality_score')
        else:
            # Classic movies (older but highly rated)
            classic_movies = self.processed_movies[
                (self.processed_movies['year'] <= 2000) &
                (self.processed_movies['vote_average'] >= 7.5)
            ]
            trending = classic_movies.nlargest(limit, 'vote_average')
        
        return [{
            'id': int(row['id']),
            'title': row['title'],
            'year': int(row['year']) if pd.notna(row['year']) else None,
            'rating': float(row['vote_average']),
            'genres': row['genres_list'],
            'cast': row['cast_list'][:3],
            'directors': row['director_list'],
            'popularity_score': float(row['popularity_score']),
            'quality_score': float(row['quality_score'])
        } for _, row in trending.iterrows()]
    
    def get_popular_movies(self, limit=20):
        """Get popular movies as fallback"""
        popular = self.processed_movies.nlargest(limit, 'popularity_score')
        
        return [{
            'id': int(row['id']),
            'title': row['title'],
            'year': int(row['year']) if pd.notna(row['year']) else None,
            'rating': float(row['vote_average']),
            'genres': row['genres_list'],
            'cast': row['cast_list'][:3],
            'directors': row['director_list'],
            'similarity_score': float(row['popularity_score']),
            'recommendation_type': 'popular'
        } for _, row in popular.iterrows()]
    
    def get_recommendations(self, user_prefs, limit=15, recommendation_type='hybrid'):
        """Main recommendation function with multiple strategies"""
        if recommendation_type == 'hybrid':
            return self.get_hybrid_recommendations(user_prefs, limit)
        elif recommendation_type == 'content':
            if user_prefs.get('favorite_movies'):
                return self.get_content_based_recommendations(
                    user_prefs['favorite_movies'][0], limit
                )
            else:
                return self.get_preference_based_recommendations(user_prefs, limit)
        elif recommendation_type == 'collaborative':
            # Placeholder for collaborative filtering (would need user ratings data)
            return self.get_preference_based_recommendations(user_prefs, limit)
        else:
            return self.get_preference_based_recommendations(user_prefs, limit)
    
    def get_movie_details(self, movie_id):
        """Get detailed information about a specific movie"""
        try:
            movie = self.processed_movies[self.processed_movies['id'] == movie_id].iloc[0]
            
            # Get similar movies
            similar_movies = self.get_content_based_recommendations(movie_id, 5)
            
            return {
                'id': int(movie['id']),
                'title': movie['title'],
                'year': int(movie['year']) if pd.notna(movie['year']) else None,
                'rating': float(movie['vote_average']),
                'vote_count': int(movie['vote_count']),
                'predicted_rating': float(movie.get('predicted_rating', movie['vote_average'])),
                'genres': movie['genres_list'],
                'cast': movie['cast_list'],
                'directors': movie['director_list'],
                'overview': movie['overview'],
                'runtime': int(movie['runtime']) if pd.notna(movie['runtime']) else None,
                'budget': int(movie['budget']) if pd.notna(movie['budget']) else None,
                'revenue': int(movie['revenue']) if pd.notna(movie['revenue']) else None,
                'age_category': movie['age_category'],
                'cluster': int(movie['cluster']),
                'popularity_score': float(movie['popularity_score']),
                'quality_score': float(movie['quality_score']),
                'similar_movies': similar_movies
            }
        except Exception as e:
            print(f"Error getting movie details: {e}")
            return None
    
    def get_genre_recommendations(self, genre, limit=20):
        """Get recommendations for a specific genre"""
        genre_movies = self.processed_movies[
            self.processed_movies['genres_list'].apply(lambda x: genre in x)
        ]
        
        if len(genre_movies) == 0:
            return []
        
        # Sort by quality score
        recommendations = genre_movies.nlargest(limit, 'quality_score')
        
        return [{
            'id': int(row['id']),
            'title': row['title'],
            'year': int(row['year']) if pd.notna(row['year']) else None,
            'rating': float(row['vote_average']),
            'genres': row['genres_list'],
            'cast': row['cast_list'][:3],
            'directors': row['director_list'],
            'quality_score': float(row['quality_score'])
        } for _, row in recommendations.iterrows()]
    
    def get_actor_filmography(self, actor_name, limit=20):
        """Get movies featuring a specific actor"""
        actor_movies = self.processed_movies[
            self.processed_movies['cast_list'].apply(lambda x: actor_name in x)
        ]
        
        if len(actor_movies) == 0:
            return []
        
        # Sort by rating and year
        filmography = actor_movies.nlargest(limit, 'vote_average')
        
        return [{
            'id': int(row['id']),
            'title': row['title'],
            'year': int(row['year']) if pd.notna(row['year']) else None,
            'rating': float(row['vote_average']),
            'genres': row['genres_list'],
            'role': 'Actor',  # Could be enhanced to show specific role
            'directors': row['director_list']
        } for _, row in filmography.iterrows()]
    
    def get_director_filmography(self, director_name, limit=20):
        """Get movies by a specific director"""
        director_movies = self.processed_movies[
            self.processed_movies['director_list'].apply(lambda x: director_name in x)
        ]
        
        if len(director_movies) == 0:
            return []
        
        # Sort by rating and year
        filmography = director_movies.nlargest(limit, 'vote_average')
        
        return [{
            'id': int(row['id']),
            'title': row['title'],
            'year': int(row['year']) if pd.notna(row['year']) else None,
            'rating': float(row['vote_average']),
            'genres': row['genres_list'],
            'cast': row['cast_list'][:3]
        } for _, row in filmography.iterrows()]
    
    def get_recommendations_by_decade(self, decade, limit=20):
        """Get top movies from a specific decade"""
        start_year = decade
        end_year = decade + 9
        
        decade_movies = self.processed_movies[
            (self.processed_movies['year'] >= start_year) &
            (self.processed_movies['year'] <= end_year)
        ]
        
        if len(decade_movies) == 0:
            return []
        
        recommendations = decade_movies.nlargest(limit, 'quality_score')
        
        return [{
            'id': int(row['id']),
            'title': row['title'],
            'year': int(row['year']) if pd.notna(row['year']) else None,
            'rating': float(row['vote_average']),
            'genres': row['genres_list'],
            'cast': row['cast_list'][:3],
            'directors': row['director_list']
        } for _, row in recommendations.iterrows()]
    
    def get_statistics(self):
        """Get dataset statistics"""
        return {
            'total_movies': len(self.processed_movies),
            'total_genres': len(self.get_all_genres()),
            'total_actors': len(self.get_popular_actors(limit=1000)),
            'total_directors': len(self.get_popular_directors(limit=1000)),
            'year_range': {
                'min': int(self.processed_movies['year'].min()),
                'max': int(self.processed_movies['year'].max())
            },
            'rating_range': {
                'min': float(self.processed_movies['vote_average'].min()),
                'max': float(self.processed_movies['vote_average'].max()),
                'avg': float(self.processed_movies['vote_average'].mean())
            },
            'top_genres': [g['name'] for g in self.get_all_genres()[:10]],
            'ml_models': {
                'svd_components': self.svd_model.n_components if self.svd_model else 0,
                'nmf_components': self.nmf_model.n_components if self.nmf_model else 0,
                'clusters': len(set(self.movie_clusters)) if hasattr(self, 'movie_clusters') else 0,
                'features_dim': self.combined_features.shape[1] if hasattr(self, 'combined_features') else 0
            }
        }
    
    def create_sample_data(self):
        """Create enhanced sample data if real dataset loading fails"""
        sample_movies = [
            {
                'id': 1, 'title': 'The Shawshank Redemption', 
                'overview': 'Two imprisoned men bond over years, finding solace and eventual redemption through acts of common decency.',
                'genres_list': ['Drama'], 'cast_list': ['Tim Robbins', 'Morgan Freeman', 'Bob Gunton'],
                'director_list': ['Frank Darabont'], 'vote_average': 9.3, 'vote_count': 2000000, 
                'year': 1994, 'runtime': 142, 'budget': 25000000, 'revenue': 16000000,
                'age_category': 'adult', 'genres_str': 'Drama', 'cast_str': 'Tim Robbins Morgan Freeman Bob Gunton',
                'director_str': 'Frank Darabont', 'popularity_score': 0.95, 'quality_score': 0.98,
                'cluster': 0, 'predicted_rating': 9.2
            },
            {
                'id': 2, 'title': 'The Godfather', 
                'overview': 'The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant son.',
                'genres_list': ['Crime', 'Drama'], 'cast_list': ['Marlon Brando', 'Al Pacino', 'James Caan'],
                'director_list': ['Francis Ford Coppola'], 'vote_average': 9.2, 'vote_count': 1500000,
                'year': 1972, 'runtime': 175, 'budget': 6000000, 'revenue': 245000000,
                'age_category': 'adult', 'genres_str': 'Crime Drama', 'cast_str': 'Marlon Brando Al Pacino James Caan',
                'director_str': 'Francis Ford Coppola', 'popularity_score': 0.93, 'quality_score': 0.96,
                'cluster': 0, 'predicted_rating': 9.1
            },
            {
                'id': 3, 'title': 'Pulp Fiction', 
                'overview': 'The lives of two mob hitmen, a boxer, a gangster and his wife intertwine in four tales of violence and redemption.',
                'genres_list': ['Crime', 'Drama'], 'cast_list': ['John Travolta', 'Uma Thurman', 'Samuel L. Jackson'],
                'director_list': ['Quentin Tarantino'], 'vote_average': 8.9, 'vote_count': 1800000,
                'year': 1994, 'runtime': 154, 'budget': 8000000, 'revenue': 214000000,
                'age_category': 'adult', 'genres_str': 'Crime Drama', 'cast_str': 'John Travolta Uma Thurman Samuel L. Jackson',
                'director_str': 'Quentin Tarantino', 'popularity_score': 0.89, 'quality_score': 0.92,
                'cluster': 0, 'predicted_rating': 8.8
            },
            {
                'id': 4, 'title': 'The Dark Knight', 
                'overview': 'When the menace known as the Joker wreaks havoc on Gotham City, Batman must face his greatest psychological challenge.',
                'genres_list': ['Action', 'Crime', 'Drama'], 'cast_list': ['Christian Bale', 'Heath Ledger', 'Aaron Eckhart'],
                'director_list': ['Christopher Nolan'], 'vote_average': 9.0, 'vote_count': 2200000,
                'year': 2008, 'runtime': 152, 'budget': 185000000, 'revenue': 1004000000,
                'age_category': 'teen', 'genres_str': 'Action Crime Drama', 'cast_str': 'Christian Bale Heath Ledger Aaron Eckhart',
                'director_str': 'Christopher Nolan', 'popularity_score': 0.91, 'quality_score': 0.94,
                'cluster': 1, 'predicted_rating': 8.9
            },
            {
                'id': 5, 'title': 'Forrest Gump', 
                'overview': 'The presidencies of Kennedy and Johnson, Vietnam, Watergate, and other history unfold through the perspective of an Alabama man.',
                'genres_list': ['Drama', 'Romance'], 'cast_list': ['Tom Hanks', 'Robin Wright', 'Gary Sinise'],
                'director_list': ['Robert Zemeckis'], 'vote_average': 8.8, 'vote_count': 1900000,
                'year': 1994, 'runtime': 142, 'budget': 55000000, 'revenue': 678000000,
                'age_category': 'teen', 'genres_str': 'Drama Romance', 'cast_str': 'Tom Hanks Robin Wright Gary Sinise',
                'director_str': 'Robert Zemeckis', 'popularity_score': 0.87, 'quality_score': 0.90,
                'cluster': 2, 'predicted_rating': 8.7
            }
        ]
        
        self.processed_movies = pd.DataFrame(sample_movies)
        
        # Create simple feature matrices for sample data
        self.combined_features = np.random.random((len(sample_movies), 100))
        self.reduced_features = np.random.random((len(sample_movies), 50))
        self.movie_clusters = np.array([0, 0, 0, 1, 2])
        self.content_similarity = np.eye(len(sample_movies))  # Identity matrix as placeholder
        
        print("Using enhanced sample data due to dataset loading error")

# Example usage and testing
if __name__ == "__main__":
    # Initialize the enhanced recommender
    recommender = EnhancedMovieRecommender()
    
    # Example user preferences
    user_prefs = {
        'favorite_genres': ['Drama', 'Crime'],
        'favorite_actors': ['Al Pacino', 'Robert De Niro'],
        'favorite_directors': ['Martin Scorsese'],
        'favorite_movies': [1, 2],  # The Shawshank Redemption, The Godfather
        'min_rating': 7.0,
        'min_year': 1990,
        'age_group': 'adult'
    }
    
    # Test different recommendation types
    print("=== Hybrid Recommendations ===")
    hybrid_recs = recommender.get_recommendations(user_prefs, limit=10, recommendation_type='hybrid')
    for rec in hybrid_recs[:5]:
        print(f"{rec['title']} ({rec['year']}) - Rating: {rec['rating']:.1f}")
    
    print("\n=== Content-Based Recommendations ===")
    content_recs = recommender.get_content_based_recommendations(1, 5)  # Based on Shawshank
    for rec in content_recs:
        print(f"{rec['title']} ({rec['year']}) - Similarity: {rec['similarity_score']:.3f}")
    
    print("\n=== Search Results ===")
    search_results = recommender.search_movies("Godfather", 3)
    for result in search_results:
        print(f"{result['title']} ({result['year']}) - {', '.join(result['genres'])}")
    
    print("\n=== Dataset Statistics ===")
    stats = recommender.get_statistics()
    print(f"Total Movies: {stats['total_movies']}")
    print(f"ML Models: {stats['ml_models']}")
    print(f"Top Genres: {', '.join(stats['top_genres'][:5])}")