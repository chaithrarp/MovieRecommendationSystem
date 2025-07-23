const API_BASE_URL = 'http://localhost:5000/api';

// TMDB Configuration
const TMDB_API_KEY = 'bed3c0900197c6e79571bffc27d8c501'; // Get from https://www.themoviedb.org/settings/api
const TMDB_BASE_URL = 'https://api.themoviedb.org/3';
const TMDB_IMAGE_BASE_URL = 'https://image.tmdb.org/t/p';
const POSTER_SIZE = 'w500'; // Available sizes: w92, w154, w185, w342, w500, w780, original

let allGenres = [];
let allActors = [];
let selectedGenres = [];
let selectedActors = [];

// DOM elements
const navButtons = document.querySelectorAll('.nav-btn');
const sections = document.querySelectorAll('.section');
const preferencesForm = document.getElementById('preferences-form');
const minRatingSlider = document.getElementById('min-rating');
const minYearSlider = document.getElementById('min-year');
const ratingValue = document.getElementById('rating-value');
const yearValue = document.getElementById('year-value');
const actorSearch = document.getElementById('actor-search');
const movieSearch = document.getElementById('movie-search');
const searchBtn = document.getElementById('search-btn');
const modal = document.getElementById('movie-modal');
const loadingOverlay = document.getElementById('loading-overlay');

// Cache for movie posters to avoid repeated API calls
const posterCache = new Map();

// Initialize app
document.addEventListener('DOMContentLoaded', function() {
    setupEventListeners();
    loadInitialData();
    updateSliderValues();
});

// Event Listeners
function setupEventListeners() {
    // Navigation
    navButtons.forEach(btn => {
        btn.addEventListener('click', (e) => {
            const section = e.target.dataset.section;
            showSection(section);
        });
    });

    // Preferences form
    preferencesForm.addEventListener('submit', handlePreferencesSubmit);

    // Sliders
    minRatingSlider.addEventListener('input', updateSliderValues);
    minYearSlider.addEventListener('input', updateSliderValues);

    // Actor search
    actorSearch.addEventListener('input', debounce(handleActorSearch, 300));

    // Movie search
    movieSearch.addEventListener('input', debounce(handleMovieSearch, 500));
    searchBtn.addEventListener('click', () => {
        const query = movieSearch.value.trim();
        if (query) searchMovies(query);
    });

    // Modal close
    document.querySelector('.close').addEventListener('click', closeModal);
    modal.addEventListener('click', (e) => {
        if (e.target === modal) closeModal();
    });

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') closeModal();
    });
}

// Get movie poster from TMDB
async function getMoviePoster(title, year = null) {
    const cacheKey = `${title}-${year}`;
    
    // Check cache first
    if (posterCache.has(cacheKey)) {
        return posterCache.get(cacheKey);
    }

    try {
        // Search for the movie on TMDB
        const searchQuery = encodeURIComponent(title);
        const searchUrl = `${TMDB_BASE_URL}/search/movie?api_key=${TMDB_API_KEY}&query=${searchQuery}${year ? `&year=${year}` : ''}`;
        
        const response = await fetch(searchUrl);
        const data = await response.json();

        if (data.results && data.results.length > 0) {
            const movie = data.results[0];
            if (movie.poster_path) {
                const posterUrl = `${TMDB_IMAGE_BASE_URL}/${POSTER_SIZE}${movie.poster_path}`;
                posterCache.set(cacheKey, posterUrl);
                return posterUrl;
            }
        }
        
        // Cache null result to avoid repeated failed requests
        posterCache.set(cacheKey, null);
        return null;
    } catch (error) {
        console.error('Error fetching poster:', error);
        posterCache.set(cacheKey, null);
        return null;
    }
}

// Alternative: Get poster from OMDb API
async function getMoviePosterOMDb(title, year = null) {
    const OMDB_API_KEY = 'YOUR_OMDB_API_KEY'; // Get from http://www.omdbapi.com/
    const cacheKey = `${title}-${year}`;
    
    if (posterCache.has(cacheKey)) {
        return posterCache.get(cacheKey);
    }

    try {
        const searchQuery = encodeURIComponent(title);
        const url = `https://www.omdbapi.com/?apikey=${OMDB_API_KEY}&t=${searchQuery}${year ? `&y=${year}` : ''}`;
        
        const response = await fetch(url);
        const data = await response.json();

        if (data.Response === 'True' && data.Poster && data.Poster !== 'N/A') {
            posterCache.set(cacheKey, data.Poster);
            return data.Poster;
        }
        
        posterCache.set(cacheKey, null);
        return null;
    } catch (error) {
        console.error('Error fetching poster from OMDb:', error);
        posterCache.set(cacheKey, null);
        return null;
    }
}

// Navigation
function showSection(sectionName) {
    // Update navigation
    navButtons.forEach(btn => {
        btn.classList.toggle('active', btn.dataset.section === sectionName);
    });

    // Update sections
    sections.forEach(section => {
        section.classList.toggle('active', section.id === sectionName);
    });
}

// Load initial data
async function loadInitialData() {
    try {
        await Promise.all([
            loadGenres(),
            loadActors(),
            loadTrendingMovies()
        ]);
    } catch (error) {
        console.error('Error loading initial data:', error);
        showError('Failed to load initial data. Please refresh the page.');
    }
}

// Load genres
async function loadGenres() {
    try {
        const response = await fetch(`${API_BASE_URL}/movies/genres`);
        const data = await response.json();
        
        if (response.ok) {
            allGenres = data.genres;
            displayGenres();
        } else {
            throw new Error(data.error || 'Failed to load genres');
        }
    } catch (error) {
        console.error('Error loading genres:', error);
        // Fallback genres
        allGenres = ['Action', 'Comedy', 'Drama', 'Horror', 'Romance', 'Sci-Fi', 'Thriller'];
        displayGenres();
    }
}

// Display genres
function displayGenres() {
    const container = document.getElementById('genres-container');
    container.innerHTML = '';

    allGenres.forEach(genre => {
        const tag = document.createElement('div');
        tag.className = 'tag';
        tag.textContent = genre;
        tag.addEventListener('click', () => toggleGenre(genre, tag));
        container.appendChild(tag);
    });
}

// Toggle genre selection
function toggleGenre(genre, element) {
    if (selectedGenres.includes(genre)) {
        selectedGenres = selectedGenres.filter(g => g !== genre);
        element.classList.remove('selected');
    } else {
        selectedGenres.push(genre);
        element.classList.add('selected');
    }
}

// Load actors
async function loadActors() {
    try {
        const response = await fetch(`${API_BASE_URL}/movies/actors`);
        const data = await response.json();
        
        if (response.ok) {
            allActors = data.actors;
        } else {
            throw new Error(data.error || 'Failed to load actors');
        }
    } catch (error) {
        console.error('Error loading actors:', error);
        // Fallback actors
        allActors = ['Leonardo DiCaprio', 'Scarlett Johansson', 'Tom Hanks', 'Meryl Streep', 
                    'Robert Downey Jr.', 'Jennifer Lawrence', 'Brad Pitt', 'Emma Stone'];
    }
}

// Handle actor search
function handleActorSearch(e) {
    const query = e.target.value.toLowerCase().trim();
    const suggestions = document.getElementById('actor-suggestions');

    if (query.length < 2) {
        suggestions.style.display = 'none';
        return;
    }

    const matches = allActors.filter(actor => 
        actor.toLowerCase().includes(query) && !selectedActors.includes(actor)
    ).slice(0, 10);

    if (matches.length === 0) {
        suggestions.style.display = 'none';
        return;
    }

    suggestions.innerHTML = '';
    matches.forEach(actor => {
        const item = document.createElement('div');
        item.className = 'suggestion-item';
        item.textContent = actor;
        item.addEventListener('click', () => selectActor(actor));
        suggestions.appendChild(item);
    });

    suggestions.style.display = 'block';
}

// Select actor
function selectActor(actor) {
    if (!selectedActors.includes(actor)) {
        selectedActors.push(actor);
        displaySelectedActors();
        document.getElementById('actor-search').value = '';
        document.getElementById('actor-suggestions').style.display = 'none';
    }
}

// Display selected actors
function displaySelectedActors() {
    const container = document.getElementById('selected-actors');
    container.innerHTML = '';

    selectedActors.forEach(actor => {
        const tag = document.createElement('div');
        tag.className = 'selected-tag';
        tag.innerHTML = `
            <span>${actor}</span>
            <button class="remove-tag" onclick="removeActor('${actor}')">&times;</button>
        `;
        container.appendChild(tag);
    });
}

// Remove actor
function removeActor(actor) {
    selectedActors = selectedActors.filter(a => a !== actor);
    displaySelectedActors();
}

// Update slider values
function updateSliderValues() {
    ratingValue.textContent = minRatingSlider.value;
    yearValue.textContent = minYearSlider.value;
}

// Handle preferences form submission
async function handlePreferencesSubmit(e) {
    e.preventDefault();
    
    if (selectedGenres.length === 0 && selectedActors.length === 0) {
        showError('Please select at least one genre or actor to get recommendations.');
        return;
    }

    const preferences = {
        genres: selectedGenres,
        actors: selectedActors,
        age_group: document.querySelector('input[name="age_group"]:checked').value,
        min_rating: parseFloat(minRatingSlider.value),
        min_year: parseInt(minYearSlider.value)
    };

    await getRecommendations(preferences);
}

// Get recommendations
async function getRecommendations(preferences) {
    showLoading(true);
    
    try {
        const response = await fetch(`${API_BASE_URL}/recommend`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(preferences)
        });

        const data = await response.json();

        if (response.ok) {
            displayRecommendations(data.recommendations);
            document.getElementById('recommendations-section').style.display = 'block';
            document.getElementById('recommendations-section').scrollIntoView({ behavior: 'smooth' });
        } else {
            throw new Error(data.error || 'Failed to get recommendations');
        }
    } catch (error) {
        console.error('Error getting recommendations:', error);
        showError('Failed to get recommendations. Please try again.');
    } finally {
        showLoading(false);
    }
}

// Display recommendations
function displayRecommendations(movies) {
    const container = document.getElementById('recommendations');
    container.innerHTML = '';

    if (movies.length === 0) {
        container.innerHTML = '<div class="loading">No movies found matching your preferences. Try adjusting your criteria.</div>';
        return;
    }

    movies.forEach(movie => {
        const movieCard = createMovieCard(movie, true);
        container.appendChild(movieCard);
    });
}

// Load trending movies
async function loadTrendingMovies() {
    try {
        const response = await fetch(`${API_BASE_URL}/movies/trending`);
        const data = await response.json();

        if (response.ok) {
            displayMovies(data.movies, 'trending-movies');
        } else {
            throw new Error(data.error || 'Failed to load trending movies');
        }
    } catch (error) {
        console.error('Error loading trending movies:', error);
        document.getElementById('trending-movies').innerHTML = 
            '<div class="loading">Failed to load trending movies</div>';
    }
}

// Handle movie search
function handleMovieSearch(e) {
    const query = e.target.value.trim();
    if (query.length >= 2) {
        searchMovies(query);
    } else {
        document.getElementById('search-results').innerHTML = '';
    }
}

// Search movies
async function searchMovies(query) {
    try {
        const response = await fetch(`${API_BASE_URL}/movies/search?q=${encodeURIComponent(query)}`);
        const data = await response.json();

        if (response.ok) {
            displayMovies(data.movies, 'search-results');
        } else {
            throw new Error(data.error || 'Search failed');
        }
    } catch (error) {
        console.error('Error searching movies:', error);
        document.getElementById('search-results').innerHTML = 
            '<div class="loading">Search failed. Please try again.</div>';
    }
}

// Display movies
function displayMovies(movies, containerId) {
    const container = document.getElementById(containerId);
    container.innerHTML = '';

    if (movies.length === 0) {
        container.innerHTML = '<div class="loading">No movies found</div>';
        return;
    }

    movies.forEach(movie => {
        const movieCard = createMovieCard(movie);
        container.appendChild(movieCard);
    });
}

// Create movie card with poster
function createMovieCard(movie, showMatchReasons = false) {
    const card = document.createElement('div');
    card.className = 'movie-card';
    card.addEventListener('click', () => showMovieDetails(movie));

    const genreTags = movie.genres ? movie.genres.slice(0, 3).map(genre => 
        `<span class="genre-tag">${genre}</span>`
    ).join('') : '';

    const cast = movie.cast && movie.cast.length > 0 ? movie.cast.join(', ') : 'Cast information not available';

    const matchReasons = showMatchReasons && movie.match_reasons ? 
        `<div class="match-reasons">
            ${movie.match_reasons.map(reason => `<span class="match-reason">${reason}</span>`).join('')}
        </div>` : '';

    // Create poster element with placeholder
    const posterElement = document.createElement('div');
    posterElement.className = 'movie-poster';
    posterElement.innerHTML = '<i class="fas fa-film"></i><div class="poster-loading">Loading...</div>';

    card.innerHTML = `
        <div class="movie-info">
            <div class="movie-title">${movie.title}</div>
            <div class="movie-meta">
                <span>${movie.year || 'N/A'}</span>
                <span class="movie-rating">${movie.rating ? movie.rating.toFixed(1) : 'N/A'} ⭐</span>
            </div>
            <div class="movie-genres">${genreTags}</div>
            <div class="movie-cast">${cast}</div>
            ${matchReasons}
        </div>
    `;

    // Insert poster at the beginning
    card.insertBefore(posterElement, card.firstChild);

    // Load poster asynchronously
    loadPosterForCard(posterElement, movie.title, movie.year);

    return card;
}

// Load poster for movie card
async function loadPosterForCard(posterElement, title, year) {
    try {
        const posterUrl = await getMoviePoster(title, year);
        
        if (posterUrl) {
            const img = document.createElement('img');
            img.src = posterUrl;
            img.alt = `${title} poster`;
            img.className = 'poster-image';
            
            // Handle image load
            img.onload = () => {
                posterElement.innerHTML = '';
                posterElement.appendChild(img);
            };
            
            // Handle image error
            img.onerror = () => {
                posterElement.innerHTML = '<i class="fas fa-film"></i><div class="poster-error">No poster</div>';
            };
        } else {
            posterElement.innerHTML = '<i class="fas fa-film"></i><div class="poster-error">No poster</div>';
        }
    } catch (error) {
        console.error('Error loading poster:', error);
        posterElement.innerHTML = '<i class="fas fa-film"></i><div class="poster-error">No poster</div>';
    }
}

// Show movie details
function showMovieDetails(movie) {
    const modalContent = document.getElementById('movie-details');
    
    const cast = movie.cast && movie.cast.length > 0 ? movie.cast.join(', ') : 'Cast information not available';
    const genres = movie.genres ? movie.genres.join(', ') : 'No genres available';

    modalContent.innerHTML = `
        <div class="modal-poster-section">
            <div class="modal-poster" id="modal-poster">
                <i class="fas fa-film"></i>
                <div class="poster-loading">Loading...</div>
            </div>
        </div>
        <div class="modal-info-section">
            <h2>${movie.title}</h2>
            <div class="movie-detail-info">
                <p><strong>Year:</strong> ${movie.year || 'N/A'}</p>
                <p><strong>Rating:</strong> ${movie.rating ? movie.rating.toFixed(1) : 'N/A'} ⭐</p>
                <p><strong>Genres:</strong> ${genres}</p>
                <p><strong>Cast:</strong> ${cast}</p>
                ${movie.score ? `<p><strong>Match Score:</strong> ${movie.score.toFixed(2)}</p>` : ''}
                ${movie.match_reasons ? `<p><strong>Why this matches:</strong> ${movie.match_reasons.join(', ')}</p>` : ''}
            </div>
            <div class="rating-section">
                <h3>Rate this movie:</h3>
                <div class="star-rating">
                    ${[1,2,3,4,5].map(rating => 
                        `<span class="star" data-rating="${rating}">⭐</span>`
                    ).join('')}
                </div>
            </div>
        </div>
    `;

    // Load poster for modal
    const modalPoster = document.getElementById('modal-poster');
    loadPosterForModal(modalPoster, movie.title, movie.year);

    // Add star rating functionality
    const stars = modalContent.querySelectorAll('.star');
    stars.forEach(star => {
        star.addEventListener('click', (e) => {
            const rating = parseInt(e.target.dataset.rating);
            rateMovie(movie.id, rating);
            
            // Visual feedback
            stars.forEach((s, index) => {
                s.style.opacity = index < rating ? '1' : '0.3';
            });
        });
    });

    modal.style.display = 'block';
}

// Load poster for modal
async function loadPosterForModal(posterElement, title, year) {
    try {
        const posterUrl = await getMoviePoster(title, year);
        
        if (posterUrl) {
            const img = document.createElement('img');
            img.src = posterUrl;
            img.alt = `${title} poster`;
            img.className = 'modal-poster-image';
            
            img.onload = () => {
                posterElement.innerHTML = '';
                posterElement.appendChild(img);
            };
            
            img.onerror = () => {
                posterElement.innerHTML = '<i class="fas fa-film"></i><div class="poster-error">No poster available</div>';
            };
        } else {
            posterElement.innerHTML = '<i class="fas fa-film"></i><div class="poster-error">No poster available</div>';
        }
    } catch (error) {
        console.error('Error loading modal poster:', error);
        posterElement.innerHTML = '<i class="fas fa-film"></i><div class="poster-error">No poster available</div>';
    }
}

// Rate movie
async function rateMovie(movieId, rating) {
    try {
        const response = await fetch(`${API_BASE_URL}/movies/rate`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                movie_id: movieId,
                rating: rating
            })
        });

        const data = await response.json();

        if (response.ok) {
            showSuccess(`Thanks for rating! Your rating: ${rating} stars`);
        } else {
            throw new Error(data.error || 'Failed to save rating');
        }
    } catch (error) {
        console.error('Error rating movie:', error);
        showError('Failed to save your rating. Please try again.');
    }
}

// Close modal
function closeModal() {
    modal.style.display = 'none';
}

// Show loading overlay
function showLoading(show) {
    loadingOverlay.style.display = show ? 'flex' : 'none';
}

// Show error message
function showError(message) {
    const errorDiv = document.createElement('div');
    errorDiv.style.cssText = `
        position: fixed;
        top: 100px;
        right: 20px;
        background: #f44336;
        color: white;
        padding: 15px 20px;
        border-radius: 5px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        z-index: 10000;
        max-width: 300px;
        font-size: 14px;
        line-height: 1.4;
        animation: slideIn 0.3s ease-out;
    `;
    errorDiv.innerHTML = `
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <span>${message}</span>
            <button style="background: none; border: none; color: white; font-size: 18px; cursor: pointer; margin-left: 10px;" onclick="this.parentElement.parentElement.remove()">×</button>
        </div>
    `;
    
    document.body.appendChild(errorDiv);
    
    setTimeout(() => {
        if (errorDiv.parentElement) {
            errorDiv.remove();
        }
    }, 5000);
}

// Show success message
function showSuccess(message) {
    const successDiv = document.createElement('div');
    successDiv.style.cssText = `
        position: fixed;
        top: 100px;
        right: 20px;
        background: #4CAF50;
        color: white;
        padding: 15px 20px;
        border-radius: 5px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        z-index: 10000;
        max-width: 300px;
        font-size: 14px;
        line-height: 1.4;
        animation: slideIn 0.3s ease-out;
    `;
    successDiv.innerHTML = `
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <span>${message}</span>
            <button style="background: none; border: none; color: white; font-size: 18px; cursor: pointer; margin-left: 10px;" onclick="this.parentElement.parentElement.remove()">×</button>
        </div>
    `;
    
    document.body.appendChild(successDiv);
    
    setTimeout(() => {
        if (successDiv.parentElement) {
            successDiv.remove();
        }
    }, 3000);
}

// Debounce utility function
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Add CSS for slide-in animation and poster styles
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    .movie-poster {
        position: relative;
        width: 100%;
        height: 300px;
        background: #f0f0f0;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        overflow: hidden;
        margin-bottom: 15px;
    }
    
    .poster-image {
        width: 100%;
        height: 100%;
        object-fit: cover;
        border-radius: 8px;
    }
    
    .poster-loading, .poster-error {
        font-size: 12px;
        color: #666;
        text-align: center;
        margin-top: 5px;
    }
    
    .modal-poster-section {
        flex: 0 0 200px;
        margin-right: 20px;
    }
    
    .modal-poster {
        width: 200px;
        height: 300px;
        background: #f0f0f0;
        border-radius: 8px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        overflow: hidden;
    }
    
    .modal-poster-image {
        width: 100%;
        height: 100%;
        object-fit: cover;
        border-radius: 8px;
    }
    
    .modal-info-section {
        flex: 1;
    }
    
    #movie-details {
        display: flex;
        gap: 20px;
    }
    
    @media (max-width: 768px) {
        #movie-details {
            flex-direction: column;
        }
        
        .modal-poster-section {
            flex: none;
            margin-right: 0;
            margin-bottom: 20px;
        }
        
        .modal-poster {
            width: 150px;
            height: 225px;
            margin: 0 auto;
        }
    }
`;
document.head.appendChild(style);