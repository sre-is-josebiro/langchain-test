import os
from typing import List, Optional, TypedDict, Dict, Any, Annotated
from datetime import datetime

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain_core.runnables import RunnablePassthrough

import httpx

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Constants
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"
TMDB_BASE_URL = "https://www.themoviedb.org/movie/"

# Models
class MovieRecommendation(BaseModel):
    id: int = Field(description="Movie ID")
    title: str = Field(description="Movie title")
    description: str = Field(description="Movie description")
    poster_path: Optional[str] = Field(description="Path to movie poster image")
    backdrop_path: Optional[str] = Field(description="Path to movie backdrop image")
    rating: float = Field(description="Movie rating out of 10", default=0.0)

class MovieRecommendations(BaseModel):
    recommendations: List[MovieRecommendation]
    explanation: str = Field(description="Brief explanation of the recommendations")
    image_base_url: str = Field(description="Base URL for TMDB images", default=TMDB_IMAGE_BASE_URL)
    tmdb_base_url: str = Field(description="Base URL for TMDB movie pages", default=TMDB_BASE_URL)

class PreferenceRequest(BaseModel):
    preferences: str

# Genre mapping to help with common search terms
GENRE_MAPPING = {
    'sci-fi': 'science fiction',
    'scifi': 'science fiction',
    'sf': 'science fiction',
    'superhero': 'action',
    'super hero': 'action',
    'rom com': 'romance',
    'romcom': 'romance',
    'romantic comedy': 'romance',
    'historical': 'history',
    'scary': 'horror',
    'crime': 'crime',
    'detective': 'crime',
    'musical': 'music',
    'biography': 'documentary',
    'bio': 'documentary',
    'war': 'war',
    'kids': 'family',
    'children': 'family',
    'animated': 'animation',
    'anime': 'animation',
}

# Tools
@tool
async def discover_movies(
    genre: str = "", 
    year_from: Optional[int] = None, 
    year_to: Optional[int] = None,
    exclude_animation: bool = False,
    company: Optional[str] = None
) -> Dict[str, Any]:
    """Discover movies based on various filters.
    Args:
        genre: Genre name to filter by (e.g., 'science fiction', 'action', 'drama')
        year_from: Start year for release date range (e.g., 1960)
        year_to: End year for release date range (e.g., 1969)
        exclude_animation: If True, excludes animated movies
        company: Production company name (e.g., 'Disney', 'Warner Bros')
    Returns:
        Dictionary containing list of movies with their details
    """
    async with httpx.AsyncClient() as client:
        # Get genre IDs
        genre_response = await client.get(
            "https://api.themoviedb.org/3/genre/movie/list",
            params={
                "api_key": TMDB_API_KEY,
                "language": "en-US",
            }
        )
        genres = genre_response.json().get("genres", [])
        
        # Map common genre terms
        search_genre = genre.lower().strip()
        if search_genre in GENRE_MAPPING:
            search_genre = GENRE_MAPPING[search_genre]
        
        # Get animation genre ID
        animation_id = next((g["id"] for g in genres if g["name"].lower() == "animation"), None)
        
        # Get specified genre ID
        genre_id = None
        if search_genre:
            # Try exact match first
            genre_id = next((g["id"] for g in genres if g["name"].lower() == search_genre), None)
            # If no exact match, try partial match
            if genre_id is None:
                for g in genres:
                    if search_genre in g["name"].lower() or g["name"].lower() in search_genre:
                        genre_id = g["id"]
                        break

        # Get company ID if specified
        company_id = None
        if company:
            company_response = await client.get(
                "https://api.themoviedb.org/3/search/company",
                params={
                    "api_key": TMDB_API_KEY,
                    "query": company,
                }
            )
            companies = company_response.json().get("results", [])
            if companies:
                company_id = companies[0]["id"]

        # Current year for filtering unreleased movies
        current_year = datetime.now().year
        
        params = {
            "api_key": TMDB_API_KEY,
            "language": "en-US",
            "sort_by": "vote_average.desc",  # Sort by rating
            "include_adult": False,
            "page": 1,
            "vote_count.gte": 100,  # Ensure significant number of votes
            "vote_average.gte": 6.0,  # Minimum rating of 6/10
            "primary_release_date.lte": f"{current_year}-12-31",  # Only released movies
        }
        
        if genre_id:
            params["with_genres"] = genre_id
        if year_from:
            params["primary_release_date.gte"] = f"{year_from}-01-01"
        if year_to:
            params["primary_release_date.lte"] = f"{year_to}-12-31"
        if company_id:
            params["with_companies"] = company_id
        if exclude_animation and animation_id:
            params["without_genres"] = animation_id

        response = await client.get(
            "https://api.themoviedb.org/3/discover/movie",
            params=params,
        )
        data = response.json()
        
        # Get more pages to ensure we have enough high-quality results
        results = []
        for page in range(1, 4):  # Check first 3 pages
            params["page"] = page
            response = await client.get(
                "https://api.themoviedb.org/3/discover/movie",
                params=params,
            )
            data = response.json()
            results.extend(data.get('results', []))
            if len(results) >= 20:  # Stop if we have enough results
                break
        
        # Sort by rating and take top results
        results.sort(key=lambda x: (x.get('vote_average', 0), x.get('vote_count', 0)), reverse=True)
        results = results[:5]
        
        # Filter and format results
        filtered_results = []
        for movie in results:
            if (movie.get("title") and movie.get("poster_path") and 
                movie.get("vote_average", 0) >= 6.0 and movie.get("vote_count", 0) >= 100):
                filtered_results.append({
                    'id': movie['id'],
                    'title': movie['title'],
                    'overview': movie.get('overview', '')[:200],
                    'release_date': movie.get('release_date', ''),
                    'poster_path': movie.get('poster_path'),
                    'backdrop_path': movie.get('backdrop_path'),
                    'vote_average': round(movie.get('vote_average', 0.0), 1),
                    'vote_count': movie.get('vote_count', 0),
                })
        
        return {'results': filtered_results}

@tool
async def search_movies(query: str) -> Dict[str, Any]:
    """Search for movies using TMDB API. Returns a list of movies matching the query.
    Args:
        query: Search query (e.g., 'science fiction', 'star wars', 'avatar')
    Returns:
        Dictionary containing list of movies with their details
    """
    async with httpx.AsyncClient() as client:
        # Current year for filtering unreleased movies
        current_year = datetime.now().year
        
        # First search for movies
        response = await client.get(
            "https://api.themoviedb.org/3/search/movie",
            params={
                "api_key": TMDB_API_KEY,
                "query": query,
                "language": "en-US",
                "page": 1,
                "include_adult": False,
            },
        )
        data = response.json()
        results = data.get('results', [])
        
        # Get more pages if needed
        if len(results) < 20:
            for page in range(2, 4):  # Check up to 3 pages
                response = await client.get(
                    "https://api.themoviedb.org/3/search/movie",
                    params={
                        "api_key": TMDB_API_KEY,
                        "query": query,
                        "language": "en-US",
                        "page": page,
                        "include_adult": False,
                    },
                )
                data = response.json()
                results.extend(data.get('results', []))
                if len(results) >= 20:
                    break
        
        # Sort by rating and vote count
        results.sort(key=lambda x: (x.get('vote_average', 0), x.get('vote_count', 0)), reverse=True)
        results = results[:10]  # Take top 10 for further filtering
        
        # Filter and format results
        filtered_results = []
        for movie in results:
            if (movie.get("title") and movie.get("poster_path") and 
                movie.get("vote_average", 0) >= 6.0 and movie.get("vote_count", 0) >= 100 and
                movie.get("release_date", "").split("-")[0].isdigit() and 
                int(movie.get("release_date", "").split("-")[0]) <= current_year):
                filtered_results.append({
                    'id': movie['id'],
                    'title': movie['title'],
                    'overview': movie.get('overview', '')[:200],
                    'release_date': movie.get('release_date', ''),
                    'poster_path': movie.get('poster_path'),
                    'backdrop_path': movie.get('backdrop_path'),
                    'vote_average': round(movie.get('vote_average', 0.0), 1),
                    'vote_count': movie.get('vote_count', 0),
                })
        
        return {'results': filtered_results[:5]}  # Return top 5 filtered results

# Create the LangChain agent
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0,
)

# Define the prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a knowledgeable movie recommendation assistant. Your task is to recommend movies based on user preferences.
    Follow these rules:
    1. Use the appropriate tool based on the user's request:
       - For genre-based searches (e.g., 'action', 'comedy', 'drama'), use discover_movies with the genre parameter
       - For specific movie titles or actors, use search_movies
       - For complex queries (e.g., 'movies like Star Wars'), use search_movies
    2. When using discover_movies:
       - For live-action movies, set exclude_animation=True
       - For Disney movies, set company='Disney'
       - For date ranges, use both year_from and year_to
       - For genres, use the exact genre name (e.g., 'science fiction' not 'sci-fi')
    3. ALWAYS use one of the tools to get movie recommendations. Never suggest movies without using a tool.
    4. Keep explanations brief and focused on why these movies match the user's preferences.
    5. Do not list the movies in your response - they will be displayed automatically.
    6. If the first tool doesn't give good results, try the other tool.
    7. Focus on currently released movies that are available to watch."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Create the agent
tools = [search_movies, discover_movies]
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=2,  # Limit the number of tool calls
    return_intermediate_steps=True,  # We need this to extract movie data
)

# FastAPI routes
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

@app.post("/recommend")
async def recommend(request: PreferenceRequest):
    try:
        # Run the agent
        result = await agent_executor.ainvoke({
            "input": f"Find and recommend movies matching these preferences: {request.preferences}",
            "chat_history": [],
        })
        
        # Extract movie recommendations from the agent's response
        movie_list = []
        if isinstance(result, dict):
            # Try to find movie results in the tool outputs
            for action in result.get("intermediate_steps", []):
                if isinstance(action, tuple) and len(action) == 2:
                    tool_result = action[1]
                    if isinstance(tool_result, dict) and "results" in tool_result:
                        for movie in tool_result["results"]:
                            # Only include movies that have both a title and a poster
                            if movie.get("title") and movie.get("poster_path"):
                                movie_list.append(MovieRecommendation(
                                    id=movie["id"],
                                    title=movie["title"],
                                    description=movie.get("overview", ""),
                                    poster_path=movie.get("poster_path"),
                                    backdrop_path=movie.get("backdrop_path"),
                                    rating=movie.get("vote_average", 0.0)
                                ))
        
        # If no movies were found through the agent, try a direct search
        if not movie_list:
            try:
                search_result = await discover_movies(genre=request.preferences)
                if isinstance(search_result, dict) and "results" in search_result:
                    for movie in search_result["results"]:
                        if movie.get("title") and movie.get("poster_path"):
                            movie_list.append(MovieRecommendation(
                                id=movie["id"],
                                title=movie["title"],
                                description=movie.get("overview", ""),
                                poster_path=movie.get("poster_path"),
                                backdrop_path=movie.get("backdrop_path"),
                                rating=movie.get("vote_average", 0.0)
                            ))
            except Exception as e:
                print(f"Fallback search failed: {str(e)}")
        
        # Get explanation text, clean it up
        explanation = result.get("output", "Here are some movies that match your preferences:")
        # Remove any numbered lists from the explanation
        explanation = explanation.split("1.")[0].strip()
        if not explanation:
            explanation = "Here are some movies that match your preferences:"
        
        # Create the response with the formatted recommendations
        recommendations = MovieRecommendations(
            recommendations=movie_list[:5],  # Limit to 5 recommendations
            explanation=explanation,
            image_base_url=TMDB_IMAGE_BASE_URL,
            tmdb_base_url=TMDB_BASE_URL,
        )
        
        return recommendations
    except Exception as e:
        print(f"Error in recommendation endpoint: {str(e)}")
        # Return a default response with an error message
        return MovieRecommendations(
            recommendations=[],
            explanation="Sorry, I encountered an error while finding movie recommendations. Please try again.",
            image_base_url=TMDB_IMAGE_BASE_URL,
            tmdb_base_url=TMDB_BASE_URL,
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
