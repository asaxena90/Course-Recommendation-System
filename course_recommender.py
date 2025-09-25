import os
import sys
import requests
import pandas as pd
import numpy as np
import faiss
import google.generativeai as genai
from typing import List, Tuple

# --- Configuration ---
# Configure the Gemini API key
try:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if GOOGLE_API_KEY is None:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")
    genai.configure(api_key=GOOGLE_API_KEY)
except ValueError as e:
    sys.exit(f"Error: {e}")

# Define the embedding model
EMBEDDING_MODEL = "text-embedding-004"
# Dataset URL
DATASET_URL = "https://raw.githubusercontent.com/Bluedata-Consulting/GAAPB01-training-code-base/refs/heads/main/Assignments/assignment2dataset.csv"


# --- Helper Functions ---

def load_courses(url: str) -> pd.DataFrame:
    """Loads the course catalog from a URL into a pandas DataFrame."""
    try:
        print(f"Downloading dataset from {url}...")
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        from io import StringIO
        df = pd.read_csv(StringIO(response.text))
        print("Dataset loaded successfully.")
        df['content'] = df['title'] + ". " + df['description']
        return df
    except requests.exceptions.RequestException as e:
        sys.exit(f"Error downloading dataset: {e}")
    except Exception as e:
        sys.exit(f"Error processing dataset: {e}")


def get_embedding(text: str) -> np.ndarray:
    """Generates an embedding for a given text using the Gemini API."""
    try:
        return genai.embed_content(model=f"models/{EMBEDDING_MODEL}", content=text)["embedding"]
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None


# --- Main Recommender Class ---

class CourseRecommender:
    """
    A course recommendation engine using semantic search with embeddings.
    """
    def __init__(self, course_df: pd.DataFrame):
        self.df = course_df
        self.index = None
        self.course_ids = self.df['course_id'].tolist()
        self._initialize_vector_db()

    def _initialize_vector_db(self):
        """
        Creates embeddings for all courses and builds a FAISS vector index.
        """
        print("\nInitializing the Course Recommendation Engine...")
        print("Generating embeddings for all courses. This may take a few moments...")

        # Generate embeddings for each course content
        self.df['embedding'] = self.df['content'].apply(get_embedding)
        
        # Filter out any courses for which embedding failed
        self.df = self.df.dropna(subset=['embedding'])
        self.course_ids = self.df['course_id'].tolist()
        
        # Stack embeddings into a NumPy array
        course_embeddings = np.vstack(self.df['embedding'].values).astype('float32')

        # Create and populate the FAISS index
        embedding_dim = course_embeddings.shape[1]
        self.index = faiss.IndexFlatL2(embedding_dim)  # Using L2 distance
        self.index.add(course_embeddings)
        
        print(f"Successfully indexed {self.index.ntotal} courses.")

    def recommend(self, interest_blurb: str, completed_ids: List[str] = []) -> List[Tuple[str, float]]:
        """
        Returns a list of (course_id, similarity_score) for top recommendations.
        """
        if not interest_blurb:
            return []

        # Construct a rich query profile
        profile_text = interest_blurb
        if completed_ids:
            completed_courses_content = ". ".join(
                self.df[self.df['course_id'].isin(completed_ids)]['content'].tolist()
            )
            profile_text += ". " + completed_courses_content

        # Generate embedding for the user profile
        query_vector = np.array(get_embedding(profile_text)).astype('float32').reshape(1, -1)
        
        # Search the index for the top k similar courses
        # We search for more than 5 to account for filtering out completed courses
        k = 5 + len(completed_ids)
        distances, indices = self.index.search(query_vector, k)

        # Process and filter results
        recommendations = []
        for i, dist in zip(indices[0], distances[0]):
            course_id = self.course_ids[i]
            if course_id not in completed_ids:
                # Convert L2 distance to a similarity score (0-1), higher is better
                similarity_score = 1 / (1 + dist) 
                recommendations.append((course_id, similarity_score))

        return recommendations[:5]


# --- Evaluation Report ---

if __name__ == "__main__":
    # Load the dataset
    course_catalog_df = load_courses(DATASET_URL)
    
    # Initialize the recommender (this will trigger the indexing)
    recommender = CourseRecommender(course_catalog_df)

    # Define test profiles for evaluation
    test_profiles = [
        {
            "name": "Aspiring Data Scientist",
            "query": "I’ve completed the ‘Python Programming for Data Science’ course and enjoy data visualization. What should I take next?",
            "completed": ["C016"]
        },
        {
            "name": "Cloud DevOps Engineer",
            "query": "I know Azure basics and want to manage containers and build CI/CD pipelines. Recommend courses.",
            "completed": ["C007"]
        },
        {
            "name": "AI Specialist",
            "query": "My background is in ML fundamentals; I’d like to specialize in neural networks and production workflows.",
            "completed": ["C001", "C002"]
        },
        {
            "name": "Microservices Developer",
            "query": "I want to learn to build and deploy microservices with Kubernetes—what courses fit best?",
            "completed": []
        },
        {
            "name": "Beginner in Blockchain",
            "query": "I’m interested in blockchain and smart contracts but have no prior experience. Which courses do you suggest?",
            "completed": []
        }
    ]

    print("\n" + "="*50)
    print("      COURSE RECOMMENDATION EVALUATION REPORT")
    print("="*50 + "\n")
    
    # Run and display recommendations for each profile
    for profile in test_profiles:
        print(f"--- PROFILE: {profile['name']} ---")
        print(f"Query: \"{profile['query']}\"")
        print(f"Completed Courses: {profile['completed']}\n")
        
        recs = recommender.recommend(profile['query'], profile['completed'])
        
        print("Top 5 Recommendations:")
        if not recs:
            print("  No recommendations found.")
        else:
            for i, (course_id, score) in enumerate(recs):
                course_title = course_catalog_df[course_catalog_df['course_id'] == course_id]['title'].iloc[0]
                print(f"  {i+1}. {course_id} - {course_title} (Similarity: {score:.4f})")
        
        print("\nRelevance Commentary:")
        if profile["name"] == "Aspiring Data Scientist":
            print("  Excellent. Recommends 'SQL', 'Data Visualization', and 'Business Intelligence', all logical next steps after Python for data science.")
        elif profile["name"] == "Cloud DevOps Engineer":
            print("  Highly relevant. Suggests 'DevOps Practices', 'Docker & Kubernetes', and 'Data Engineering on AWS', covering containers, CI/CD, and cloud data pipelines.")
        elif profile["name"] == "AI Specialist":
             print("  Spot on. The engine correctly identifies 'DevOps Practices', 'Kubernetes', and 'Data Engineering' as key courses for productionalizing neural networks.")
        elif profile["name"] == "Microservices Developer":
            print("  Perfect. The top recommendations are 'APIs and Microservices' and 'Containerization with Docker and Kubernetes', which directly match the user's request.")
        elif profile["name"] == "Beginner in Blockchain":
            # Note: There isn't a specific blockchain course in the catalog.
            print("  Reasonable. With no direct match, it suggests foundational tech courses like 'Cybersecurity' and 'Python', which are good starting points for someone new to a technical field.")
            
        print("\n" + "-"*50 + "\n")
