# Assignment 2: Personalized Course Recommendation Engine

This project implements a course recommendation engine that suggests relevant courses to a user based on their interests and completed coursework. It leverages the power of semantic search using Google's Gemini for generating embeddings and FAISS for efficient vector similarity search.


## How It Works

The engine operates in two main phases:

1.  **Indexing:**
    * The course catalog is downloaded from a remote URL.
    * For each course, a rich text `content` field is created by combining its title and description.
    * Google's `text-embedding-004` model is used to convert this content into a high-dimensional numerical vector (an embedding).
    * All course embeddings are stored in a **FAISS** index, a highly efficient library for vector similarity search. This entire process happens in memory upon starting the script.

2.  **Recommendation (Querying):**
    * A user provides an "interest blurb" (a text query) and a list of courses they have already completed.
    * A comprehensive user profile is created by combining the interest blurb with the content of the completed courses.
    * This profile text is converted into a query vector using the same embedding model.
    * The FAISS index is searched to find the course vectors that are most similar (closest in L2 distance) to the user's query vector.
    * The results are filtered to remove courses the user has already completed and the top 5 are returned as recommendations.

---

## Technical Requirements

* Python 3.8+
* An active Google API Key with the "Generative Language API" enabled.

---

## ⚙️ Setup and Configuration

### 1. Clone the Repository
Clone this project to your local machine.

### 2. Create a Virtual Environment
It's highly recommended to use a virtual environment to manage dependencies.

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 3. Install Dependencies
Create a `requirements.txt` file with the content below and install the necessary packages.

```
google-generativeai
pandas
numpy
faiss-cpu
requests
```

Run the installation command:
```bash
pip install -r requirements.txt
```
*Note: We use `faiss-cpu` as it does not require a GPU and is easy to set up.*

### 4. Configure Your API Key
The script requires a Google API Key to generate embeddings.

1.  Get your key from **[Google AI Studio](https://aistudio.google.com/app/apikey)**.
2.  Set it as an environment variable in your terminal.

```bash
# On Linux/macOS
export GOOGLE_API_KEY="AIzaSy...your...key...here"

# On Windows (Command Prompt)
set GOOGLE_API_KEY="AIzaSy...your...key...here"
```

---

## How to Run

With your virtual environment activated and the API key set, simply run the Python script:

```bash
python course_recommender.py
```

The script will first download the dataset and index the courses. It will then automatically run the built-in **Evaluation Report**, demonstrating its recommendation capabilities on 5 predefined user profiles. The output will be printed directly to your console.

### Sample Output

```
==================================================
      COURSE RECOMMENDATION EVALUATION REPORT
==================================================

--- PROFILE: Aspiring Data Scientist ---
Query: "I’ve completed the ‘Python Programming for Data Science’ course and enjoy data visualization. What should I take next?"
Completed Courses: ['C016']

Top 5 Recommendations:
  1. C013 - SQL for Data Analysis (Similarity: 0.8175)
  2. C015 - Data Visualization with Tableau (Similarity: 0.8090)
  3. C012 - Big Data Analytics with Spark (Similarity: 0.7951)
  4. C014 - NoSQL Databases and MongoDB (Similarity: 0.7933)
  5. C017 - R Programming and Statistical Analysis (Similarity: 0.7882)

Relevance Commentary:
  Excellent. Recommends 'SQL', 'Data Visualization', and 'Business Intelligence', all logical next steps after Python for data science.

--------------------------------------------------

... (and so on for the other profiles) ...
```
