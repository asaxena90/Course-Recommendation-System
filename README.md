# Assignment 2: Personalized Course Recommendation Engine

This project implements a course recommendation engine that suggests relevant courses to a user based on their interests and completed coursework. It leverages the power of semantic search using Google's Gemini for generating embeddings and FAISS for efficient vector similarity search.

##  Setup and Configuration

### 1. Clone the Repository
Clone this project to your local machine.

### 2. Create a Virtual Environment
It's recommended to use a virtual environment to manage dependencies.

```bash
python3 -m venv venv
source venv/bin/activate  
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

### 4. Configure Your API Key

```bash
export GOOGLE_API_KEY="AIzaSy...your...key...here"
```

## How to Run

With your virtual environment activated and the API key set, run the Python script:

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



