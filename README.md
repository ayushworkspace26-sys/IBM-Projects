# IBM-Projects

# Data Science & AI Projects Portfolio

A collection of **Data Science, Machine Learning, AI, Data Analysis, Web Scraping, and Simulation projects** developed as part of my academic and practical learning journey.

The repository demonstrates practical skills in **Python, Data Analysis, Machine Learning, Natural Language Processing, AI/LLMs, Web Scraping, Data Visualization, Recommendation Systems, and Algorithmic Simulation**.

---

## 👨‍💻 Author

Name: Ayush Uniyal
Course: B.Tech – Electronics & Communication Engineering (ECE)

---

## 📂 Projects Included

| # | Project                                     | Domain                                    | Primary Technologies                                           |
| - | ------------------------------------------- | ----------------------------------------- | -------------------------------------------------------------- |
| 1 | 🎬 Movie Recommendation System              | Machine Learning / Recommendation Systems | Python, Pandas, NumPy, Scikit-learn                            |
| 2 | 🚕 Uber Ride Allocation Case Study          | Simulation / Algorithms                   | Python, NumPy, Pandas, Matplotlib, Folium                      |
| 3 | 📰 CJP / NEET Protest News Scraping         | Web Scraping / Data Engineering           | Python, Requests, RSS, APIs, Pandas, Google Sheets             |
| 4 | 🤖 AI Resume Screening & Job Recommendation | Generative AI / NLP                       | Python, Gemini API, PyPDF2, python-docx, Pandas                |
| 5 | 👟 Adidas Sales Analysis                    | Data Analysis / Business Intelligence     | Python, Pandas, NumPy, Matplotlib, Seaborn, SciPy, Statsmodels |

---

# 1. 🎬 Movie Recommendation System

### Overview

A recommendation system that suggests movies to users using multiple recommendation techniques.

The project combines:

* **Content-Based Filtering**
* **User-Based Collaborative Filtering**
* **Hybrid Recommendation**
* **Cosine Similarity**
* **TF-IDF Vectorization**
* **Recall@10 Evaluation**

The content-based component converts movie genres and descriptions into TF-IDF representations and calculates cosine similarity between movies.

The collaborative filtering component creates a user-item rating matrix and identifies similar users using cosine similarity.

The hybrid recommender combines content-based and collaborative scores using a configurable weighting parameter (`alpha`).

### Key Features

* Movie dataset loading
* Synthetic fallback dataset
* TF-IDF-based movie similarity
* User-item rating matrix
* Similar-user identification
* Content-based recommendations
* Collaborative recommendations
* Hybrid recommendations
* Recall@10 evaluation
* Command-line configuration for movie, user, number of recommendations, and hybrid weighting

### Technologies

```text
Python
Pandas
NumPy
Scikit-learn
TF-IDF
Cosine Similarity
```

### Main File

```text
movie_recommendation_system.py
```

---

# 2. 🚕 Uber Ride Allocation Case Study

### Overview

A Python-based simulation of a ride-hailing system inspired by Uber's driver allocation workflow.

The project models a city grid, drivers, ride requests, driver availability, pricing, and estimated pickup time.

### Key Features

* Simulated city map
* Random driver locations
* Random ride requests
* Driver availability management
* Nearest-driver allocation
* Weighted driver assignment
* Distance calculation
* ETA estimation
* Dynamic surge pricing
* Ride assignment simulation

The simulation uses a weighted driver-selection algorithm based primarily on driver distance and availability.

### Pricing Model

The simulated fare considers:

* Base fare
* Distance
* Estimated travel time
* Surge multiplier

### Technologies

```text
Python
NumPy
Random
Pandas
Matplotlib
Folium
```

### Main File

```text
Uber_Case_Study.ipynb
```

---

# 3. 📰 CJP / NEET Protest News Scraping & Tracking

### Overview

A news data collection and structuring project designed to track news related to **NEET, CJP protests, paper leaks, examinations, irregularities, investigations, and related events**.

The project collects news from multiple sources and converts the collected articles into a structured dataset.

### Data Sources

The project supports:

* Google News RSS
* GNews API
* NewsData.io API

### Key Features

* Keyword-based news searching
* RSS news collection
* News API integration
* Duplicate article removal
* Exam detection
* News category classification
* Indian state detection
* Reason extraction
* Exam year extraction
* Publication date processing
* Structured Pandas DataFrame
* Google Sheets integration
* CSV export

### Example Categories

The project identifies categories such as:

```text
Paper Leak
Protest
Court Case
Investigation
Arrest
Malpractice
Other
```

### Output Dataset

The structured dataset contains fields such as:

```text
Name of Exam
Exam Category
Board
State
Conducted In
Conducted By
Link
Category
Reason
PBT/CBT
Exam Year
Published Date
```

### Technologies

```text
Python
Requests
Feedparser
Pandas
Google News RSS
GNews API
NewsData.io API
Google Sheets
gspread
Google Authentication
Regular Expressions
```

### Main File

```text
CJP_Protest_News_Scraping.ipynb
```

---

# 4. 🤖 AI Resume Screening & Job Recommendation System

### Overview

An AI-powered resume analysis system that extracts information from a candidate's resume and uses a Generative AI model to analyze the candidate profile.

The system accepts resume documents and generates structured information about the candidate.

### Supported Resume Formats

```text
PDF
DOCX
```

### Key Features

* Resume text extraction from PDF
* Resume text extraction from DOCX
* AI-powered resume analysis
* Technical skill identification
* Soft skill identification
* Education extraction
* Experience extraction
* Candidate profile summary
* Recommended job role
* Missing skills identification
* ATS score generation

### AI Analysis

The system sends the extracted resume content to Google's Gemini model with a structured analysis prompt.

The generated output includes:

```text
Candidate Name
Technical Skills
Soft Skills
Education
Experience
Profile Summary
Recommended Job Role
Missing Skills
ATS Score
```

### Technologies

```text
Python
Google Gemini API
PyPDF2
python-docx
Pandas
JSON
Generative AI
NLP
```

### Main File

```text
AI_Resume (1).ipynb
```

> **Security Note:** API keys should never be committed to GitHub. Store API credentials in environment variables or a `.env` file and add `.env` to `.gitignore`.

---

# 5. 👟 Adidas Sales Analysis

### Overview

A comprehensive business-oriented data analysis project using Adidas sales data.

The project performs data cleaning, feature engineering, exploratory data analysis, statistical testing, regression analysis, segmentation, and short-term forecasting.

### Data Preparation

The analysis includes:

* Missing-value analysis
* Duplicate detection
* Data type validation
* Product name standardization
* Currency conversion
* Numerical data cleaning
* Zero-unit transaction handling
* Sales integrity checks
* Profit and margin calculations

### Feature Engineering

New analytical features include:

```text
Year
Month
Month Name
Season
Margin %
Year-Month
```

### Exploratory Data Analysis

The project analyzes sales performance across:

* Retailers
* Regions
* Product categories
* Sales methods
* Monthly trends

### Statistical Analysis

The project applies:

* One-way ANOVA
* Welch's t-test
* Correlation analysis
* OLS regression

### Business Insights

The analysis investigates:

* Differences in margins across sales channels
* Regional performance
* Product profitability
* Retailer performance
* Seasonal effects
* Revenue vs. margin relationships
* Retailer-region segmentation

The regression analysis examines margin while controlling for factors such as price, units sold, sales method, region, product, and season.

### Forecasting

A simple linear trend is used to produce a short-horizon sales forecast based on the consistently sampled 2021 data.

The project also documents limitations of this approach, including limited historical coverage and the need for more advanced seasonal forecasting methods for production use.

### Technologies

```text
Python
Pandas
NumPy
Matplotlib
Seaborn
SciPy
Statsmodels
Statistical Analysis
OLS Regression
Data Visualization
```

### Main File

```text
adidas-sales--analysis.ipynb
```

---

# 🛠️ Skills Demonstrated

These projects collectively demonstrate practical experience with:

### Programming

* Python
* Object-Oriented Programming
* Functions
* Data Structures
* File Handling

### Data Science

* Pandas
* NumPy
* Data Cleaning
* Feature Engineering
* Exploratory Data Analysis
* Statistical Analysis

### Machine Learning

* TF-IDF
* Cosine Similarity
* Collaborative Filtering
* Content-Based Recommendation
* Hybrid Recommendation
* Model Evaluation

### Artificial Intelligence

* Generative AI
* Google Gemini API
* Resume Analysis
* NLP-based Information Extraction
* AI-assisted Recommendation

### Data Engineering & Web Scraping

* RSS feeds
* REST APIs
* Requests
* Feedparser
* Data Deduplication
* Google Sheets API
* Automated Data Collection

### Data Visualization

* Matplotlib
* Seaborn
* Exploratory Charts
* Business Dashboards/Visual Analysis

### Statistical Analysis

* ANOVA
* Welch's t-test
* Correlation Analysis
* OLS Regression
* Trend Forecasting

---

# 📁 Repository Structure

```text
Data-Science-Projects/
│
├── README.md
│
├── movie_recommendation_system.py
│
├── Uber_Case_Study.ipynb
│
├── CJP_Protest_News_Scraping.ipynb
│
├── AI_Resume.ipynb
│
└── adidas-sales--analysis.ipynb
```

Depending on the individual project, additional datasets, configuration files, or output files may be required.

---

# 🚀 Getting Started

## 1. Clone the Repository

```bash
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
```

## 2. Install Dependencies

```bash
pip install numpy pandas matplotlib seaborn scipy statsmodels scikit-learn requests feedparser gspread google-auth python-dateutil PyPDF2 python-docx google-genai
```

## 3. Run Jupyter Notebooks

```bash
jupyter notebook
```

Open the required `.ipynb` file and execute the cells sequentially.

## 4. Movie Recommendation System

The Python project can be executed using:

```bash
python movie_recommendation_system.py
```

The program supports configurable options for the movie title, user ID, number of recommendations, and hybrid recommendation weight.

---

# 🔐 API & Credential Security

Some projects use external APIs and Google services.

**Do not upload API keys, passwords, tokens, or private credentials to GitHub.**

Use environment variables instead:

```text
GNEWS_API_KEY
NEWSDATA_API_KEY
GOOGLE_SHEET_URL
GEMINI_API_KEY
```

Add sensitive files to `.gitignore`:

```text
.env
*.json
credentials.json
__pycache__/
.ipynb_checkpoints/
```

---

# 🎯 Purpose of This Repository

This repository showcases my practical application of **Data Science, Artificial Intelligence, Machine Learning, Data Analysis, and Python programming** to solve different real-world problems.

The projects cover different stages of the data workflow:

```text
Data Collection
      ↓
Data Cleaning
      ↓
Data Processing
      ↓
Exploratory Analysis
      ↓
Machine Learning / AI
      ↓
Evaluation
      ↓
Insights & Recommendations
```

---

# 👨‍🎓 Author

**Ayush Uniyal**

**B.Tech – Electronics & Communication Engineering (ECE)**

This repository represents a collection of academic, learning, and practical projects developed to strengthen my skills in **Python, Data Science, Machine Learning, Artificial Intelligence, and Data Analytics**.

---

## ⭐ If You Find This Repository Useful

Feel free to explore the projects, review the notebooks, and provide feedback or suggestions for improvement.

**Thank you for visiting my repository!**
