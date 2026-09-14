

````markdown
# 📊 Intelligent Customer Segmentation & Behavioral Analytics Platform

An end-to-end customer segmentation platform using **K-Means clustering**, behavioral analytics, FastAPI, Streamlit, Docker, and Render.

## 🚀 Features

- Customer behavioral segmentation using K-Means
- RFM-style behavioral features
- Customer segment profiling
- Business recommendations for each segment
- Individual customer segment prediction through FastAPI
- Interactive Streamlit dashboard
- PCA-based cluster visualization
- Dockerized backend and frontend
- Cloud deployment using Render
- Precomputed PCA sample for lightweight frontend deployment

## 🏗️ Architecture

```text
Raw Customer Data
       │
       ▼
Data Cleaning & Feature Engineering
       │
       ▼
Behavioral Features
       │
       ▼
Standardization
       │
       ▼
K-Means Clustering
       │
       ├──────────────► Cluster Profiles
       │
       ├──────────────► Business Profiles
       │
       ├──────────────► Recommendations
       │
       └──────────────► PCA Sample
                              │
                              ▼
                         Streamlit
                              │
                              ▼
                           User
                              │
                              ▼
                           FastAPI
                              │
                              ▼
                      Customer Prediction
````

## 📌 Customer Features

The clustering model uses:

* Recency
* Frequency
* Monetary Value
* Average Order Value
* Engagement Score
* Discount Dependency
* Return Rate
* Online Purchase Ratio
* In-Store Purchase Ratio
* Average Items per Transaction
* Support Interaction Rate

## 📊 Current Segments

| Cluster | Segment                     | Customers | Percentage |
| ------- | --------------------------- | --------: | ---------: |
| 0       | Mainstream Customers        |   980,233 |     98.02% |
| 1       | Support-Intensive Customers |    19,767 |      1.98% |

### Mainstream Customers

The baseline customer population.

Recommended strategy:

* Targeted engagement
* Retention campaigns
* Personalized recommendations

### Support-Intensive Customers

Customers with comparatively higher support interaction behavior.

Recommended strategy:

* Retention
* Cross-selling
* Premium products
* Personalized offers
* Avoid unnecessary blanket discounts

## 🛠️ Technology Stack

### Machine Learning

* Python
* Pandas
* NumPy
* Scikit-learn
* K-Means
* PCA

### Backend

* FastAPI
* Uvicorn

### Frontend

* Streamlit

### Infrastructure

* Docker
* Docker Compose
* Render

### Version Control

* Git
* GitHub

## 📁 Project Structure

```text
customer-segmentation-platform/
│
├── backend/
│   ├── app.py
│   └── ...
│
├── frontend/
│   └── streamlit_app.py
│
├── outputs/
│   ├── cluster_profile.csv
│   ├── cluster_business_profiles.csv
│   ├── cluster_recommendations.csv
│   └── pca_sample.csv
│
├── notebooks/
│   └── ...
│
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .gitignore
└── README.md
```

## ▶️ Run Locally

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run FastAPI:

```bash
uvicorn backend.app:app --reload --port 8000
```

Run Streamlit:

```bash
streamlit run frontend/streamlit_app.py
```

Open:

```text
http://localhost:8501
```

FastAPI documentation:

```text
http://localhost:8000/docs
```

## 🐳 Docker

Build and start the complete application:

```bash
docker compose up -d --build
```

Check containers:

```bash
docker compose ps
```

Frontend:

```text
http://localhost:8501
```

Backend:

```text
http://localhost:8000
```

## 🔌 API

### Health Check

```http
GET /health
```

Response:

```json
{
  "status": "ok"
}
```

### Customer Prediction

```http
POST /predict
```

Example request:

```json
{
  "Recency": 180,
  "Frequency": 10,
  "Monetary_Value": 5000,
  "Average_Order_Value": 500,
  "Engagement_Score": 50,
  "Discount_Dependency": 0.25,
  "Return_Rate": 0.06,
  "Online_Purchase_Ratio": 2,
  "InStore_Purchase_Ratio": 2,
  "Avg_Items_Per_Transaction": 5.5,
  "Support_Interaction_Rate": 1
}
```

## ☁️ Deployment

The application is containerized and can be deployed as separate frontend and backend services.

### Backend

FastAPI service.

### Frontend

Streamlit service.

The frontend communicates with the backend using:

```text
API_URL
```

Example:

```text
API_URL=https://your-backend-url/predict
```

## 📈 PCA Visualization

The production frontend does not load the large customer-level clustering dataset.

Instead, PCA coordinates are precomputed and stored in:

```text
outputs/pca_sample.csv
```

The file contains:

```text
customer_id
PCA1
PCA2
Cluster
```

This keeps the deployed Streamlit application lightweight while still providing an interactive cluster visualization.

## 🔒 Data & Git Management

Large datasets and generated artifacts are excluded from Git.

Examples:

```text
data/
*.joblib
outputs/customer_clusters.csv
outputs/clustering_features.csv
```

Small analytics files required by the application are retained:

```text
outputs/cluster_profile.csv
outputs/cluster_business_profiles.csv
outputs/cluster_recommendations.csv
outputs/pca_sample.csv
```

## 🎯 Business Objective

The platform transforms raw customer behavioral data into actionable customer segments.

The goal is to help businesses:

* Understand customer behavior
* Identify valuable customer groups
* Detect support-intensive customers
* Design targeted retention strategies
* Personalize offers
* Improve customer engagement
* Support data-driven marketing decisions

## 👨‍💻 Author

Om

Built as an end-to-end machine learning and data engineering portfolio project.

````

Save it.

### Step 2 — Check Git

```powershell
git status
````

Then:

```powershell
git add README.md
git commit -m "Improve project documentation"
git push origin main
```

### Step 3 — Final verification

Run:

```powershell
git status
```

You want:

```text
nothing to commit, working tree clean
```

Then your project has the important pieces in place:

**ML → API → Streamlit → Docker → GitHub → Render**
