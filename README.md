# 🎬 Movie Recommendation System (Matrix Factorization)

A machine learning project that builds a **movie recommendation system** using the **MovieLens 100K dataset** from https://grouplens.org/datasets/movielens/100k/.

This project compares several approaches for predicting user movie ratings:

- ⭐ **Movie Average Baseline**
- 📉 **Low-Rank Matrix Approximation (SVD)**
- 🧠 **Regularized Matrix Factorization (Alternating Minimization)**

The models are evaluated using **Mean Squared Error (MSE)** on both training and test datasets.

---

# 📊 Dataset

**MovieLens 100K**

- 🎥 1,682 movies  
- 👤 943 users  
- ⭐ 100,000 ratings  

Each record contains:
```text
user_id | movie_id | rating
```

Ratings range from **1 to 5 stars**.

Dataset source:

https://grouplens.org/datasets/movielens/

---

# 🧠 Methods

## 1️⃣ Movie Average Baseline

The simplest recommendation model.

For each movie, compute the **average rating** in the training set:
```text
R̂ = μ · 1ᵀ
```

Where

- μ = average movie ratings  
- all users receive the same prediction for a movie  

This forms a **rank-1 estimator**.

---

## 2️⃣ Low-Rank Matrix Approximation (SVD)

Construct a sparse rating matrix:
```text
R̃(i,j) =
observed rating if rated
0 otherwise
```

Then compute the **best rank-d approximation** using **truncated Singular Value Decomposition (SVD)**:
```text
R̂_d = U S Vᵀ
```

This learns **low-dimensional embeddings for users and movies**.

Latent dimensions tested:
```text
d ∈ {1, 2, 5, 10, 20, 50}
```

---

## 3️⃣ Matrix Factorization (Alternating Minimization)

Instead of filling missing ratings with zero, we optimize **only over observed ratings**.

We learn latent vectors:
```text
movie vector: uᵢ
user vector: vⱼ
```

Predicted rating:
```text
R̂ᵢⱼ = uᵢᵀ vⱼ
```

Optimization objective:
```text
min Σ (uᵢᵀ vⱼ − Rᵢⱼ)² + λ(||U||² + ||V||²)
```

Solved using **Alternating Least Squares (ALS)**.

---

# 📈 Results

Model performance is evaluated using **Mean Squared Error (MSE)**.

Example comparison:

| Model | Description |
|-----|-----|
| Movie Average | Baseline predictor |
| SVD | Low-rank collaborative filtering |
| Matrix Factorization | Regularized latent factor model |

Example result plot:
<img width="662" height="492" alt="截屏2026-03-10 上午11 02 44" src="https://github.com/user-attachments/assets/9cb20b74-5df6-4692-a85d-1cdc5ae07dc6" />
