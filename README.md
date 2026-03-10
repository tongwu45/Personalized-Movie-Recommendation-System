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
user_id | movie_id | rating
