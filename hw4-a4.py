# ============================================================
# Movie Recommendation System via Matrix Factorization
# Dataset: MovieLens 100K
#
# This script implements several approaches for predicting
# user-movie ratings:
#
# (a) Movie-average baseline (rank-1 estimator)
# (b) Low-rank matrix approximation via truncated SVD
# (c) Regularized matrix factorization using alternating minimization
#
# The performance of each method is evaluated using Mean
# Squared Error (MSE) on both training and test datasets.
# ============================================================


# ============================================================
# Imports
# ============================================================
# Standard libraries for data handling, numerical computation,
# matrix factorization, and visualization.

import csv
import numpy as np
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
import torch


# ============================================================
# Load MovieLens Dataset
# ============================================================
# The MovieLens 100K dataset contains 100,000 user ratings
# for 1,682 movies from 943 users.
#
# Each record contains:
#   user_id | movie_id | rating
#
# We convert indices to 0-based indexing and split the data
# into training (80%) and test (20%) sets.

data = []
with open('u.data') as csvfile:
    spamreader = csv.reader(csvfile, delimiter='\t')
    for row in spamreader:
        data.append([int(row[0])-1, int(row[1])-1, int(row[2])])
data = np.array(data)

num_observations = len(data)  # total ratings
num_users = max(data[:,0])+1  # 943 users
num_items = max(data[:,1])+1  # 1682 movies

np.random.seed(1)
num_train = int(0.8*num_observations)
perm = np.random.permutation(data.shape[0])
train = data[perm[0:num_train],:]
test = data[perm[num_train::],:]

print(f"Successfully loaded MovieLens dataset with",
      f"{len(train)} training samples and {len(test)} test samples")



# ============================================================
# Part (a): Movie-Average Baseline
# ============================================================
# Baseline recommendation approach:
#
# For each movie, compute the average rating in the training set.
# Every user receives the same predicted rating for that movie.
#
# This produces a rank-1 prediction matrix:
#
#       R_hat = mu * 1^T
#
# where mu is the vector of average movie ratings.


# Compute movie average ratings
movie_sum = np.zeros(num_items)
movie_count = np.zeros(num_items)

for user_idx, movie_idx, rating in train:
    movie_sum[movie_idx] += rating
    movie_count[movie_idx] += 1

mu = np.zeros(num_items)
nonzero_mask = movie_count > 0
mu[nonzero_mask] = movie_sum[nonzero_mask] / movie_count[nonzero_mask]

# Construct rank-1 estimator
R_hat_a = np.outer(mu, np.ones(num_users))

# Evaluate prediction accuracy on the test set
test_preds = np.array([R_hat_a[movie_idx, user_idx] for user_idx, movie_idx, _ in test])
test_true = np.array([rating for _, _, rating in test])
test_mse_a = np.mean((test_preds - test_true) ** 2)

print("Part (a)")
print("Rank-one estimator: R_hat = mu 1^T")
print("Test MSE:", test_mse_a)


# ============================================================
# Part (b): Low-Rank Approximation via SVD
# ============================================================
# Construct a sparse rating matrix:
#
#       R~(i,j) = observed rating if available
#       R~(i,j) = 0 otherwise
#
# Then compute the best rank-d approximation using truncated
# Singular Value Decomposition (SVD).
#
# This approach learns a low-dimensional representation for
# users and movies.


# Build rating matrix from training data
r_twiddle = np.zeros((num_items, num_users))

for user_idx, movie_idx, rating in train:
    r_twiddle[movie_idx, user_idx] = rating


def construct_estimator(d, r_twiddle):
    """
    Compute the best rank-d approximation of the rating matrix
    using truncated SVD.
    """
    U, s, Vt = svds(r_twiddle, k=d)

    # Sort singular values in descending order
    idx = np.argsort(s)[::-1]
    s = s[idx]
    U = U[:, idx]
    Vt = Vt[idx, :]

    S = np.diag(s)
    R_hat_d = U @ S @ Vt
    return R_hat_d


def get_error_from_matrix(R_hat, dataset):
    """
    Compute mean squared prediction error for a dataset.
    """
    preds = np.array([R_hat[movie_idx, user_idx] for user_idx, movie_idx, _ in dataset])
    true = np.array([rating for _, _, rating in dataset])
    return np.mean((preds - true) ** 2)


def get_error(d, r_twiddle, dataset):
    """
    Build the rank-d estimator and compute prediction MSE.
    """
    R_hat_d = construct_estimator(d, r_twiddle)
    return get_error_from_matrix(R_hat_d, dataset)


# Evaluate performance for different latent dimensions
d_values = [1, 2, 5, 10, 20, 50]
train_errors_b = []
test_errors_b = []

for d in d_values:
    R_hat_d = construct_estimator(d, r_twiddle)
    train_mse = get_error_from_matrix(R_hat_d, train)
    test_mse = get_error_from_matrix(R_hat_d, test)

    train_errors_b.append(train_mse)
    test_errors_b.append(test_mse)

    print(f"d = {d}: train MSE = {train_mse:.4f}, test MSE = {test_mse:.4f}")


# Plot train vs test error
plt.figure(figsize=(7, 5))
plt.plot(d_values, train_errors_b, marker='o', label='Train MSE')
plt.plot(d_values, test_errors_b, marker='s', label='Test MSE')
plt.xlabel("Latent Dimension (d)")
plt.ylabel("Mean Squared Error")
plt.title("Low-Rank SVD Recommendation Model")
plt.legend()
plt.grid(True)
plt.show()


# ============================================================
# Part (c): Regularized Matrix Factorization
# ============================================================
# Instead of treating missing ratings as zero, this method
# learns latent user and movie vectors by minimizing squared
# error only over observed ratings.
#
# Optimization is performed using alternating minimization
# with L2 regularization.


# Create lookup tables for fast access to ratings
movie_to_users = [[] for _ in range(num_items)]
user_to_movies = [[] for _ in range(num_users)]

for user_idx, movie_idx, rating in train:
    movie_to_users[movie_idx].append((user_idx, rating))
    user_to_movies[user_idx].append((movie_idx, rating))

def closed_form_u(V, U, l):
    """
    Update all movie vectors U while holding V fixed.

    U shape: (num_items, d)
    V shape: (num_users, d)
    """
    num_items, d = U.shape
    new_U = np.zeros_like(U)

    I = np.eye(d)

    for i in range(num_items):
        A = l * I
        b = np.zeros(d)

        for user_idx, rating in movie_to_users[i]:
            v_j = V[user_idx]
            A += np.outer(v_j, v_j)
            b += rating * v_j

        new_U[i] = np.linalg.solve(A, b)

    return new_U

def closed_form_v(V, U, l):
    """
    Update all user vectors V while holding U fixed.

    U shape: (num_items, d)
    V shape: (num_users, d)
    """
    num_users, d = V.shape
    new_V = np.zeros_like(V)

    I = np.eye(d)

    for j in range(num_users):
        A = l * I
        b = np.zeros(d)

        for movie_idx, rating in user_to_movies[j]:
            u_i = U[movie_idx]
            A += np.outer(u_i, u_i)
            b += rating * u_i

        new_V[j] = np.linalg.solve(A, b)

    return new_V

def predict_matrix(U, V):
    """
    Return full predicted matrix R_hat = U V^T, shape (num_items, num_users)
    """
    return U @ V.T


def mse_from_factors(U, V, dataset):
    """
    Compute MSE on train/test set from factor matrices U, V.
    """
    preds = np.array([U[movie_idx] @ V[user_idx] for user_idx, movie_idx, _ in dataset])
    true = np.array([rating for _, _, rating in dataset])
    return np.mean((preds - true) ** 2)


def regularized_loss(U, V, l):
    """
    Compute the objective in the homework:
    sum of squared errors on train + lambda * ||U||_F^2 + lambda * ||V||_F^2
    """
    sse = 0.0
    for user_idx, movie_idx, rating in train:
        pred = U[movie_idx] @ V[user_idx]
        sse += (pred - rating) ** 2

    reg = l * np.sum(U ** 2) + l * np.sum(V ** 2)
    return sse + reg

def construct_alternating_estimator(
    d, r_twiddle, l=10.0, delta=1e-1, sigma=0.1, U=None, V=None, max_iter=50
):
    """
    Learn factor matrices U, V using alternating minimization.

    Args:
        d: latent dimension
        l: lambda regularization
        delta: stopping threshold
        sigma: scale for random initialization
        U, V: optional initial values
        max_iter: max number of alternating updates

    Returns:
        U, V, losses
    """
    if U is None:
        U = sigma * np.random.randn(num_items, d)
    if V is None:
        V = sigma * np.random.randn(num_users, d)

    losses = []

    for it in range(max_iter):
        old_U = U.copy()
        old_V = V.copy()

        U = closed_form_u(V, U, l)
        V = closed_form_v(V, U, l)

        loss = regularized_loss(U, V, l)
        losses.append(loss)

        change = max(
            np.max(np.abs(U - old_U)),
            np.max(np.abs(V - old_V))
        )

        print(f"iter {it+1}: loss = {loss:.4f}, max change = {change:.6f}")

        if change <= delta:
            break

    return U, V, losses


# Evaluate train and test error for d = 1, 2, 5, 10, 20, 50
d_values = [1, 2, 5, 10, 20, 50]
train_errors_c = []
test_errors_c = []

for d in d_values:
    print(f"\nRunning alternating minimization for d = {d}")
    U, V, losses = construct_alternating_estimator(
        d=d,
        r_twiddle=r_twiddle,
        l=10.0,
        delta=1e-1,
        sigma=0.1,
        max_iter=30
    )

    train_mse = mse_from_factors(U, V, train)
    test_mse = mse_from_factors(U, V, test)

    train_errors_c.append(train_mse)
    test_errors_c.append(test_mse)

    print(f"d = {d}: train MSE = {train_mse:.4f}, test MSE = {test_mse:.4f}")


# Plot both train and test error as a function of d on the same plot.
plt.figure(figsize=(7, 5))
plt.plot(d_values, train_errors_c, marker='o', label='Train MSE')
plt.plot(d_values, test_errors_c, marker='s', label='Test MSE')
plt.xlabel("d")
plt.ylabel("MSE")
plt.title("Part (c): Alternating Minimization Train/Test MSE vs d")
plt.legend()
plt.grid(True)
plt.show()
