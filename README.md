# CMPUT 267 - Evaluating Predictors

## Overview

This assignment explores the concepts of expected vs. estimated loss, bias-variance decomposition, model selection using test sets, and regularization in supervised learning. All tasks were completed in a Google Colab notebook and use synthetic data for visualization and analysis.

## Structure

The notebook is divided into three parts:

### Part 1: Visualizing Different Errors
- Understand the behavior of the **Bayes predictor** `f_Bayes`.
- Implement the **polynomial feature mapping** φ₃.
- Visualize:
  - The true optimal predictor `f*ₚ`.
  - The learned predictor `f̂ₚ` from a finite dataset.
  - The expected predictor `f̄ₚ` averaged over datasets.
- Implement `estimated_loss` to compute empirical loss.
- Plot comparisons among:
  - Irreducible error `L(f_Bayes)`
  - Approximation error `L(f*ₚ)`
  - Estimation error `L(f̂ₚ)`
  - Empirical training loss `ĤL(f̂ₚ)`

### Part 2: Picking the Best Polynomial Predictor
- Implement `split_dataset` to separate training and test data.
- Compare training vs. test loss across models with different complexity (degree `p`).
- Justify model selection using test loss estimates.

### Part 3: Regularization
- Implement `regularized_bgd_learner` for L2-regularized batch gradient descent.
- Visualize how regularization affects:
  - Bias of the expected predictor
  - Variance of predictors across datasets
- Plot training and test loss as a function of regularization strength `λ`.

## Implemented Functions

| Function Name               | Purpose                                                                 |
|----------------------------|-------------------------------------------------------------------------|
| `phi_3(x)`                 | Maps input feature vector to degree-3 polynomial feature space          |
| `estimated_loss(f, X, Y)`  | Computes mean squared loss for predictor `f` over dataset `(X, Y)`      |
| `split_dataset(X, Y)`      | Splits dataset into training and test subsets based on a `train_size`   |
| `regularized_bgd_learner`  | Learns weights using BGD with L2 regularization on polynomial features  |

---

