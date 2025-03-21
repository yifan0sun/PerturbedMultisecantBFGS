# Advancing Multi-Secant Quasi-Newton Methods for General Convex Functions

**Mokhwa Lee and Yifan Sun**  
Email: mokhwa.lee@stonybrook.edu, yifan.0.sun@gmail.com

---

## Overview

This repository provides part of the code implementation accompanying our paper:

**Advancing Multi-Secant Quasi-Newton Methods for General Convex Functions**  
*Mokhwa Lee and Yifan Sun*

In this work, we address the challenges of stability and descent in quasi-Newton (QN) methods for general convex, non-quadratic functions. By leveraging multiple secant updates and introducing a novel diagonal perturbation technique, our multisecant QN methods:
- Improve Hessian approximation quality at modest computational overhead.
- Ensure symmetric and positive semidefinite (PSD) Hessian estimates.
- Attain superlinear convergence rates even in ill-conditioned optimization landscapes.
- Extend naturally to limited-memory scenarios for large-scale problems.

A Matlab version of this code is also given in [this repository]([https://github.com/Mokhwalee/AlmostMultisecantBFGS])

---

## Key Features

- **Multisecant Updates:** Incorporate multiple secant conditions to better approximate the Hessian compared to traditional single-secant methods.
- **Diagonal Perturbation:** A computationally efficient PSD correction strategy that adjusts the Hessian estimate for stability.
- **Limited-Memory Extension:** Implementation of a multisecant L-BFGS variant that scales to high-dimensional problems.
- **Secant Rejection Mechanism:** Filters out nearly collinear update directions to enhance numerical conditioning.
- **Safe Linear Algebra Routines:** Custom routines (`safe_solve`, `safe_inverse`) ensure robustness when solving near-singular systems.
- **Superlinear Convergence:** The theoretical framework guarantees superlinear convergence when the perturbation parameter decays appropriately.

---

## Repository Structure

```
.
├── bfgs.py         # Implementation of multisecant QN methods and safe linear algebra routines.
├── data.py         # Synthetic data generator for testing optimization on convex problems.
├── models.py       # Simple neural network definition and training/evaluation routines.
├── demo.ipynb      # Demo notebook comparing Gradient Descent with our LMSBFGSOptim on synthetic data.
├── Multisecant_Quasi_Newton__JOTA_.pdf  # The research paper.
└── README.md       # This file.
```

### File Descriptions

- **bfgs.py:**  
  Contains the implementation of the limited-memory multisecant BFGS optimizer (`LMSBFGSOptim`) along with helper functions:
  - `safe_solve` and `safe_inverse` for robust linear system solving.
  - `getmu` for dynamic estimation of the diagonal perturbation parameter.
  - `rejection` for discarding nearly collinear secant vectors.

- **data.py:**  
  Provides a function `get_data` to generate synthetic datasets with controlled eigenvalue decay and noise—ideal for testing optimization algorithms on ill-conditioned convex problems.

- **models.py:**  
  Implements a simple feedforward neural network (`SimpleNN`) using Softplus activations and includes a custom training loop (`train_model`) that integrates our optimizer updates. An evaluation routine (`evaluate`) is provided to compute misclassification rates.

- **demo.ipynb:**  
  A Jupyter Notebook demonstrating how to train a model using both a standard gradient descent optimizer and our multisecant BFGS method. It also provides comparative plots of loss, gradient norm, and misclassification metrics.

---

## Usage

1. **Install Dependencies:**  
   Make sure you have [PyTorch](https://pytorch.org/) and [tqdm](https://github.com/tqdm/tqdm) installed.
   
2. **Run a Demo Notebook:**  
   Open `demo.ipynb` in Jupyter Notebook/Lab to observe a side-by-side comparison of optimization methods on a synthetic logistic regression task.

3. **Experiment with the Code:**  
   - Modify parameters in `data.py` to generate different problem instances.
   - Adjust hyperparameters and options in `bfgs.py` and `models.py` to explore the effect of multisecant updates, diagonal perturbation, and limited-memory strategies.

---

## Results

Our numerical experiments demonstrate that in ill-conditioned regimes, the multisecant quasi-Newton method:
- Achieves faster convergence than both standard gradient descent and classical single-secant QN methods.
- Exhibits robust behavior when augmented with secant rejection and adaptive diagonal perturbation.
- Scales effectively to larger problem sizes using the limited-memory extension.

For detailed numerical comparisons and theoretical convergence proofs, please refer to the paper.



