# Singular Value Decomposition (SVD) from Scratch

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![NumPy](https://img.shields.io/badge/Library-NumPy-blue)
![Machine Learning](https://img.shields.io/badge/Domain-Machine%20Learning-orange)

An educational, from-scratch implementation of Singular Value Decomposition (SVD). 
To prove the robustness of the custom math engine, the `SVD` class is deployed across three vastly different Machine Learning domains: **Computer Vision**, **Natural Language Processing (NLP)**, and **Recommendation Systems**.

## Implemented from scratch (`svd_from_scratch.py`):
- Eigen decomposition using AᵀA
- Singular value computation
- Construction of U, Σ, Vᵀ
- Rank-k approximation
- Reconstruction pipeline
- Latent space transformation

$$ \Large A = U \times \Sigma \times V^T $$



## The 3 Real-World Applications

### 1. Image Compression (Computer Vision)
**File:** `image_compresion.py`
* **How it works:** Treats a grayscale image as a 2D matrix (or an RGB image as three stacked matrices). By calculating the SVD and keeping only the top $k$ singular values (Truncated SVD), we discard high-frequency noise and retain only the most dominant spatial patterns.
* **Result:** Drastically reduces the data footprint of the image (e.g., `cat_image.avif`) while maintaining human-recognizable visual quality.

### 2. Latent Semantic Analysis (NLP)
**File:** `Latent Semantic_Analysis(NLP).py`
* **How it works:** Takes a raw corpus of text, removes stop words, and builds a Document-Term Matrix from scratch. The SVD engine condenses this massive, sparse vocabulary into $k$ dense "Latent Concepts".
* **Result:** Mathematically segments documents into distinct topics (e.g., Space vs. Cooking) based on hidden word-co-occurrence patterns, without any human labeling or pre-trained neural networks.

### 3. Collaborative Filtering (Recommendation System)
**File:** `recomentation_system.py`
* **How it works:** Recreates the core logic of classic recommendation engines (like the early Netflix algorithm). It takes a sparse User-Item rating matrix, imputes missing values using column means, and compresses the data into a low-dimensional "taste space".
* **Result:** Reconstructs a dense matrix where previously unrated movies are now filled with mathematically predicted ratings, personalized to each user's latent preferences.

---

##  Repository Structure
```text
 svd_from_scratch
 ┣  svd_from_scratch.py              # The core Math/SVD engine class
 ┣  image_compresion.py              # Application 1: CV Compression
 ┣  Latent Semantic_Analysis(NLP).py # Application 2: NLP Topic Modeling
 ┣  recomentation_system.py          # Application 3: Collab. Filtering
 ┣  All_exprements.ipynb             # Jupyter Notebook with all experiments
 ┗  cat_image.avif                   # Sample image for the compression script

```

 How to Run Locally
---------------------

**1\. Clone the repository:**

Bash

```
git clone https://github.com/imrancoder786/svd_from_scratch.git
cd svd_from_scratch

```

**2\. Install dependencies:** This project relies almost entirely on standard Python and `numpy`. `matplotlib` is used strictly for visualizing the results.

Bash

```
pip install numpy matplotlib 

```

**3\. Run the applications:**

Bash

```
# Run the Image Compressor
python image_compresion.py

# Run the NLP Segmentation
python "Latent Semantic_Analysis(NLP).py"

# Run the Recommendation Engine
python recomentation_system.py

```

*(Alternatively, you can step through all the code interactively using the `All_exprements.ipynb` Jupyter Notebook!)*

 Key Learnings
----------------

-   **Linear Algebra is Universal:** Whether the data represents image pixels, word frequencies, or movie ratings, finding hidden patterns always boils down to discovering the eigenvectors of a covariance matrix.

-   **Numerical Stability:** Building math from scratch highlights the real-world challenges of floating-point drift and the necessity of normalization (like epsilon tolerances).

-   **Matrix Deflation:** Extracting multiple features requires altering the matrix state iteratively, a powerful concept used across deep learning.

*Always digging deeper into the math behind the machine_learning.*
