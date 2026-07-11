from svd_from_scratch import SVD 
import matplotlib.pyplot as plt
import re
import numpy as np

# 1. Define a raw text corpus with two blindingly obvious topic segments
corpus = [
    # Segment 1: Space & Technology

    "The rocket launched into deep space for planet exploration.",
    "NASA astronauts successfully landed on the moon surface.",
    "A new communications satellite is orbiting the earth.",
    "SpaceX engineers built a massive booster rocket engine.",
    
    # Segment 2: Culinary & Cooking
    "Bake the delicious chocolate cake in a preheated oven.",
    "The chef prepared a fresh pasta recipe with olive oil.",
    "Delicious homemade vegetable soup simmering in a pot.",
    "A pastry chef frosted the vanilla wedding cake."
]

# Labels for evaluation/plotting later
true_labels = ["Space", "Space", "Space", "Space", "Food", "Food", "Food", "Food"]

# 2. Text Preprocessing & Tokenization from Scratch
def preprocess_and_tokenize(text):
    # Lowercase and strip out basic punctuation marks
    clean_text = re.sub(r'[^\w\s]', '', text.lower())
    return clean_text.split()

# Simple stop-words list to filter out noisy, meaningless words
STOP_WORDS = {"the", "a", "and", "in", "on", "for", "with", "is", "into", "to"}

# 3. Build Vocabulary
vocabulary = []
tokenized_docs = []

for doc in corpus:
    tokens = [word for word in preprocess_and_tokenize(doc) if word not in STOP_WORDS]
    tokenized_docs.append(tokens)
    for token in tokens:
        if token not in vocabulary:
            vocabulary.append(token)

vocabulary = sorted(vocabulary)
num_docs = len(corpus)
num_words = len(vocabulary)

print(f"Total Documents: {num_docs}")
print(f"Unique words in vocabulary (Features): {num_words}")

# 4. Construct the Document-Term Matrix (A)
# Rows = Documents, Columns = Words
A = np.zeros((num_docs, num_words))

for doc_idx, tokens in enumerate(tokenized_docs):
    for token in tokens:
        word_idx = vocabulary.index(token)
        A[doc_idx, word_idx] += 1  # Raw term frequency count

print(f"\nDocument-Term Matrix shape: {A.shape}")

# 5. Run your Custom SVD for Topic Segmentation (k=2 Latent Topics)
k_topics = 2
svd_nlp = SVD(n_components=k_topics)
svd_nlp.fit(A)

# Project our documents into the 2D Latent Topic Space
# This uses your class method: return A @ self.VT.T
documents_encoded = svd_nlp.transform(A)

# 6. Analyze and Extract the Meaning of the Learned Segments
print("\n--- Uncovering Latent Topics via Right Singular Vectors (V^T) ---")
# svd_nlp.VT shape is (k_topics, num_words)
for topic_idx in range(k_topics):
    word_weights = svd_nlp.VT[topic_idx, :]
    # Get indices of the top 4 heaviest weighted words for this latent concept
    top_word_indices = np.argsort(word_weights)[::-1][:4]
    top_words = [vocabulary[i] for i in top_word_indices]
    print(f"Latent Concept #{topic_idx + 1} Keywords: {', '.join(top_words)}")

# 7. Visualize the Matrix Segmentation Result
plt.figure(figsize=(9, 6))

# Plot space docs vs food docs based on their true labels
for i in range(num_docs):
    marker = 'o' if true_labels[i] == "Space" else 's'
    color = 'blue' if true_labels[i] == "Space" else 'orange'
    
    plt.scatter(documents_encoded[i, 0], documents_encoded[i, 1], 
                s=200, marker=marker, color=color, alpha=0.7)
    # Label each point with its document index number
    plt.text(documents_encoded[i, 0] + 0.02, documents_encoded[i, 1] + 0.02, 
             f"Doc {i}", fontsize=11, fontweight='bold')

plt.title("NLP Document Segmentation via Scratch SVD", fontsize=14, fontweight='bold')
plt.xlabel("Latent Concept 1 Weight (e.g., Space Concept Strength)")
plt.ylabel("Latent Concept 2 Weight (e.g., Cooking Concept Strength)")
plt.grid(True, linestyle='--', alpha=0.5)

# Build a clean legend map
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=12, label='Space & Tech Docs'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='orange', markersize=12, label='Culinary & Food Docs')
]
plt.legend(handles=legend_elements, loc='best')
plt.show()