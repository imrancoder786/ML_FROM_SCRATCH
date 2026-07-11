from svd_from_scratch import SVD

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
import os


def compress_color_image(image_path, k_values):
    """
    Compresses an image using the custom SVD class and compares 
    different ranks (k values) side-by-side.
    """
    # 1. Load the image and normalize pixels to [0, 1] range
    img = imread(image_path)
    if img.max() > 1.0:
        img = img / 255.0
        
    # Check if the image is grayscale or RGB
    is_color = len(img.shape) == 3
    m, n = img.shape[:2]
    
    print(f"Original Image Resolution: {m} x {n} pixels")
    
    # 2. Setup the visualization canvas
    fig, axes = plt.subplots(1, len(k_values) + 1, figsize=(16, 5))
    
    # Plot original
    axes[0].imshow(img, cmap='gray' if not is_color else None)
    axes[0].set_title("Original Image\n100% Data")
    axes[0].axis('off')

    # 3. Compress using your custom SVD class
    for idx, k in enumerate(k_values):
  
        
        if is_color:
            # Split into Red, Green, and Blue channels
            R = img[:, :, 0]
            G = img[:, :, 1]
            B = img[:, :, 2]
            
            # Fit and Reconstruct each channel using your class
            svd_R = SVD(n_components=k).fit(R)
            svd_G = SVD(n_components=k).fit(G)
            svd_B = SVD(n_components=k).fit(B)
            
            R_reconstructed = svd_R.reconstruct()
            G_reconstructed = svd_G.reconstruct()
            B_reconstructed = svd_B.reconstruct()
            
            # Stack channels back into a single color image
            reconstructed_img = np.dstack((R_reconstructed, G_reconstructed, B_reconstructed))
            
        else:
            # Grayscale compression is just one matrix
            svd_gray = SVD(n_components=k).fit(img)
            reconstructed_img = svd_gray.reconstruct()

        # Clip values to ensure they stay in valid pixel ranges [0, 1]
        reconstructed_img = np.clip(reconstructed_img, 0, 1)
        
        # 4. Calculate Data footprint and Compression Ratio
        # Original matrix takes m*n numbers. 
        # SVD takes k*(m) for U, k for S, and k*(n) for VT.
        original_size = m * n
        compressed_size = k * (m + n + 1)
        
        if compressed_size >= original_size:
            savings = "0% (k is too high)"
        else:
            compression_ratio = (1 - (compressed_size / original_size)) * 100
            savings = f"{compression_ratio:.1f}% reduced"
            
        # 5. Plot the result
        axes[idx+1].imshow(reconstructed_img, cmap='gray' if not is_color else None)
        axes[idx+1].set_title(f"k = {k}\n{savings}")
        axes[idx+1].axis('off')
        
    plt.tight_layout()
    plt.show()
    
IMAGE_PATH = "premium_photo-1667030474693-6d0632f97029.avif" 
K_TEST_VALUES = [10, 25,50, 100]
    
# call the image    
compress_color_image(IMAGE_PATH, K_TEST_VALUES)