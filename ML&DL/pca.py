import cv2
import numpy as np
import matplotlib.pyplot as plt


def pca_channel(channel, k):
    """
    Perform PCA on a single image channel.
    """
    A = np.float64(channel)

    # Mean Centering
    mean = np.mean(A, axis=0)
    A_centered = A - mean

    # Covariance Matrix
    covariance = np.cov(A_centered, rowvar=False)

    # Eigenvalues & Eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    # Sort in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]

    # Select first k principal components
    W = eigenvectors[:, :k]

    # Projection
    compressed = np.dot(A_centered, W)

    # Reconstruction
    reconstructed = np.dot(compressed, W.T) + mean

    reconstructed = np.clip(reconstructed, 0, 255)

    return reconstructed.astype(np.uint8)


def pca_color_image(image, k):
    """
    Apply PCA to each RGB channel separately.
    """
    R, G, B = cv2.split(image)

    R_new = pca_channel(R, k)
    G_new = pca_channel(G, k)
    B_new = pca_channel(B, k)

    return cv2.merge((R_new, G_new, B_new))


# ============================================
# Read Image
# ============================================

image = cv2.imread("image.jpg")

if image is None:
    raise FileNotFoundError("Image not found!")

# Convert BGR to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Number of Principal Components
components = [128,64,32,16]

compressed_images = []

for k in components:
    img = pca_color_image(image, k)
    compressed_images.append(img)

    # Save
    cv2.imwrite(
        f"PCA_{k}_components.png",
        cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    )

# ============================================
# Plot
# ============================================

fig, axes = plt.subplots(1, 5, figsize=(22, 5))

# Original Image
axes[0].imshow(image)
axes[0].set_title("Original Image", fontsize=13)
axes[0].axis("off")

# PCA Compressed Images
for i, k in enumerate(components):
    axes[i+1].imshow(compressed_images[i])
    axes[i+1].set_title(f"{k} Components", fontsize=13)
    axes[i+1].axis("off")

plt.suptitle(
    "Compressing Colour Image using PCA by Reducing the Feature Dimensions to 128, 64, 32 and 16",
    fontsize=16,
    fontweight="bold"
)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.show()