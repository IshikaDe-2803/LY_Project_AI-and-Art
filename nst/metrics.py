
import numpy as np
import cv2 

def calculate_color_variance(image):
  # Convert image to NumPy array if needed
  if not isinstance(image, np.ndarray):
    image = np.array(image)

  # Check if image is grayscale or color
  if len(image.shape) == 2:
    # Grayscale image, no color variance
    return 0.0
  else:
    # Color image, proceed with variance calculation
    # Calculate the variance for each color channel
    channel_variances = np.var(image, axis=0)
    # Combine variances across channels, assuming equal weight for each
    color_variance = np.mean(channel_variances)
    # Normalize variance to the range [0, 1]
    max_variance = np.max(channel_variances)  # Find maximum variance across channels
    if max_variance == 0:  # Handle case where all pixels are the same color
        return 0.0
    normalized_variance = color_variance / max_variance

    return normalized_variance


def calculate_coarseness(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Perform filtering (e.g., averaging) to obtain a coarser version of the image
    filtered = cv2.blur(gray, (5, 5))

    # Calculate the absolute difference between the original and filtered images
    diff = cv2.absdiff(gray, filtered)

    # Normalize the difference image to the range [0, 1]
    max_diff = np.max(diff)
    if max_diff == 0:  # Handle case where all pixels are the same
        return 0.0
    diff_norm = diff / max_diff

    # Compute the mean of the normalized difference image to represent coarseness
    coarseness = np.mean(diff_norm)

    return coarseness


def calculate_edge_density(image):
    # Assuming 'image' is a NumPy array (e.g., output from CartoonGAN)
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(grayscale_image, 50, 150)  # Adjust parameters as needed
    edge_density = np.sum(edges) / np.sum(image)
    return edge_density


def calculate_saturation(image):
    # Assuming 'image' is a NumPy array (e.g., output from CartoonGAN)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = np.mean(hsv_image[:, :, 1])
    return saturation / 255


def calculate_shape_complexity(image):
    # Convert the image to grayscale
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Binarize the grayscale image
    _, binary_image = cv2.threshold(grayscale_image, 128, 255, cv2.THRESH_BINARY)
    # Find contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Calculate total area of the image
    total_area = binary_image.shape[0] * binary_image.shape[1]
    # Calculate shape complexity as the sum of contour areas relative to the total area
    shape_complexity = sum(cv2.contourArea(contour) / total_area for contour in contours)

    return shape_complexity

def evaluate_result(image):
    return calculate_color_variance(image), calculate_saturation(image), calculate_coarseness(image), calculate_edge_density(image),calculate_shape_complexity(image) 