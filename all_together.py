import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

def apply_canny_edge_detection(img):
    """Apply Canny edge detection"""
    return cv2.Canny(img, 100, 300)

def apply_sobel_edge_detection(img):
    """Apply Sobel edge detection"""
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Define the vertical and horizontal filters
    vertical_filter = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    horizontal_filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    
    # Get the dimensions of the image
    n, m = gray.shape
    # Initialize the edges image
    edges_img = np.zeros_like(gray, dtype=np.float32)
    
    # Loop over all pixels in the image
    for row in range(1, n-1):
        for col in range(1, m-1):
            # Create little local 3x3 box
            local_pixels = gray[row-1:row+2, col-1:col+2]
            # Apply the vertical filter
            vertical_transformed_pixels = vertical_filter * local_pixels
            # Calculate the vertical score
            vertical_score = vertical_transformed_pixels.sum() / 4
            # Apply the horizontal filter
            horizontal_transformed_pixels = horizontal_filter * local_pixels
            # Calculate the horizontal score
            horizontal_score = horizontal_transformed_pixels.sum() / 4
            # Combine the horizontal and vertical scores into a total edge score
            edge_score = np.sqrt(vertical_score**2 + horizontal_score**2)
            # Insert this edge score into the edges image
            edges_img[row, col] = edge_score
    
    # Normalize the result
    edges_img = cv2.normalize(edges_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return edges_img

def apply_prewitt_edge_detection(img):
    """Apply Prewitt edge detection"""
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Define the vertical and horizontal filters
    vertical_filter = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    horizontal_filter = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    
    # Get the dimensions of the image
    n, m = gray.shape
    # Initialize the edges image
    edges_img = np.zeros_like(gray, dtype=np.float32)
    
    # Loop over all pixels in the image
    for row in range(1, n-1):
        for col in range(1, m-1):
            # Create little local 3x3 box
            local_pixels = gray[row-1:row+2, col-1:col+2]
            # Apply the vertical filter
            vertical_transformed_pixels = vertical_filter * local_pixels
            # Calculate the vertical score
            vertical_score = (vertical_transformed_pixels.sum() + 3) / 6
            # Apply the horizontal filter
            horizontal_transformed_pixels = horizontal_filter * local_pixels
            # Calculate the horizontal score
            horizontal_score = (horizontal_transformed_pixels.sum() + 3) / 6
            # Combine the horizontal and vertical scores into a total edge score
            edge_score = np.sqrt(vertical_score**2 + horizontal_score**2)
            # Insert this edge score into the edges image
            edges_img[row, col] = edge_score
    
    # Normalize the result
    edges_img = cv2.normalize(edges_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return edges_img

def apply_roberts_edge_detection(img):
    """Apply Roberts edge detection"""
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Define the vertical and horizontal filters
    vertical_filter = np.array([[1, 0], [0, -1]])
    horizontal_filter = np.array([[0, 1], [-1, 0]])
    
    # Get the dimensions of the image
    n, m = gray.shape
    # Initialize the edges image
    edges_img = np.zeros_like(gray, dtype=np.float32)
    
    # Loop over all pixels in the image
    for row in range(1, n-1):
        for col in range(1, m-1):
            # Create little local 2x2 box
            local_pixels = gray[row-1:row+1, col-1:col+1]
            # Apply the vertical filter
            vertical_transformed_pixels = vertical_filter * local_pixels
            # Calculate the vertical score
            vertical_score = vertical_transformed_pixels.sum()
            # Apply the horizontal filter
            horizontal_transformed_pixels = horizontal_filter * local_pixels
            # Calculate the horizontal score
            horizontal_score = horizontal_transformed_pixels.sum()
            # Combine the horizontal and vertical scores into a total edge score
            edge_score = np.sqrt(vertical_score**2 + horizontal_score**2)
            # Insert this edge score into the edges image
            edges_img[row, col] = edge_score
    
    # Normalize the result
    edges_img = cv2.normalize(edges_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return edges_img

def apply_laplacian_edge_detection(img):
    """Apply Laplacian edge detection"""
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Define the Laplacian filter
    lap_filter = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    
    # Get the dimensions of the image
    n, m = gray.shape
    # Initialize the edges image
    edges_img = np.zeros_like(gray, dtype=np.float32)
    
    # Loop over all pixels in the image
    for row in range(1, n-1):
        for col in range(1, m-1):
            # Create little local 3x3 box
            local_pixels = gray[row-1:row+2, col-1:col+2]
            # Apply the Laplacian filter
            lap_transformed_pixels = lap_filter * local_pixels
            # Calculate the Laplacian score
            lap_score = np.sqrt(lap_transformed_pixels.sum()**2)
            # Insert this score into the edges image
            edges_img[row, col] = lap_score
    
    # Normalize the result
    edges_img = cv2.normalize(edges_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return edges_img

def compare_edge_detection_methods(image_path):
    """Compare all edge detection methods on the same image"""
    
    # Read the image
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image from {image_path}")
            print("Please provide a valid image path or use a sample image.")
            return
    except Exception as e:
        print(f"Error reading image: {e}")
        return
    
    # Apply all edge detection methods
    canny_result = apply_canny_edge_detection(img)
    sobel_result = apply_sobel_edge_detection(img)
    prewitt_result = apply_prewitt_edge_detection(img)
    roberts_result = apply_roberts_edge_detection(img)
    laplacian_result = apply_laplacian_edge_detection(img)
    
    # Convert BGR to RGB for matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create a comprehensive comparison plot
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig)
    
    # Original image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img_rgb)
    ax1.set_title('Original Image', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Canny
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(canny_result, cmap='gray')
    ax2.set_title('Canny Edge Detection', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    # Sobel
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(sobel_result, cmap='gray')
    ax3.set_title('Sobel Edge Detection', fontsize=14, fontweight='bold')
    ax3.axis('off')
    
    # Prewitt
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.imshow(prewitt_result, cmap='gray')
    ax4.set_title('Prewitt Edge Detection', fontsize=14, fontweight='bold')
    ax4.axis('off')
    
    # Roberts
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.imshow(roberts_result, cmap='gray')
    ax5.set_title('Roberts Edge Detection', fontsize=14, fontweight='bold')
    ax5.axis('off')
    
    # Laplacian
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.imshow(laplacian_result, cmap='gray')
    ax6.set_title('Laplacian Edge Detection', fontsize=14, fontweight='bold')
    ax6.axis('off')
    
    plt.tight_layout()
    plt.suptitle('Comparison of Edge Detection Methods', fontsize=16, fontweight='bold', y=0.98)
    plt.show()
    
    # Print comparison summary
    print("\n" + "="*60)
    print("EDGE DETECTION METHODS COMPARISON")
    print("="*60)
    print("1. CANNY EDGE DETECTION:")
    print("   - Uses Gaussian smoothing + gradient calculation + non-maximum suppression + double thresholding")
    print("   - Best for: Clean, well-defined edges with minimal noise")
    print("   - Advantages: Robust, handles noise well, produces thin edges")
    print("   - Disadvantages: More computationally intensive")
    
    print("\n2. SOBEL EDGE DETECTION:")
    print("   - Uses 3x3 convolution kernels for horizontal and vertical gradients")
    print("   - Best for: Detecting edges in noisy images")
    print("   - Advantages: Good noise suppression, directional edge detection")
    print("   - Disadvantages: May miss some edges, produces thicker edges")
    
    print("\n3. PREWITT EDGE DETECTION:")
    print("   - Similar to Sobel but with different kernel weights")
    print("   - Best for: Edge detection with moderate noise")
    print("   - Advantages: Good noise handling, isotropic response")
    print("   - Disadvantages: May produce thicker edges than Canny")
    
    print("\n4. ROBERTS EDGE DETECTION:")
    print("   - Uses 2x2 convolution kernels")
    print("   - Best for: Detecting edges at 45-degree angles")
    print("   - Advantages: Simple, fast, good for diagonal edges")
    print("   - Disadvantages: Very sensitive to noise, may miss some edges")
    
    print("\n5. LAPLACIAN EDGE DETECTION:")
    print("   - Uses second-order derivative operator")
    print("   - Best for: Detecting fine details and zero-crossings")
    print("   - Advantages: Detects edges in all directions, good for fine details")
    print("   - Disadvantages: Very sensitive to noise, may produce double edges")
    print("="*60)

def main():
    """Main function to run the edge detection comparison"""
    print("Image Processing Edge Detection Comparison")
    print("="*50)
    
    # Try to use the original image path from the existing files
    original_path = r'C:\Users\MOSALAS\Pictures\elephant.jpg'
    
    # Ask user for image path or use default
    user_path = input(f"Enter image path (or press Enter to use default: {original_path}): ").strip()
    
    if user_path:
        image_path = user_path
    else:
        image_path = original_path
    
    # Run the comparison
    compare_edge_detection_methods(image_path)
    
    # Also show individual windows using OpenCV (optional)
    show_individual_windows = input("\nDo you want to see individual windows as well? (y/n): ").lower().strip()
    
    if show_individual_windows == 'y':
        try:
            img = cv2.imread(image_path)
            if img is not None:
                # Apply all methods
                canny_result = apply_canny_edge_detection(img)
                sobel_result = apply_sobel_edge_detection(img)
                prewitt_result = apply_prewitt_edge_detection(img)
                roberts_result = apply_roberts_edge_detection(img)
                laplacian_result = apply_laplacian_edge_detection(img)
                
                # Show individual windows
                cv2.imshow('Original Image', img)
                cv2.imshow('Canny Edge Detection', canny_result)
                cv2.imshow('Sobel Edge Detection', sobel_result)
                cv2.imshow('Prewitt Edge Detection', prewitt_result)
                cv2.imshow('Roberts Edge Detection', roberts_result)
                cv2.imshow('Laplacian Edge Detection', laplacian_result)
                
                print("Press any key to close all windows...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        except Exception as e:
            print(f"Error showing individual windows: {e}")

if __name__ == "__main__":
    main()
