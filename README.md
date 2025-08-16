# Image Processing Edge Detection Methods

## 📚 Bachelor Course Project

This project was developed as part of my **Bachelor's degree coursework** in Computer Engineering, focusing on **Digital Image Processing**. The project demonstrates and compares five fundamental edge detection algorithms commonly used in computer vision and image analysis.

## 🎯 Project Overview

Edge detection is a fundamental technique in image processing that identifies boundaries and significant changes in image intensity. This project implements and compares five classical edge detection methods, providing both individual implementations and a comprehensive comparison tool.

## 🔍 Edge Detection Methods Implemented

### 1. **Canny Edge Detection**
- **Algorithm**: Multi-stage algorithm using Gaussian smoothing, gradient calculation, non-maximum suppression, and double thresholding
- **Best for**: Clean, well-defined edges with minimal noise
- **Advantages**: Robust, handles noise well, produces thin edges
- **Disadvantages**: More computationally intensive

### 2. **Sobel Edge Detection**
- **Algorithm**: Uses 3×3 convolution kernels for horizontal and vertical gradients
- **Best for**: Detecting edges in noisy images
- **Advantages**: Good noise suppression, directional edge detection
- **Disadvantages**: May miss some edges, produces thicker edges

### 3. **Prewitt Edge Detection**
- **Algorithm**: Similar to Sobel but with different kernel weights
- **Best for**: Edge detection with moderate noise
- **Advantages**: Good noise handling, isotropic response
- **Disadvantages**: May produce thicker edges than Canny

### 4. **Roberts Edge Detection**
- **Algorithm**: Uses 2×2 convolution kernels
- **Best for**: Detecting edges at 45-degree angles
- **Advantages**: Simple, fast, good for diagonal edges
- **Disadvantages**: Very sensitive to noise, may miss some edges

### 5. **Laplacian Edge Detection**
- **Algorithm**: Uses second-order derivative operator
- **Best for**: Detecting fine details and zero-crossings
- **Advantages**: Detects edges in all directions, good for fine details
- **Disadvantages**: Very sensitive to noise, may produce double edges

## 📁 Project Structure

```
Image-Processing-main/
├── all_together.py          # Main comprehensive comparison script
├── canny.py                 # Individual Canny edge detection
├── sobel.py                 # Individual Sobel edge detection
├── prewitt.py               # Individual Prewitt edge detection
├── roberts.py               # Individual Roberts edge detection
├── lap.py                   # Individual Laplacian edge detection
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── final.pdf               # Project documentation
└── final.pptx              # Project presentation
```

## 🚀 Installation & Setup

### Prerequisites
- Python 3.6 or higher
- pip (Python package installer)

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Image-Processing-main.git
   cd Image-Processing-main
   ```

2. **Install required dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

### Running the Comprehensive Comparison

The main script `all_together.py` provides a complete comparison of all edge detection methods:

```bash
python all_together.py
```

**Features:**
- Interactive image path input
- Side-by-side comparison using matplotlib
- Detailed analysis of each method
- Optional individual OpenCV windows
- Educational explanations and comparisons

### Running Individual Methods

You can also run each edge detection method separately:

```bash
python canny.py      # Canny edge detection
python sobel.py      # Sobel edge detection
python prewitt.py    # Prewitt edge detection
python roberts.py    # Roberts edge detection
python lap.py        # Laplacian edge detection
```

## 📊 Output Examples

The comprehensive script provides:

1. **Visual Comparison**: A 2×3 grid showing:
   - Original image
   - Canny edge detection result
   - Sobel edge detection result
   - Prewitt edge detection result
   - Roberts edge detection result
   - Laplacian edge detection result

2. **Detailed Analysis**: Printed comparison of each method's characteristics, advantages, and disadvantages

3. **Interactive Features**: Option to view individual OpenCV windows for detailed inspection

## 🔧 Technical Implementation Details

### Kernel Definitions

**Sobel Kernels:**
```
Vertical:    Horizontal:
[-1 -2 -1]   [-1  0  1]
[ 0  0  0]   [-2  0  2]
[ 1  2  1]   [-1  0  1]
```

**Prewitt Kernels:**
```
Vertical:    Horizontal:
[-1 -1 -1]   [-1  0  1]
[ 0  0  0]   [-1  0  1]
[ 1  1  1]   [-1  0  1]
```

**Roberts Kernels:**
```
Vertical:    Horizontal:
[ 1  0]      [ 0  1]
[ 0 -1]      [-1  0]
```

**Laplacian Kernel:**
```
[ 0  1  0]
[ 1 -4  1]
[ 0  1  0]
```

## 📋 Dependencies

- **opencv-python** ≥ 4.5.0 - Computer vision library
- **numpy** ≥ 1.19.0 - Numerical computing library
- **matplotlib** ≥ 3.3.0 - Plotting and visualization library

## 🎓 Academic Context

This project was developed as part of my **Bachelor's degree coursework** in **Digital Image Processing**. The project demonstrates:

- **Algorithm Implementation**: From-scratch implementation of classical edge detection algorithms
- **Comparative Analysis**: Systematic comparison of different approaches
- **Practical Application**: Real-world image processing techniques
- **Educational Value**: Understanding fundamental computer vision concepts

### Learning Objectives Achieved

- Understanding of convolution operations and kernel-based filtering
- Implementation of gradient-based edge detection methods
- Comparison of different edge detection approaches
- Practical experience with OpenCV and image processing libraries
- Analysis of algorithm performance and characteristics

## 🔍 Key Features

- **Educational**: All algorithms implemented from scratch for learning purposes
- **Comprehensive**: Complete comparison of five different methods
- **Interactive**: User-friendly interface with multiple visualization options
- **Well-Documented**: Detailed explanations and technical documentation
- **Modular**: Individual scripts for each method plus comprehensive comparison

## 🐛 Known Issues & Improvements

- Fixed `cv2.waitkey(0)` typo (corrected to `cv2.waitKey(0)`) in the comprehensive version
- Added proper error handling and input validation
- Implemented normalization for better visualization
- Enhanced user interface and documentation

## 🤝 Contributing

This is an academic project, but suggestions and improvements are welcome! Feel free to:

- Report bugs or issues
- Suggest improvements to the algorithms
- Add new edge detection methods
- Improve documentation

## 📄 License

This project is part of academic coursework and is provided for educational purposes.

## 👨‍🎓 Author

**Student Name**  
Bachelor's Degree in Computer Science/Engineering  
Digital Image Processing Course Project

---

*This project demonstrates fundamental concepts in digital image processing and computer vision, providing both theoretical understanding and practical implementation of edge detection algorithms.*
