# Road Lane Detection using Image Analysis

This project is a Road Lane Detection web application built using classical image processing techniques. The system takes a road image as input and detects lane markings using grayscale conversion, Gaussian blur, Canny edge detection, color masking, region of interest selection, and Hough Line Transform.

## Live Demo

🚀 **Hugging Face Space:** https://apurv20-road-lane-detection.hf.space

## Project Overview

The goal of this project is to detect lane boundaries from road images using an image analysis pipeline. Instead of using a deep learning model, this project focuses on traditional computer vision methods to understand how lane features can be extracted step by step from an image.

## Features

- 📸 Upload a road image
- ⚪ Convert image to grayscale
- 🌫️ Apply Gaussian blur for noise reduction
- 🔍 Detect edges using Canny edge detection
- 🎨 Apply color masking for lane-like regions
- 🎯 Use region of interest masking
- 📐 Detect lane lines using Hough Transform
- ✨ Display the final lane overlay result

## Tech Stack

- **Language:** Python
- **Image Processing:** OpenCV
- **Numerical Computing:** NumPy
- **Image Handling:** Pillow
- **Web UI:** Gradio
- **Deployment:** Hugging Face Spaces

## Project Structure

```
Road_Lane_Detection/
├── app.py                    # Gradio application for deployment
├── requirements.txt          # Python dependencies
├── Road_lane.ipynb          # Jupyter Notebook version of the project
└── README.md                # Project documentation
```

## How to Run Locally

### Prerequisites
- Python 3.7 or higher
- pip (Python package manager)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/apurvchokshi/Road_Lane_Detection.git
cd Road_Lane_Detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

4. Open your browser and navigate to the URL provided by Gradio (usually `http://localhost:7860`)

## Usage

1. Upload a road image (JPG, PNG, etc.)
2. The application will process the image through the following pipeline:
   - Grayscale conversion
   - Gaussian blur filtering
   - Canny edge detection
   - Color-based lane masking
   - Region of interest selection
   - Hough Line Transform for lane detection
3. View the output with detected lane lines overlaid on the original image

## How It Works

The lane detection pipeline uses classical computer vision techniques:

1. **Grayscale Conversion** - Reduces color information to intensity values
2. **Gaussian Blur** - Smooths the image to reduce noise
3. **Canny Edge Detection** - Identifies sharp intensity gradients (edges)
4. **Color Masking** - Filters for white/yellow lane colors
5. **Region of Interest** - Focuses on the relevant road area
6. **Hough Line Transform** - Detects line segments from edges

## Dependencies

See `requirements.txt` for all required packages:
- opencv-python
- numpy
- pillow
- gradio

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests for improvements.

## License

This project is open source and available under the MIT License.

## Contact

For questions or feedback, feel free to reach out via GitHub issues.

---

**Note:** This project is designed for educational purposes to understand classical computer vision techniques for lane detection.
