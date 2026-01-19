# eXCNN: Explainable CNN Visualization Platform

eXCNN is a comprehensive interactive platform designed to visualize and decode the decision-making process of Convolutional Neural Networks (CNNs). By leveraging state-of-the-art Explainable AI (XAI) methods, it allows users to see exactly "where" and "what" a deep learning model looks at when classifying images.

<img src="https://github.com/halim-cv/eXCNN-Explainable-CNN/blob/main/assets/screenshots/2026-01-18%20(3).png">

## Key Features

*   **Interactive Analysis**: Upload any image (JPG/PNG) to receive real-time classification and visual explanations.
*   **Multi-Method Visualization**:
    *   **Grad-CAM**: Visualizes class-discriminative regions using gradient information.
    *   **Occlusion Sensitivity**: Measures feature importance by systematically hiding image parts.
    *   **Guided Backpropagation**: Highlights fine-grained pixel-level details.
    *   **Guided Grad-CAM**: Combines localization with high-resolution details for the sharpest insight.
*   **Precomputed Gallery**: Explore curated examples with detailed breakdowns of model behavior.
*   **Robust Backend**: Powered by PyTorch models (ResNet50) and FastAPI for high-performance inference.
*   **Modern Frontend**: A clean, monochrome interface focusing on usability and data visualization.


## Interface Gallery

## Interface Gallery

| Analysis View | Feature Visualization |
| :---: | :---: |
| <img src="https://github.com/halim-cv/eXCNN-Explainable-CNN/blob/main/assets/screenshots/2026-01-18%20(4).png"> | <img src="https://github.com/halim-cv/eXCNN-Explainable-CNN/blob/main/assets/screenshots/2026-01-18%20(5).png"> |
| **Model Insights**
| <img src="https://github.com/halim-cv/eXCNN-Explainable-CNN/blob/main/assets/screenshots/2026-01-18%20(6).png"> |

---

## Technology Stack

*   **Backend**: Python 3.10+, FastAPI, PyTorch, Torchvision, NumPy, OpenCV, Uvicorn.
*   **Frontend**: Native HTML5, CSS3 (Modern/Flexbox/Grid), Vanilla JavaScript (ES6+).
*   **Model**: ResNet50 (Pre-trained on ImageNet-1k).

## Installation & Setup

### Prerequisites
*   Python 3.8 or higher
*   pip (Python package manager)

### 1. Clone & Setup
Clone the repository and install the dependencies:

```bash
# Install required Python packages
pip install -r requirements.txt
```

### 2. Run the Application
Start the entire platform with a single command:

```bash
python run_app.py
```

This will automatically launch:
*   **Backend API** at `http://localhost:8000`
*   **Frontend Interface** at `http://localhost:8080`

Wait for the "[OK] Application is fully running!" message, then open your browser.

### 3. Access
Open your browser and navigate to: **[http://localhost:8080](http://localhost:8080)**

---

## Project Structure

```
eXCNN/
├── assets/                 # Static assets and database files
│   ├── imagenet_labels.json # Full 1000 ImageNet class labels
│   ├── screenshots/        # Application screenshots
│   └── precomputed/        # Gallery examples with generated visualizations
├── backend/                # FastAPI application
│   ├── api.py              # Main API endpoints
│   ├── explainers.py       # Implementation of XAI methods (GradCAM, etc.)
│   └── model_loader.py     # ResNet50 model management
├── frontend/               # User Interface
│   ├── index.html          # Main application structure
│   ├── styles.css          # Monochrome design system
│   └── app.js              # Application logic and API integration
├── scripts/                # Utility scripts
│   ├── generate_precomputed.py # Script to generate gallery assets
│   └── test_backend.py     # Backend test suite
├── requirements.txt        # Python dependencies
└── README.md               # This documentation
```

---

## Understand the XAI Methods

| Method | Description | Best For |
| :--- | :--- | :--- |
| **Grad-CAM** | Uses the gradients of the target class flowing into the final convolutional layer to produce a coarse localization map highlighting important regions. | identifying *where* the object is. |
| **Occlusion** | Slides a grey patch over the image and records the drop in probability. If confidence drops drastically, that area is critical. | Verifying robustness without gradients. |
| **Guided Backprop** | visualizes gradients with respect to the image, but zeroing out negative gradients during backpropagation to show only positive influences. | Seeing *textures* and *patterns*. |
| **Guided Grad-CAM** | Element-wise multiplication of Guided Backprop and Grad-CAM (bilinear upsampled). | The most interpretable and sharp visualization. |

---

## License
This project is built for educational purposes. Code is open for modification and study.





