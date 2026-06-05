# Fabric Classification Using OpenCV2

## Overview

Fabric Classification Using OpenCV2 is a machine learning and computer vision project that automatically identifies fabric types from images based on their texture characteristics.

The system uses image processing techniques and Local Binary Pattern (LBP) feature extraction to analyse fabric textures. These features are then classified using a Support Vector Machine (SVM) model. A simple Streamlit-based web interface allows users to upload fabric images and receive instant predictions.

This project was developed as an Industrial-Oriented Mini Project for the Bachelor of Technology degree in Computer Science and Engineering (AI & ML).

---

## Features

* Fabric image classification using texture analysis
* Local Binary Pattern (LBP) feature extraction
* Support Vector Machine (SVM) classifier
* Streamlit-based user interface
* Real-time fabric prediction
* Confidence score for predictions
* Image preprocessing using OpenCV2

---

## Technologies Used

### Programming Language

* Python

### Libraries

* OpenCV
* NumPy
* Scikit-learn
* Scikit-image
* Streamlit
* Pillow
* Joblib

---

## Project Structure

```
Fabric-Classification/
│
├── app.py
├── model.py
├── requirements.txt
├── README.md
│
└── FabricDataset/
    ├── Cotton/
    ├── Denim/
    ├── Nylon/
    ├── Silk/
    ├── Velvet/
    └── ...
```

---

## Working Pipeline

1. Upload a fabric image through the Streamlit interface.
2. The image is preprocessed using OpenCV.
3. Local Binary Pattern (LBP) features are extracted.
4. Extracted features are passed to the trained SVM model.
5. The model predicts the fabric type.
6. The predicted class and confidence score are displayed.

---

## Image Processing Techniques

The project includes:

* Image resizing
* Grayscale conversion
* Noise reduction
* Texture feature extraction using LBP
* Feature normalisation

---

## Machine Learning Model

### Feature Extraction

* Local Binary Pattern (LBP)

### Classifier

* Support Vector Machine (SVM)

### Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1 Score

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/yourusername/fabric-classification.git
cd fabric-classification
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
streamlit run app.py
```

---

## Dataset

The training dataset is **not included in this repository** due to ownership and distribution restrictions. The project was trained using a fabric image dataset containing multiple fabric categories, such as:

* Cotton
* Denim
* Silk
* Nylon
* Velvet
* Wool
* Oxford Cotton

Images should be organised into separate folders, where each folder name represents the fabric class.

Example:

```
datasets/
├── Cotton/
├── Denim/
├── Nylon/
├── Silk/
├── Velvet/
└── Wool/
```

### Dataset Reference

If you wish to reproduce the project, you may use publicly available fabric texture datasets such as:

**Ten Fabrics Dataset**
[https://www.kaggle.com/datasets](https://www.kaggle.com/datasets/saharshakir/ten-fabrics-dataset-tfd)

or any equivalent textile/fabric texture dataset suitable for image classification research.

Please ensure that you comply with the original dataset license and usage terms before downloading or distributing the dataset.

---

## Example Output

```
Predicted Fabric Type: Denim
Confidence: 100.00%
```

---

## Future Improvements

* Deep learning-based classification using CNNs
* Support for additional fabric categories
* Mobile application deployment
* Real-time camera-based classification
* Improved feature extraction using GLCM and Gabor filters
* Cloud deployment for online access

---

## Applications

* Textile Manufacturing
* Quality Control
* Inventory Management
* Fabric Identification
* Textile Research
* E-commerce Product Classification

---

## Authors

**B. Saaketh**
Computer Science and Engineering (AI & ML)
Mahatma Gandhi Institute of Technology

**Aman Adusumilli**
Computer Science and Engineering (AI & ML)
Mahatma Gandhi Institute of Technology
