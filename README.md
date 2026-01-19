# Deep Learning Projects – End-to-End Deployment

A collection of **Deep Learning & Machine Learning projects** covering **Natural Language Processing (NLP)** and **Computer Vision (CNN)**, built using **Python, TensorFlow/Keras, and Scikit-learn**, and **deployed on AWS EC2 using Nginx and Route 53**.

This repository demonstrates the **complete machine learning lifecycle** — from data preprocessing and model training to **production-level cloud deployment**.

---

## Live Deployment

All projects are deployed on a **Linux-based AWS EC2 instance**, served via **Nginx** as a reverse proxy, and connected to a custom domain using **AWS Route 53**.

- **Live Demo:**:- https://imranx.dpdns.org/

---

## 🧠 Project Categories

## 1️⃣ NLP Learning Projects

### Email Classification
- Classifies emails into predefined categories (e.g., Spam / Not Spam)
- Uses text preprocessing, tokenization, and feature extraction
- Implements ML/DL models for classification
- Real-world NLP use case

### Fake News Classification
- Detects whether a news article is **Real or Fake**
- Uses both **title and content** for better accuracy
- Handles class imbalance and overfitting
- Practical and socially impactful application

### Movie Review Sentiment Analysis
- Predicts sentiment (**Positive / Negative**) from movie reviews
- Uses word embeddings and deep learning models
- Demonstrates NLP pipeline: cleaning → tokenization → padding → prediction

---

## 2️⃣ CNN Related Projects

### Dog vs Cat Classifier
- Binary image classification using Convolutional Neural Networks
- Includes image preprocessing and augmentation
- Beginner-friendly CNN project

### Mask Image Classifier
- Detects whether a person is wearing a mask or not
- Trained on real-world image datasets
- Practical computer vision application

### Handwritten Digit Recognition
- Recognizes handwritten digits (0–9)
- Trained on the MNIST dataset
- High accuracy and fast inference

### Real-Time Emotion Detector
- Detects human emotions using webcam input
- Built using **CNN + OpenCV**
- Supports real-time inference
- Demonstrates deep learning + computer vision integration

---

## Tech Stack

### Programming & ML Libraries
- Python
- TensorFlow / Keras
- Scikit-learn
- NumPy, Pandas
- OpenCV

### Deployment & Cloud
- AWS EC2 (Ubuntu)
- Nginx (Reverse Proxy & Web Server)
- AWS Route 53 (DNS & Domain Management)
- Flask (Model Serving)
- Gunicorn
- Linux Server Configuration

---

## Deployment Architecture

```text
User Browser
     ↓
Route 53 (DNS)
     ↓
Nginx (Reverse Proxy)
     ↓
Flask Application (Model API)
     ↓
Deep Learning Models
```

# ⚙️ How to Run Locally

```bash
git clone https://github.com/your-username/Deep-Learning-Projects.git
cd Deep-Learning-Projects
pip install -r requirements.txt
python app.py
```
Acces at:- http://127.0.0.1:5000

# Author

**Gaohar Imran**

Machine Learning & Deep Learning Enthusiast with hands-on experience in **NLP, Computer Vision, and Cloud Deployment**.  
Focused on building **real-world, production-ready AI solutions** rather than only academic models.

## Skills & Interests
- Deep Learning (CNN, NLP)
- TensorFlow / Keras / Scikit-learn
- Python & Data Science
- AWS (EC2, Route 53, Nginx)
- Model Deployment & MLOps Basics

##  Connect With Me
- **LinkedIn:** :- https://www.linkedin.com/in/gaohar-imran-5a4063379/

---

> Passionate about turning machine learning models into real, usable applications.
> 
# Support

If you find this project helpful or inspiring, your support is appreciated!

## How You Can Support
- Star this repository
- Fork the project and build upon it
- Share it with others who are learning ML & Deep Learning
- Report issues or suggest improvements

##  Contributions
Contributions are welcome!  
Feel free to open an issue or submit a pull request for improvements, bug fixes, or new features.

---

> Supporting this project helps motivate continuous improvement and future enhancements.
