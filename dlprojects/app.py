from flask import Flask , request , render_template , jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PIL import Image
import numpy as np
import pandas as pd
import joblib
import re
import nltk
import tensorflow as tf
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import cv2
from flask_cors import CORS
from tensorflow.keras.applications.resnet50 import preprocess_input

nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('punkt')


models = {
    'emailclassifier':{
        'name':'emailclassifier' ,
        'model':joblib.load(r'model&datasets/Email_classifier2.pkl')
    } ,

    'Fake news detector':{
        'name':'fake news detector' ,
        'model':load_model(r'model&datasets/Fake_news_model2.keras' , compile=False)
    } ,

    'Dog vs Cat':{
        'name':'dogvscatclassifier' ,
        'model':load_model(r"model&datasets/dog-vs-cat-classifier5.h5" , compile=False)
    } ,

    'No mask vs with mask':{
        'name':'nomaskvswithmask' ,
        'model':load_model(r"model&datasets/face-mask-detection.h5" , compile=False)
    } ,

    'Handwritten Digit Recognition':{
        'name':'Handrwitten Digit Recognition' ,
        'model':load_model(r"model&datasets/handwritten_digit_cnn.h5" , compile=False)
    } ,

    'Sentiment Analysis':{
        'name':'Sentiment Analysis' ,
        'model':load_model(r'model&datasets/sentiment_lstm.keras' , compile=False)
    } ,

    'Emotion Detector':{
        'name':'Emotion Detector',
        'model':load_model(r'model&datasets/facial_emotion_3resnet50.keras' , compile=False)
    } ,

}

face_cascade = cv2.CascadeClassifier(r'model&datasets/haarcascade_frontalface_default.xml')

app = Flask(__name__)
CORS(app)

# for dogvscat classifier 
def prepare_image(file):
    """
    Load an uploaded image file, preprocess it for model prediction.
    """
    try:
        # Open image and convert to RGB (3 channels)
        img = Image.open(file).convert('RGB')

        # Resize the image to match model input
        img = img.resize((128, 128), resample=Image.Resampling.BILINEAR)

        # Convert to array
        img_array = img_to_array(img)

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    except Exception as e:
        print("Error processing image:", e)
        return None

def clean_text(text):
    STOPWORDS = set(stopwords.words('english'))
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)      # remove HTML
    text = re.sub(r"[^a-z\s]", "", text)   # remove punctuation
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in STOPWORDS]
    return " ".join(tokens)

# for emailclassifier
def prepare_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text

def preprocess_digit_image(file):
    try: 
        img = Image.open(file)
        img = img.convert("L")          # Grayscale
        img = img.resize((28,28))

        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=-1)  # channel dim
        img_array = np.expand_dims(img_array, axis=0)   # batch dim

        return img_array

    except Exception as e:
        print("Error processing image:", e)
        return None
    
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/emailclassifier.html' , methods=['GET' , 'POST'])
def emailclassfier():
    model = models['emailclassifier']['model']
    prediction=None
    if request.method=="POST":
        try:
            text = request.form.get('email')
            email = prepare_text(text)
            
            if email=="":
                prediction="Please enter field"
            else:
                prediction = model.predict([email])[0]
                probability = model.predict_proba([email])[0]
                label = "SPAM 🚫" if prediction==1 else "HAM ✅"
                confidence = f"{probability[prediction]*100:.2f}%"
                prediction = f"{label} (Score: {confidence})"

        except Exception as e:
            prediction = f"Error: {e}"
        
    return render_template('emailclassifier.html' , prediction=prediction)

@app.route('/Fake news classifier.html' , methods=['GET' , 'POST'])
def fakenewsclassifier():
    model = models['Fake news detector']['model']
    prediction = None
    tokenizer = pickle.load(open(r'model&datasets/tokenizer_fakenews.pkl' , 'rb'))

    if request.method=="POST":
        try:
            title = request.form.get("title", "")
            text = request.form.get("text", "")

            title = clean_text(title)
            text = clean_text(text)

            combined = (title + " " + text).strip()
            seq = tokenizer.texts_to_sequences([combined])
            padded = pad_sequences(seq, maxlen=150, padding="post")

            if combined=="":
                prediction = "Please enter both fields"
            else:
                score = model.predict(padded)[0][0]
                label = "Real news" if score<=0.5 else "Fake News"
                prediction = f"{label} (Score: {score})"

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template('Fakenewsclassifier.html' , prediction=prediction)

@app.route('/dogvscatclassifier.html' , methods=['GET' , "POST"])
def dogvscatclassifier():
    model = models['Dog vs Cat']['model']
    prediction = None
    try:
        if request.method=="POST":
            if "file" not in request.files:
                return "No file uploaded",400

            file = request.files['file']
            if file.filename=="":
                return "No selected file",400
            
            img_array= prepare_image(file)

            score = model.predict(img_array)[0][0]
            label = "It is a cat 🐱" if score<=0.5 else "It is a dog 🐶"
            prediction = f"{label} (Score: {score})"

    except Exception as e:
        prediction = f"Error: {e}"
    return render_template('dogvscatclassifier.html' , prediction=prediction)

@app.route('/maskvswithmaskclassifier.html' , methods=["GET" , "POST"])
def maskvswithmaskclassifier():
    model = models['No mask vs with mask']['model']
    prediction = None
    try:
        if request.method=="POST":
            if "file" not in request.files:
                return "No file uploaded",400

            file = request.files['file']
            if file.filename=="":
                return "No selected file",400
            
            img_array= prepare_image(file)

            score = model.predict(img_array)[0][0]
            label = "Yes there is mask in image 😷" if score<=0.5 else "No there is no mask"
            prediction = f"{label} (Score: {score})"

    except Exception as e:
        prediction = f"Error: {e}"
    return render_template('maskvswithmaskclassifier.html' , prediction=prediction)
        
@app.route('/handwrittendigit.html' , methods=['GET' , "POST"])
def handwrittendigit():
    model = models['Handwritten Digit Recognition']['model']
    digit = None
    confidence = None
    try:
        if request.method=="POST":
            if "file" not in request.files:
                return "No file uploaded",400

            file = request.files['file']
            if file.filename=="":
                return "No selected file",400
            
            img_array= preprocess_digit_image(file)

            pred = model.predict(img_array)
            digit = np.argmax(pred[0])
            confidence = pred[0][digit]

    except Exception as e:
        digit = f"Error: {e}"
    return render_template('handwrittendigit.html' , digit=digit , confidence=confidence)

@app.route('/sentimentanalysis.html' , methods=['GET' , "POST"])
def sentimentanalysis():
    model = models['Sentiment Analysis']['model']
    tokenizer = pickle.load(open(r'model&datasets/tokenizer_sentiment.pkl' , 'rb'))
    prediction = None
    try:
        if request.method=='POST':
            review = request.form.get('review')
            review = clean_text(review)
            seq = tokenizer.texts_to_sequences([review])
            padded = pad_sequences(seq, maxlen=150, padding="post")
            score = model.predict(padded)[0][0]
            label = "Positive" if score>=0.5 else "Negative"
            prediction = f"{label} (Score: {score})"
    except Exception as e:
        prediction=f"Error: {e}"
    return render_template('sentimentanalysis.html' , prediction=prediction)

@app.route('/real_time_emotion.html')
def emotion_page():
    return render_template('real_time_emotion.html')

@app.route('/predict', methods=['POST'])
def predict_emotion():
    model = models['Emotion Detector']['model']

    print("Predict called")

    if 'image' not in request.files:
        print("No image key")
        return jsonify({"error": "No image received"}), 400

    file = request.files['image']
    npimg = np.frombuffer(file.read(), np.uint8)

    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if img is None:
        print("Image decode failed")
        return jsonify({"error": "Decode failed"}), 400

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))

    if len(faces) == 0:
        return jsonify({"emotion": "No Face Detected", "confidence": 0})

    # Pick largest face
    x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
    face_img = img[y:y+h, x:x+w]

    # Resize to 224x224 and convert to RGB
    face_img = cv2.resize(face_img, (224, 224))
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

    # Preprocess and add batch dimension
    x_input = preprocess_input(face_img)
    x_input = np.expand_dims(x_input, axis=0)  # shape: (1, 224, 224, 3)
    print("Preprocessed shape:", x_input.shape)

    # Predict
    preds = model.predict(x_input, verbose=0)[0]
    idx = int(np.argmax(preds))

    print("Prediction:", EMOTIONS[idx], preds[idx])

    return jsonify({
        "emotion": EMOTIONS[idx],
        "confidence": float(preds[idx])
    })

if __name__ == "__main__":
    app.run(debug=True)
