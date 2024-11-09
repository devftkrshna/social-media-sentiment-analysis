from flask import Flask, render_template, request
import pytesseract
from PIL import Image
import pickle
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load models and vectorizers
with open('english_model.pkl', 'rb') as f:
    model_english = pickle.load(f)
with open('bow_vectorizer_english.pkl', 'rb') as f:
    bow_vectorizer_english = pickle.load(f)

with open('hindi_model.pkl', 'rb') as f:
    model_hindi = pickle.load(f)
with open('bow_vectorizer_hindi.pkl', 'rb') as f:
    bow_vectorizer_hindi = pickle.load(f)

# Configure Tesseract path if needed


UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    lang = request.form['lang']
    text = None

    # Check if text input is provided
    if 'text' in request.form and request.form['text'].strip():
        text = request.form['text'].strip()
        print(f"Text input detected: {text}")

    # Check if an image file was uploaded and process it
    elif 'image' in request.files and request.files['image'].filename != '':
        image_file = request.files['image']
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(image_file.filename))
        image_file.save(image_path)

        # Perform OCR on the saved image
        try:
            extracted_text = pytesseract.image_to_string(Image.open(image_path))
            text = extracted_text.strip()
            print(f"Extracted text from image: {text}")
        except Exception as e:
            print(f"Error in OCR processing: {e}")
            return render_template('result.html', prediction="Error: Unable to process the image.")

    # Ensure there's text to analyze after both checks
    if not text:
        return render_template('result.html', prediction="Error: No text found to analyze.")

    # Predict sentiment based on the selected language
    try:
        if lang == 'english':
            bow_text = bow_vectorizer_english.transform([text])
            prediction = model_english.predict(bow_text)
        elif lang == 'hindi':
            bow_text = bow_vectorizer_hindi.transform([text])
            prediction = model_hindi.predict(bow_text)
        else:
            return render_template('result.html', prediction="Error: Unsupported Language")

        # Return prediction result
        return render_template('result.html', prediction=prediction[0])

    except Exception as e:
        print(f"Error in prediction: {e}")
        return render_template('result.html', prediction="Error: Prediction failed.")

if __name__ == '__main__':
    app.run(debug=True)
