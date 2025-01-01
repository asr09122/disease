# detection/views.py
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import numpy as np
import pickle
from tensorflow.keras.preprocessing import image

# Load your model
with open('plant_disease_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define class labels (as before)
CLASS_LABELS = {
    0: "Apple___Apple_scab",
    1: "Apple___Black_rot",
    2: "Apple___Cedar_apple_rust",
    3: "Apple___healthy",
    4: "Blueberry___healthy",
    5: "Cherry_(including_sour)___Powdery_mildew",
    6: "Cherry_(including_sour)___healthy",
    7: "Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot",
    8: "Corn_(maize)___Common_rust",
    9: "Corn_(maize)___Northern_Leaf_Blight",
    10: "Corn_(maize)___healthy",
    11: "Grape___Black_rot",
    12: "Grape___Esca_(Black_Measles)",
    13: "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    14: "Grape___healthy",
    15: "Orange___Haunglongbing_(Citrus_greening)",
    16: "Peach___Bacterial_spot",
    17: "Peach___healthy",
    18: "Pepper,_bell___Bacterial_spot",
    19: "Pepper,_bell___healthy",
    20: "Potato___Early_blight",
    21: "Potato___Late_blight",
    22: "Potato___healthy",
    23: "Raspberry___healthy",
    24: "Soybean___healthy",
    25: "Squash___Powdery_mildew",
    26: "Strawberry___Leaf_scorch",
    27: "Strawberry___healthy",
    28: "Tomato___Bacterial_spot",
    29: "Tomato___Early_blight",
    30: "Tomato___Late_blight",
    31: "Tomato___Leaf_Mold",
    32: "Tomato___Septoria_leaf_spot",
    33: "Tomato___Spider_mites Two-spotted_spider_mite",
    34: "Tomato___Target_Spot",
    35: "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    36: "Tomato___Tomato_mosaic_virus",
    37: "Tomato___healthy"
}

def index(request):
    prediction_result = None
    
    # If the request is a POST request, handle the uploaded file
    if request.method == 'POST' and request.FILES['uploaded_file']:
        uploaded_file = request.FILES['uploaded_file']
        fs = FileSystemStorage()
        filename = fs.save(uploaded_file.name, uploaded_file)
        file_path = fs.path(filename)  # Get the full path to the saved file

        # Prepare the image for prediction
        img = image.load_img(file_path, target_size=(128, 128))  # Adjust size as necessary
        img_array = image.img_to_array(img) / 255.0  # Normalize the image if needed
        img_array = np.expand_dims(img_array, axis=0)

        # Make predictions
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)  # If using a classification model
        predicted_label = CLASS_LABELS[predicted_class[0]]  # Use CLASS_LABELS dictionary

        prediction_result = {'filename': uploaded_file.name, 'prediction': predicted_label}
        print(prediction_result)
    return render(request, 'disease_detector/index.html', {'result': prediction_result})