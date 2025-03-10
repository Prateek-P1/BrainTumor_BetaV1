import os
import numpy as np
import nibabel as nib
import cv2
import tensorflow as tf
import keras
import matplotlib
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

# ✅ Prevent Tkinter errors from Matplotlib
matplotlib.use('Agg')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['OUTPUT_FOLDER'] = 'static/outputs/'
app.config['SLICE_FOLDER'] = 'static/slices/'  # New folder for slice images
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
os.makedirs(app.config['SLICE_FOLDER'], exist_ok=True)  # Create slice folder

# ✅ Model Loading (Only Used for front.html)
MODEL_PATH = "model_x1_1.h5"
print(f"Loading model from: {MODEL_PATH}")

try:
    model = keras.models.load_model(MODEL_PATH, custom_objects={
        'accuracy': tf.keras.metrics.MeanIoU(num_classes=4),
        "dice_coef": lambda y_true, y_pred: tf.keras.backend.sum(y_true * y_pred),
        "precision": tf.keras.metrics.Precision(),
        "sensitivity": tf.keras.metrics.Recall(),
        "specificity": tf.keras.metrics.SpecificityAtSensitivity(0.5),
        "dice_coef_necrotic": lambda y_true, y_pred: tf.keras.backend.sum(y_true * y_pred),
        "dice_coef_edema": lambda y_true, y_pred: tf.keras.backend.sum(y_true * y_pred),
        "dice_coef_enhancing": lambda y_true, y_pred: tf.keras.backend.sum(y_true * y_pred)
    }, compile=False)
    print("Model loaded successfully!")
except Exception as e:
    print("Error loading model:", e)
    model = None

IMG_SIZE = 128

# Store the current NIfTI file path
current_nifti_file = None

def preprocess_image(file_path):
    """Preprocesses a NIfTI (.nii) file for model input."""
    try:
        print(f"Preprocessing file: {file_path}")
        flair = nib.load(file_path).get_fdata()
        slices = flair.shape[2]
        X = np.zeros((slices, IMG_SIZE, IMG_SIZE, 1))

        for i in range(slices):
            X[i, :, :, 0] = cv2.resize(flair[:, :, i], (IMG_SIZE, IMG_SIZE))

        X = X / np.max(X)  # Normalize
        X = np.repeat(X, 2, axis=-1)  # Duplicate channel (128,128,2)

        return X, slices
    except Exception as e:
        print(f"Error in preprocess_image: {e}")
        return None, 0

def get_predictions(X):
    """Generates predictions from the model."""
    try:
        print("Generating predictions...")
        predictions = model.predict(X)
        return np.argmax(predictions, axis=-1)  # Get class labels
    except Exception as e:
        print(f"Error in get_predictions: {e}")
        return None

def process_nii(file_path):
    """Processes the uploaded NIfTI file and generates segmentation masks."""
    try:
        X, slices = preprocess_image(file_path)
        if X is None or slices == 0:
            print("Error: Preprocessed data is empty!")
            return None

        predictions = get_predictions(X)
        if predictions is None:
            print("Error: Model failed to generate predictions!")
            return None

        slice_paths = []
        for slice_num in range(slices):
            flair_path = os.path.join(app.config['OUTPUT_FOLDER'], f"flair_{slice_num}.png")
            mask_path = os.path.join(app.config['OUTPUT_FOLDER'], f"mask_{slice_num}.png")
            overlay_path = os.path.join(app.config['OUTPUT_FOLDER'], f"overlay_{slice_num}.png")

            plt.imsave(flair_path, X[slice_num, :, :, 0], cmap='gray')
            plt.imsave(mask_path, predictions[slice_num], cmap='jet')

            fig, ax = plt.subplots()
            ax.imshow(X[slice_num, :, :, 0], cmap='gray')
            ax.imshow(predictions[slice_num], cmap='jet', alpha=0.5)
            plt.axis('off')
            plt.savefig(overlay_path, bbox_inches='tight', pad_inches=0)
            plt.close(fig)

            slice_paths.append({
                "flair": f"/static/outputs/flair_{slice_num}.png",
                "mask": f"/static/outputs/mask_{slice_num}.png",
                "overlay": f"/static/outputs/overlay_{slice_num}.png"
            })

        print("Processing complete! Returning slice paths.")
        return slice_paths
    except Exception as e:
        print(f"Error in process_nii: {e}")
        return None

# ✅ Route for `front.html` (Runs Model for Tumor Detection)
@app.route('/detect_tumor', methods=['POST'])
def detect_tumor():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    print(f"File received for model processing: {file_path}")

    # Process NIfTI file using the model
    slices = process_nii(file_path)

    if slices:
        return jsonify({'slices': slices})
    else:
        return jsonify({'error': 'Failed to process file'}), 500

# ✅ Route for `advanced.html` (No Model, Just Upload)
@app.route('/upload_nii', methods=['POST'])
def upload_nii():
    global current_nifti_file
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    # Save the path for slice viewing
    current_nifti_file = file_path

    print(f"File received for 3D viewing: {file_path}")

    return jsonify({'message': 'File uploaded successfully', 'file_path': f"/uploads/{filename}"})

# ✅ New route to fetch slices for the advanced viewer
@app.route('/get_slice/<axis>/<int:index>', methods=['GET'])
def get_slice(axis, index):
    global current_nifti_file
    
    if not current_nifti_file or not os.path.exists(current_nifti_file):
        return jsonify({'error': 'No NIfTI file loaded'}), 400
    
    try:
        # Load NIfTI data
        nii_img = nib.load(current_nifti_file)
        data = nii_img.get_fdata()
        
        # Determine slice based on axis
        if axis == 'axial':
            if index >= data.shape[2]:
                return jsonify({'error': f'Index {index} out of range'}), 400
            slice_data = data[:, :, index]
        elif axis == 'coronal':
            if index >= data.shape[1]:
                return jsonify({'error': f'Index {index} out of range'}), 400
            slice_data = data[:, index, :]
        elif axis == 'sagittal':
            if index >= data.shape[0]:
                return jsonify({'error': f'Index {index} out of range'}), 400
            slice_data = data[index, :, :]
        else:
            return jsonify({'error': 'Invalid axis'}), 400
        
        # Normalize for visualization
        min_val = np.min(slice_data)
        max_val = np.max(slice_data)
        if max_val > min_val:
            slice_data = (slice_data - min_val) / (max_val - min_val)
        
        # Generate filename
        file_name = f"{axis}_{index}.png"
        file_path = os.path.join(app.config['SLICE_FOLDER'], file_name)
        
        # Save the slice as an image
        plt.figure(figsize=(4, 4))
        plt.imshow(slice_data, cmap='gray')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        return jsonify({'slice_path': f"/static/slices/{file_name}"})
    
    except Exception as e:
        print(f"Error generating slice: {e}")
        return jsonify({'error': f'Failed to generate slice: {str(e)}'}), 500

# ✅ Routes for serving pages
@app.route('/')
def home():
    return render_template('front.html')

@app.route('/advanced')
def advanced():
    return render_template('advanced.html')

# ✅ Routes for serving static files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/static/outputs/<filename>')
def output_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

# ✅ New route for serving slice images
@app.route('/static/slices/<filename>')
def slice_file(filename):
    return send_from_directory(app.config['SLICE_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)