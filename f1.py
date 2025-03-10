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

#  Prevent Tkinter errors from Matplotlib
matplotlib.use('Agg')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['OUTPUT_FOLDER'] = 'static/outputs/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

#  Model Loading (Only Used for front.html)
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

#  New function to generate and save axial, coronal, and sagittal views
def generate_orthogonal_slices(file_path):
    """Generates axial, coronal, and sagittal slices from a NIfTI file."""
    try:
        print(f"Generating orthogonal slices for: {file_path}")
        nii_img = nib.load(file_path)
        img_data = nii_img.get_fdata()
        
        # Get dimensions
        x_dim, y_dim, z_dim = img_data.shape
        
        # Create directories for different slice types
        axial_dir = os.path.join(app.config['OUTPUT_FOLDER'], 'axial')
        coronal_dir = os.path.join(app.config['OUTPUT_FOLDER'], 'coronal')
        sagittal_dir = os.path.join(app.config['OUTPUT_FOLDER'], 'sagittal')
        
        os.makedirs(axial_dir, exist_ok=True)
        os.makedirs(coronal_dir, exist_ok=True)
        os.makedirs(sagittal_dir, exist_ok=True)
        
        # Normalize the data for better visualization
        img_data = img_data / np.max(img_data) if np.max(img_data) > 0 else img_data
        
        # Generate axial slices (top to bottom view)
        for i in range(z_dim):
            axial_slice = img_data[:, :, i]
            slice_path = os.path.join(axial_dir, f"slice_{i}.png")
            plt.imsave(slice_path, axial_slice.T, cmap='gray')
        
        # Generate coronal slices (front to back view)
        for i in range(y_dim):
            coronal_slice = img_data[:, i, :]
            slice_path = os.path.join(coronal_dir, f"slice_{i}.png")
            plt.imsave(slice_path, coronal_slice.T, cmap='gray')
        
        # Generate sagittal slices (side view)
        for i in range(x_dim):
            sagittal_slice = img_data[i, :, :]
            slice_path = os.path.join(sagittal_dir, f"slice_{i}.png")
            plt.imsave(slice_path, sagittal_slice.T, cmap='gray')
        
        return {
            'axial': {'count': z_dim, 'min': 0, 'max': z_dim-1},
            'coronal': {'count': y_dim, 'min': 0, 'max': y_dim-1},
            'sagittal': {'count': x_dim, 'min': 0, 'max': x_dim-1}
        }
    except Exception as e:
        print(f"Error generating orthogonal slices: {e}")
        return None

#  Route for front.html (Runs Model for Tumor Detection)
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

#  Route for advanced.html (No Model, Just Upload)
@app.route('/upload_nii', methods=['POST'])
def upload_nii():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    print(f"File received for 3D viewing: {file_path}")
    
    # Generate orthogonal slices for advanced viewer
    slice_info = generate_orthogonal_slices(file_path)
    
    if slice_info:
        return jsonify({
            'message': 'File uploaded successfully', 
            'file_path': f"/uploads/{filename}",
            'slice_info': slice_info
        })
    else:
        return jsonify({'error': 'Failed to process file'}), 500

#  New endpoint to get specific slices
@app.route('/get_slice/<axis>/<int:index>', methods=['GET'])
def get_slice(axis, index):
    try:
        if axis not in ['axial', 'coronal', 'sagittal']:
            return jsonify({'error': 'Invalid slice type'}), 400
            
        slice_path = f"/static/outputs/{axis}/slice_{index}.png"
        full_path = os.path.join(os.getcwd(), app.config['OUTPUT_FOLDER'], axis, f"slice_{index}.png")
        
        if not os.path.exists(full_path):
            return jsonify({'error': 'Slice not found'}), 404
            
        return jsonify({'slice_path': slice_path})
    except Exception as e:
        print(f"Error fetching slice: {e}")
        return jsonify({'error': 'Failed to fetch slice'}), 500

#  Routes for serving pages
@app.route('/')
def home():
    return render_template('front.html')

@app.route('/advanced')
def advanced():
    return render_template('advanced.html')

#  Route for serving uploaded files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/static/outputs/<path:filename>')
def output_file(filename):
    # Handle nested directories
    directory = os.path.join(app.config['OUTPUT_FOLDER'], os.path.dirname(filename))
    return send_from_directory(directory, os.path.basename(filename))

if __name__ == '__main__':
    app.run(debug=True)