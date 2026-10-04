# 🧠 Brain Tumor Detection & Classification (Flask Web App)

An end-to-end deep learning web application built with **Flask**, **TensorFlow/Keras**, and **OpenCV** to classify brain MRI scans into four distinct tumor categories.

---

## 📌 Overview

This project assists in the preliminary screening of brain tumors from MRI scans. It provides an intuitive web interface for direct user uploads and an automated REST API endpoint for integration into external pipelines.

### Supported Classes
- **Glioma Tumor**
- **Meningioma Tumor**
- **Pituitary Tumor**
- **No Tumor**

---

## 🛠️ Tech Stack & Dependencies

- **Web Framework:** Flask, Flask-WTF, WTForms, Werkzeug
- **Deep Learning:** TensorFlow 2.13.1, Keras 2.13.1
- **Computer Vision & Data:** OpenCV (`opencv-python`), NumPy, Pillow, h5py
- **Frontend:** HTML5, CSS3, Jinja2 Templates

---

## 📂 Project Structure

```text
Brain-Tumor-Prediction-Flask-App/
├── dataset/                     # Training and testing datasets
├── static/
│   └── image/                   # Temporary upload directory for inference
├── templates/
│   ├── home.html                # Image upload form
│   └── predict.html             # Classification result and confidence display
├── app.py                       # Flask server and routing logic
├── prediction.py                # Preprocessing and model inference routines
├── model-training-script.ipynb  # Dataset preparation & training notebook
├── model.h5                     # Pre-trained Keras model weights
├── requirements                 # Project dependencies list
├── .gitignore                   # Version control exclusions
└── README.md                    # Project documentation

## 📂 File & Directory Architecture

A comprehensive breakdown of all repository modules, data sources, and configuration assets.

---

### 1. Root Application Files

#### `app.py`
The primary backend entry point that initializes and serves the Flask web application[cite: 1].
* **Application Initialization:** Configures application parameters, secret keys, and runtime paths[cite: 1].
* **Form Validation:** Employs `Flask-WTF` via `UploadFileForm` for secure CSRF-protected file uploads and validation[cite: 1].
* **Web Route (`/`):**
  * `GET`: Renders the upload interface (`templates/home.html`)[cite: 1].
  * `POST`: Saves incoming MRI scans to `static/image/`, triggers classification via `prediction.py`, and renders results in `templates/predict.html`[cite: 1].
* **REST API Endpoint (`/api/predict`):** Handles programmatic `POST` multipart form requests, processes images, and returns structured JSON responses containing `success`, `filename`, `diagnosis`, and `confidence`[cite: 1].
* **Directory Provisioning:** Automatically provisions the `static/image/` folder upon startup if missing[cite: 1].

#### `prediction.py`
The computer vision and inference engine, isolating deep learning logic from HTTP routing[cite: 1, 4].
* **Model Instantiation:** Loads `model.h5` into memory via `keras.models.load_model`[cite: 4].
* **Label Mapping:** Defines classification targets: `['glioma_tumor', 'no_tumor', 'meningioma_tumor', 'pituitary_tumor']`[cite: 4].
* **`preprocess_image(image_path, image_size)`:**
  * Ingests scans via OpenCV (`cv2.imread`)[cite: 4].
  * Resizes image matrices to $299 \times 299$ pixels[cite: 4].
  * Normalizes color pixel tensors into the $[0, 1]$ floating-point range[cite: 4].
  * Expands array dimensions (`np.expand_dims`) to provide batch-ready tensor shapes `(1, 299, 299, 3)`[cite: 4].
* **`predict_tumor_class(model, image_path, labels)`:** Executes forward propagation through `model.predict()`, derives the dominant class label using `np.argmax()`, and computes confidence percentages via `np.max() * 100`[cite: 4].

#### `model.h5`
The serialized HDF5 binary artifact containing the complete trained deep learning model[cite: 2, 4].
* Stores neural network topology, learned weights, convolutional kernels, and optimizer states[cite: 2].
* Receives inputs of shape `(None, 299, 299, 3)` and generates a 4-class softmax probability distribution[cite: 2].
* Pre-loaded persistently during application startup to minimize per-request inference latency[cite: 4].

#### `model-training-script.ipynb`
The complete research and development pipeline for dataset processing and model optimization[cite: 3].
* Extracts raw scan slices across the four target categories from `dataset/`[cite: 3].
* Performs image resizing to $299 \times 299$, data normalization, and array transformations[cite: 3].
* Merges, splits, and shuffles training and validation distributions[cite: 3].
* Implements model architecture (Convolutional Neural Network / Transfer Learning), compilation settings, loss functions, optimizer configurations, and validation metrics (confusion matrices, classification reports)[cite: 2, 3].

#### `requirements`
The pinned dependency manifest guaranteeing environment parity across systems[cite: 5].
* **Web Frameworks:** `Flask`, `Flask-WTF`, `WTForms`, `Werkzeug`[cite: 5].
* **Deep Learning & Vision:** `tensorflow`, `keras`, `opencv-python`, `numpy`, `h5py`, `scikit-learn`[cite: 5].
* Designed for reproducible setup using `pip install -r requirements`[cite: 5].

#### `.gitignore`
Version control exclusion list that prevents tracking extraneous, local, or large binary files.
* **Environments:** `venv/`, `env/` (prevents checking in machine-specific virtual environments).
* **Bytecode & Cache:** `__pycache__/`, `*.py[cod]` (omits compiled Python artifacts).
* **Notebook Metadata:** `.ipynb_checkpoints/` (excludes local Jupyter autosaves).
* **Transient Storage:** `static/image/*` (ignores runtime user uploads while keeping folder structure tracked via `.gitkeep`).
* **Large Files:** `model.h5` (prevents pushes exceeding GitHub's 100 MB file limit).

#### `README.md`
Front-facing repository documentation detailing installation steps, runtime setup, architectural overviews, API usage, and operational guidelines.

---

### 2. Template Directory (`templates/`)

Jinja2 HTML templates responsible for rendering application views[cite: 1].

* **`templates/home.html`**
  * The main dashboard interface[cite: 1].
  * Displays the upload interface and file selector using the Flask-WTF `UploadFileForm`[cite: 1].
  * Submits uploads to `/` via HTTP `POST`[cite: 1].
* **`templates/predict.html`**
  * The diagnostic results screen[cite: 1].
  * Renders the processed MRI scan alongside prediction metrics[cite: 1]:
    * Predicted tumor class (`predicted_class`)[cite: 1].
    * Calculated confidence percentage (`confidence`)[cite: 1].
  * Provides UI navigation to return to the upload interface for additional scans[cite: 1, 6].

---

### 3. Static Assets Directory (`static/`)

* **`static/image/`**
  * Serves as the temporary local file storage for incoming scans[cite: 1].
  * Incoming files sanitized via `werkzeug.utils.secure_filename` are written here before OpenCV loads them for prediction[cite: 1, 4].

---

### 4. Data Directory (`dataset/`)

Structured collection of MRI imaging data utilized across training and evaluation lifecycles[cite: 3].

* **`dataset/Training/`:** Subcategorized into `glioma_tumor/`, `meningioma_tumor/`, `no_tumor/`, and `pituitary_tumor/` containing the primary model training images[cite: 3].
* **`dataset/Testing/`:** Contains the holdout validation image splits across identical subcategory folders to evaluate model generalization and accuracy[cite: 3].
