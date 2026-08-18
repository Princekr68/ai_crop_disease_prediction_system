# 🌿AI Crop Disease Prediction using CNN

A deep learning project that predicts crop diseases from leaf images using a **Convolutional Neural Network (CNN)**.

The application is built with Streamlit, so users can simply upload a leaf image or capture one using the camera and get a prediction with the model's confidence score.

---

##  What This Project Does

-  Upload or capture a leaf image
-  Preprocess the image for the CNN model
-  Predict the crop disease
-  Show prediction confidence
-  Identify healthy leaves
-  Run everything through a Streamlit web interface

---

##  Model

**Custom CNN built with TensorFlow / Keras**

| Detail | Value |
|---|---|
| Input Image | 224 × 224 |
| Classes | 38 |
| Training Images | 70,295 |
| Validation Images | 17,572 |
| Validation Accuracy | ~98% |

The CNN uses convolutional layers, batch normalization, max pooling, data augmentation and fully connected layers for classification.

---

##  Results

The model achieved approximately **98% accuracy** on the validation set.

| Metric | Score |
|---|---:|
| Accuracy | 98% |
| Precision | 99% |
| Recall | 98% |
| F1-Score | 98% |

A classification report and confusion matrix were also used to evaluate the model.

---

## 🌾 Supported Crops

The model currently works with multiple crops including:

**Apple · Blueberry · Cherry · Corn · Grape · Peach · Pepper · Potato · Raspberry · Soybean · Squash · Strawberry · Tomato**

---

##  Technologies

`Python` · `TensorFlow` · `Keras` · `CNN` · `Streamlit` · `NumPy` · `Pillow` · `Matplotlib`

---

## Run Locally

Clone the repository:

```bash
git clone https://github.com/Princekr68/ai_crop_disease_prediction_system.git
cd ai_crop_disease_prediction_system
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

##  Project Structure

```text
ai_crop_disease_prediction_system/
│
├── app.py
├── requirements.txt
├── README.md
│
├── model/
│   └── crop_disease_model_cnn.keras
│
└── crop_disease_prediction_cnn.ipynb
```

---

## 📌 Dataset

The model was trained using the [**New Plant Diseases Dataset (Augmented)**.]
(https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)

---


