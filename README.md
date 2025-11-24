# LungCancerDetection

A machine learning project for predicting lung cancer risk based on patient survey data. This repository contains the dataset, trained models, notebooks for analysis, and a Streamlit app for interactive predictions.

---

## **Project Structure**

```
LungCancerDetection/
│
├─ folder-data/            
│   ├─ survey lung cancer.csv
│   └─ lung_cancer_predictions.csv
│
├─ folder-model/           
│   ├─ lungcancer_svm.pkl
│   └─ lungcancer_scaler.pkl
│
├─ folder-notebooks/        
│   ├─ LungCancerDetection.ipynb
│   └─ test_notebook.ipynb
│
├─ lungcancer.py            
├─ README.md               
```

---

## **Dataset**

The dataset (`survey lung cancer.csv`) contains patient survey responses with the following features:

* Gender
* Age
* Smoking habits
* Yellow fingers
* Anxiety
* Peer pressure
* Chronic disease
* Fatigue
* Allergy
* Wheezing
* Alcohol consumption
* Coughing
* Shortness of breath
* Swallowing difficulty
* Chest pain

**Target:** `LUNG_CANCER` (Yes/No)

---

## **Models**

This project trains and evaluates multiple machine learning models:

* Random Forest
* Decision Tree
* Support Vector Machine (SVM) ✅ *(used for Streamlit app)*
* Perceptron

The trained SVM model (`lungcancer_svm.pkl`) along with the scaler (`lungcancer_scaler.pkl`) are saved in `folder-model/` for use in the web app.

---

## **Installation & Setup**

1. Clone the repository:

```bash
git clone https://github.com/yourusername/LungCancerDetection.git
cd LungCancerDetection
```

2. Install required packages:

```bash
pip install -r requirements.txt
```

*If you don’t have a `requirements.txt`, install manually:*

```bash
pip install streamlit pandas scikit-learn joblib numpy
```

---

## **Run the Streamlit App**

Make sure you are in the folder containing `lungcancer.py`, then run:

```bash
streamlit run lungcancer.py
```

Open the URL provided in the terminal (usually `http://localhost:8501`) in your browser to use the interactive lung cancer prediction app.

---

## **Usage**

1. Fill in patient information in the app.
2. Click **Predict** to get the result:

   * **Lung Cancer Detected** with probability
   * **No Lung Cancer Detected** with probability
