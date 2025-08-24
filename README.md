# 🏥 Health Diagnosis Prediction System  

## 🌐 Live Demo

Try the app here: [Health Diagnosis App](https://healthdiagnosis.streamlit.app/)

A **Machine Learning-based Health Diagnosis Application** that predicts the most likely diseases based on user symptoms.  
The project is built with **Python, Scikit-learn, TensorFlow, and Streamlit** to provide an interactive interface for users.  

This README explains the **project objectives, features, workflow, installation steps, usage guide, and future improvements**.  

## 📌 Introduction  

Health diagnosis has become one of the most important applications of artificial intelligence.  
With the rise of lifestyle diseases and complex medical conditions, **early detection** is critical.  

This project aims to provide an **AI-powered assistant** that:  
- Takes user symptoms as input.  
- Uses trained Machine Learning models to predict the most likely diseases.  
- Displays the **top 3 possible diseases** with confidence scores.  
- Provides additional information such as **causes, precautions, and recommended doctors**.  

By integrating **Streamlit**, the project offers a **simple and interactive web-based user interface** that can be easily deployed.  

---

## 🎯 Objectives  

The main goals of this project are:  
- ✅ To build a **symptom-to-disease prediction system** using ML models.  
- ✅ To provide an **easy-to-use web interface** with Streamlit.  
- ✅ To display **confidence scores** for the top predictions.  
- ✅ To offer **relevant information about diseases** (causes, precautions, recommended doctors).  
- ✅ To demonstrate the integration of **data science + web UI** in a single project.  

---

## ✨ Features  

- 🔹 **Symptom-based Prediction**: Enter your symptoms, and the model predicts possible diseases.  
- 🔹 **Top 3 Diseases**: Instead of a single result, the model shows the top 3 likely diseases with probability scores.  
- 🔹 **Interactive Web UI**: Built with Streamlit, no coding knowledge is needed to use the app.  
- 🔹 **Disease Information**: Each disease comes with details like causes, precautions, and recommended specialists.  
- 🔹 **Lightweight and Fast**: Optimized models ensure predictions are quick.  
- 🔹 **Scalable**: Can be extended to include more symptoms, diseases, and models.  

---

## 🛠 Tech Stack  

- **Programming Language**: Python 3.x  
- **Libraries**:  
  - `numpy`, `pandas` (data handling)  
  - `scikit-learn` (ML models)  
  - `tensorflow`, `keras` (deep learning)  
  - `joblib` (model saving/loading)  
  - `streamlit` (web UI)  
- **Tools**:  
  - PyCharm (IDE)  
  - Git (Version Control)  
  - Anaconda/Virtualenv (environment management)  

---

## 🏗 System Architecture  

The system follows this workflow:  

1. **Data Preprocessing**  
   - Cleaning the dataset.  
   - Encoding symptoms and diseases.  

2. **Model Training**  
   - Training ML models  Neural Network
   - Evaluating model accuracy.  
   - Selecting the best-performing model.  

3. **Prediction Layer**  
   - User enters symptoms.  
   - Encoded symptoms are passed to the model.  
   - Model predicts probability scores for each disease.  

4. **User Interface**  
   - Built with Streamlit.  
   - Displays top 3 predicted diseases with confidence.  
   - Provides information on each disease.  

---

## 📂 Dataset  

- The dataset contains **diseases and associated symptoms**.  
- Each row represents a disease with binary symptom indicators (`1` if symptom present, `0` otherwise).  
- The final column is the **target disease label**.  

Example:  

| fever | cough | chest_pain | nausea | disease     |  
|-------|-------|------------|--------|-------------|  
| 1     | 1     | 0          | 0      | Flu         |  
| 0     | 1     | 1          | 1      | Pneumonia   |  
| 1     | 0     | 0          | 0      | Dengue      |  

---

## 🤖 Model Training  

We trained multiple models, including:  
- **Decision Tree**  
- **Random Forest**  
- **LightGBM**  
- **Neural Network (Keras)**  


Final model was saved as `model.pkl` for deployment.  

---

## 📂 Project Structure  

