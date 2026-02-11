# 🎓 Student Final Grade (G3) Predictor

A machine learning web application that predicts a student's final grade (G3) based on their academic performance and habits, built with **Scikit-learn** and deployed using **Streamlit**.

## 📌 About

This project uses the [UCI Student Performance Dataset](https://archive.ics.uci.edu/ml/datasets/Student+Performance) to train a **Random Forest Regressor** that predicts a student's final grade (G3, on a 0–20 scale) from 6 key features:

| Feature | Description |
|---------|-------------|
| **G1** | First period grade (0–20) |
| **G2** | Second period grade (0–20) |
| **Absences** | Number of school absences |
| **Study Time** | Weekly study time (1–4 scale) |
| **Failures** | Number of past class failures (0–4) |
| **Higher Education** | Whether the student wants to pursue higher education |

## 🚀 Demo

The Streamlit app provides an interactive interface where users can adjust student parameters and get an instant grade prediction.

## 🛠️ Tech Stack

- **Python**
- **Scikit-learn** — Model training (Random Forest, Linear Regression)
- **Pandas / NumPy** — Data processing
- **Matplotlib** — Visualizations
- **Streamlit** — Web app deployment

## 📂 Project Structure

```
├── ML_Project_SSI.ipynb     # Training notebook (EDA, model training, evaluation)
├── train_model.py           # Script to retrain and export model artifacts
├── main.py                  # Streamlit web application
├── random_forest_model.pkl  # Trained model
├── scaler.pkl               # Fitted StandardScaler
├── feature_columns.pkl      # Feature column list
├── student-por.csv          # Portuguese student data
├── student-mat.csv          # Math student data
├── predictions.csv          # Exported test set predictions
└── README.md
```

## ⚙️ Setup & Usage

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/student-grade-predictor.git
cd student-grade-predictor
```

### 2. Install dependencies
```bash
pip install pandas numpy scikit-learn matplotlib streamlit
```

### 3. Train the model
Run the notebook or the training script:
```bash
python train_model.py
```
This generates the `.pkl` artifacts needed by the app.

### 4. Launch the app
```bash
streamlit run main.py
```

## 📊 Model Performance

| Model | MAE | RMSE | R² | Adjusted R² |
|-------|-----|------|----|-------------|
| Linear Regression | 0.9425 | 1.4956 | 0.8467 | 0.8421 |
| **Random Forest** | 0.9590 | 1.5292 | 0.8397 | 0.8349 |

## 📝 License

This project is for educational purposes.
