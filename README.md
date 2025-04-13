# 🛡️ Online Fraud Detection System using Flask & ML

A Flask-based web application that predicts whether a financial transaction is fraudulent using trained machine learning models (XGBoost and Logistic Regression).

---

for Dataset: https://www.kaggle.com/datasets/yashajaysingh/fraud-detection

## ✨ Features
- User-friendly web UI
- Form-based input for transaction details
- Real-time fraud prediction with probability
- Supports XGBoost and Logistic Regression
- Integrated data preprocessing pipeline

---

## 🛠️ Technologies & Tools

### Backend
- **Python 3.11+**
- **Flask**
- **Jinja2**
- **scikit-learn**
- **xgboost**
- **pandas**, **numpy**

### ML & Utilities
- **StandardScaler**, **OneHotEncoder**, **ColumnTransformer**
- **Pipeline**
- **pickle** for model serialization

---

## 📁 Project Structure

```
Frud_detection/
├── app.py                      # Main Flask app
├── onlinefraud.csv            # Dataset (CSV format)
├── xgb.sav                    # Trained XGBoost model
├── lr.sav                     # Trained Logistic Regression model
├── templates/
│   └── index.html             # UI template
```

---

## 🗒️ Installation & Running Locally

```bash
# Clone the repository
git clone https://github.com/your-username/fraud-detection-flask.git
cd fraud-detection-flask

# Install required packages
pip install -r requirements.txt

# Run the Flask app
python app.py
```

Open your browser and navigate to:
```
http://127.0.0.1:5000/
```

---

## 🌍 Demo Screenshot
_Add a screenshot here of the app UI after prediction._

---

## 🌐 Requirements

Create a file named `requirements.txt` with the following content:

```
flask
numpy
pandas
scikit-learn
xgboost
```

---

## 📖 Model Training Notebook
You can optionally include a `train_model.ipynb` to demonstrate how the models were trained and saved using `pickle`.

---

## 📄 License
[MIT License](LICENSE)

---

## 💪 Contributions
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

