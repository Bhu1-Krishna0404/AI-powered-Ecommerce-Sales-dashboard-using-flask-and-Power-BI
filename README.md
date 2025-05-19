# AI-powered-Ecommerce-Sales-dashboard-using-flask-and-Power-BI


This project is a complete AI-powered web application that predicts e-commerce sales and profit using machine learning and visualizes insights with interactive Python plots and a Power BI dashboard. It is built using Flask and can be easily customized for other datasets.

---

## 🚀 Features

- 📈 Predict e-commerce **sales** using a trained ML regression model
- 📊 Classify **profit status** using a classification model
- 📉 Interactive dashboards with **Seaborn**, **Matplotlib**, **Plotly**
- 🖥️ Integrated **Power BI** dashboard viewer
- 🌐 Web interface with Flask

---

## 🧰 Tech Stack

- **Frontend**: HTML (via Flask templates)
- **Backend**: Python 3, Flask
- **ML Models**: scikit-learn (DecisionTreeRegressor), XGBoost (profit classifier)
- **Data Processing**: pandas, numpy
- **Visualization**: seaborn, matplotlib, plotly
- **BI Tool**: Microsoft Power BI (Desktop)

---

## 📁 Project Structure

📦 ai-ecommerce-dashboard/
├── app.py # Main Flask App
├── templates/
│ └── index.html # Web frontend
├── static/ # (Optional) CSS/JS assets
├── model/
│ ├── sales_model.pkl # Trained Sales Regression Model
│ └── profit_model.pkl # Trained Profit Classifier Model
├── Ecommerce.pbix # Power BI Dashboard
├── requirements.txt # Python Dependencies
└── README.md # Project Overview


# Create Virtual Environment (Recommended)
bash
Copy
Edit
python -m venv venv
venv\Scripts\activate
#Install Requirements
bash
Copy
Edit
pip install -r requirements.txt
# Power BI Configuration
Before launching the app, make sure Power BI Desktop is installed on your system.

In your app.py, configure these two paths to match your system:

python
Copy
Edit
# Power BI paths
POWER_BI_PATH = r"C:\Program Files\Microsoft Power BI Desktop\bin\PBIDesktop.exe"

PBIX_FILE = r"C:\Users\YourUsername\Desktop\file\Ecommerce.pbix"
Replace with your actual paths.

🧠 ML Models Used
📊 Sales Prediction:
Model: DecisionTreeRegressor

Library: scikit-learn (v1.0.2 - important due to compatibility)

💰 Profit Status Classification:
Model: XGBoostClassifier

Library: xgboost

⚠️ Make sure to train and save models with the same version of scikit-learn and xgboost as used at inference time.

🖥️ Run the Application
bash
Copy
Edit
python app.py
Visit http://localhost:5000 in your browser.

# 📈 Power BI Dashboard
Open the file Ecommerce.pbix in Power BI Desktop to access advanced insights such as:

Monthly Sales Trends

Region-wise Performance

Category Profitability

Product Sales Comparison

You can also launch Power BI from the Flask app using the configured paths.
