# 📱 SmartPrice AI

### Smartphone Price Prediction & Market Intelligence Platform

**SmartPrice AI** is a machine-learning-based smartphone price prediction system that estimates the expected selling price of a smartphone based on its hardware specifications, features, and configuration.

The project follows an end-to-end data science workflow:

```text
Web Scraping
     ↓
Data Collection
     ↓
Data Cleaning
     ↓
Exploratory Data Analysis
     ↓
Feature Engineering
     ↓
Model Training
     ↓
Model Evaluation
     ↓
Price Prediction
     ↓
Interactive Streamlit Application
```

The system compares **Multiple Linear Regression** and **Random Forest Regression** models and provides an interactive interface where users can enter smartphone specifications and receive estimated prices from both models.

---

# 🚀 Key Features

### 📊 Smartphone Data Collection

The project includes a Selenium-based scraping workflow for collecting smartphone information from **Smartprix**.

The scraper automates browser interaction, applies filters, loads additional phone listings, and saves the resulting HTML content for further processing.

---

### 🧹 Data Cleaning

Raw smartphone data is processed to make it suitable for machine-learning workflows.

The repository contains dedicated preprocessing notebooks and multiple versions of the processed datasets:

```text
smartphones.csv
        ↓
EDA_smartphones.csv
        ↓
Cleaned_SmartPhone.csv
        ↓
final_smartphones.csv
```

This separation makes the data-processing workflow easier to inspect and reproduce.

---

### 🔎 Exploratory Data Analysis

The project performs exploratory analysis to understand relationships between smartphone specifications and their prices.

Important attributes include:

* Brand
* Rating
* 5G support
* Processor
* Processor cores
* Processor speed
* RAM
* ROM
* Battery capacity
* Fast charging
* Display size
* Display refresh rate
* Camera configuration
* Camera megapixels
* Memory-card support
* Operating system
* OS version

These features are subsequently used by the prediction pipeline.

---

### 🤖 Multiple Regression Models

The application evaluates two regression approaches:

#### 1. Multiple Linear Regression

A linear regression pipeline is constructed using:

* Categorical feature encoding
* One-hot encoding
* Numerical features
* Linear Regression

#### 2. Random Forest Regression

A second pipeline uses:

* One-hot encoding
* Numerical features
* Random Forest Regressor

Both pipelines are trained on the smartphone dataset and evaluated on a held-out test set.

---

### 📈 Model Evaluation

The application calculates multiple regression metrics for both models:

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* Median Absolute Error
* R² Score

The metrics are displayed inside the Streamlit application's evaluation section, allowing the models to be compared side by side.

---

### 🖥️ Interactive Streamlit Application

The project includes a Streamlit interface where users can configure a hypothetical smartphone using its specifications.

Users can select or provide:

```text
Brand
Rating
5G Support
Chipset
Processor Company
Processor Name
Processor Cores
Processor Speed
RAM
ROM
Battery Capacity
Fast Charging
Display Size
Refresh Rate
Rear Cameras
Front Cameras
Rear Camera MP
Front Camera MP
Memory Card Support
Operating System
OS Version
```

The application then predicts the smartphone's estimated price using both trained regression models.

---

# 🧠 System Architecture

```text
                         ┌──────────────────────┐
                         │   Smartprix Website  │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Selenium Web Scraper  │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │    Raw Smartphone    │
                         │        Data          │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Data Cleaning & EDA  │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Feature Engineering  │
                         │ & Encoding           │
                         └──────────┬───────────┘
                                    │
                         ┌──────────┴───────────┐
                         │                      │
                         ▼                      ▼
                ┌──────────────────┐   ┌──────────────────┐
                │ Linear Regression│   │ Random Forest    │
                │                  │   │ Regression       │
                └────────┬─────────┘   └────────┬─────────┘
                         │                      │
                         └──────────┬───────────┘
                                    ▼
                         ┌──────────────────────┐
                         │ Model Evaluation     │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Streamlit Prediction │
                         │       Interface      │
                         └──────────────────────┘
```

---

# 🔄 End-to-End ML Pipeline

## 1. Data Collection

Smartphone specifications are collected from Smartprix using Selenium.

The scraper opens the mobile listing page, interacts with filters, repeatedly loads additional listings, and saves the resulting HTML for processing.

```text
Smartprix
   ↓
Selenium
   ↓
Browser Automation
   ↓
Load Smartphone Listings
   ↓
HTML Extraction
```

---

## 2. Data Cleaning

The raw dataset is cleaned and transformed into a structured dataset.

```text
Raw Data
   ↓
Missing Value Handling
   ↓
Data Type Conversion
   ↓
Text Normalization
   ↓
Feature Cleaning
   ↓
Clean Dataset
```

---

## 3. Exploratory Data Analysis

The cleaned dataset is analyzed to understand:

* Distribution of smartphone prices
* Relationship between RAM and price
* Relationship between ROM and price
* Battery capacity trends
* Processor characteristics
* Camera specifications
* Display specifications
* Brand-level pricing
* Other feature-price relationships

---

## 4. Feature Preparation

Categorical features are transformed using **OneHotEncoder**.

The application uses a `ColumnTransformer` to apply categorical encoding while passing numerical features through the pipeline.

Conceptually:

```text
Categorical Features
        │
        ▼
 One-Hot Encoding
        │
        ├───────────────┐
                        │
Numerical Features ────┤
                        ▼
                 Feature Matrix
                        │
                        ▼
                   ML Model
```

---

# 🤖 Model Training

The dataset is divided into training and testing sets using an 80/20 split.

```text
Dataset
   │
   ├───────────────┐
   │               │
   ▼               ▼
80% Training     20% Testing
   │               │
   ▼               │
Model Training     │
   │               │
   └───────┬───────┘
           ▼
      Evaluation
```

Two models are trained independently:

```text
                    Training Data
                         │
             ┌───────────┴───────────┐
             │                       │
             ▼                       ▼
      Linear Regression       Random Forest
             │                       │
             ▼                       ▼
        Predictions            Predictions
             │                       │
             └───────────┬───────────┘
                         ▼
                   Model Comparison
```

---

# 📊 Model Comparison

The application generates a comparison table containing:

| Metric                | Multiple Linear Regression |             Random Forest |
| --------------------- | -------------------------: | ------------------------: |
| MAE                   |  Calculated during runtime | Calculated during runtime |
| MSE                   |  Calculated during runtime | Calculated during runtime |
| RMSE                  |  Calculated during runtime | Calculated during runtime |
| Median Absolute Error |  Calculated during runtime | Calculated during runtime |
| R² Score              |  Calculated during runtime | Calculated during runtime |

The application also presents Random Forest as the more accurate model in its current conclusion.

> **Note:** Model performance can vary depending on the dataset version, train/test split, preprocessing, and model configuration.

---

# 🖥️ Streamlit Application

The Streamlit application provides interactive controls for configuring a smartphone.

### Example Input

```text
Brand: Samsung
Rating: 85
5G: Yes
Processor: Snapdragon
RAM: 8 GB
ROM: 128 GB
Battery: 5000 mAh
Display: 6.6 inches
Refresh Rate: 120 Hz
Rear Camera: 3
Front Camera: 1
Rear Camera: 50 MP
Front Camera: 16 MP
Fast Charging: Yes
OS: Android
```

The application generates:

```text
Predicted Price using Random Forest:
₹ XX,XXX

Predicted Price using Multiple Linear Regression:
₹ XX,XXX
```

The exact prediction depends on the model trained on the current dataset.

---

# 🛠️ Technology Stack

| Technology                  | Purpose                                  |
| --------------------------- | ---------------------------------------- |
| **Python**                  | Data science and application development |
| **Pandas**                  | Data manipulation                        |
| **NumPy**                   | Numerical processing                     |
| **Scikit-learn**            | Machine learning                         |
| **Linear Regression**       | Baseline regression model                |
| **Random Forest Regressor** | Non-linear regression model              |
| **OneHotEncoder**           | Categorical feature encoding             |
| **ColumnTransformer**       | Feature preprocessing                    |
| **Pipeline**                | ML preprocessing + model workflow        |
| **Streamlit**               | Interactive web application              |
| **Selenium**                | Smartphone data collection               |

The repository's requirements file explicitly includes Streamlit, NumPy, Pandas, and Scikit-learn.

---

# 📂 Project Structure

```text
smartphone_price_predictor/
│
├── smartphones.csv
├── EDA_smartphones.csv
├── Cleaned_SmartPhone.csv
├── final_smartphones.csv
│
├── smartprix.py
├── app.py
│
├── smartprix-phones.ipynb
├── Cleaning_SmartPhones.ipynb
├── EDA.ipynb
├── MultiLinear_Regression_Model.ipynb
│
├── df.pkl
├── pipe.pkl
│
└── requirements.txt
```

The repository currently contains the scraping script, multiple data-processing/model-development notebooks, several dataset stages, serialized artifacts, the Streamlit application, and requirements file.

---

# 📓 Notebook Workflow

### `smartprix-phones.ipynb`

Exploration and processing of the smartphone data collected from the source website.

### `Cleaning_SmartPhones.ipynb`

Data-cleaning and transformation workflow.

### `EDA.ipynb`

Exploratory Data Analysis of smartphone specifications and prices.

### `MultiLinear_Regression_Model.ipynb`

Development and experimentation with the regression models.

The repository separates these stages into individual notebooks, making the overall data-science workflow easier to follow.

---

# ⚙️ Installation

## Prerequisites

Make sure you have:

* Python 3.x
* pip
* Google Chrome
* ChromeDriver if you want to reproduce the scraping workflow

---

## 1. Clone the Repository

```bash
git clone https://github.com/ANSH1370/smartphone_price_predictor.git

cd smartphone_price_predictor
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

The current repository requirements include:

```text
streamlit
numpy
pandas
scikit-learn
```

---

# ▶️ Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

Streamlit will provide a local URL, typically:

```text
http://localhost:8501
```

Open the URL in your browser and enter the desired smartphone specifications.

---

# 🕷️ Reproducing the Data Collection

The repository also includes a Selenium-based scraper.

The current `smartprix.py` script:

1. Starts a Chrome WebDriver.
2. Opens the Smartprix mobile listing page.
3. Interacts with page filters.
4. Loads additional smartphone listings.
5. Detects when no additional content is loaded.
6. Saves the resulting page HTML.

Conceptually:

```text
Start Chrome
     ↓
Open Smartprix Mobiles
     ↓
Apply Filters
     ↓
Load More
     ↓
Check Page Height
     ↓
Repeat
     ↓
Save HTML
```

### ⚠️ Scraping note

The current scraper contains a machine-specific ChromeDriver path:

```text
C:/Users/Ansh/Desktop/chromedriver.exe
```

This path will not work on another machine without modification.

For a reusable version, configure ChromeDriver through the environment or use Selenium Manager.

---

# 🎯 Use Cases

SmartPrice AI can be used for:

* 📱 Estimating smartphone prices
* 💰 Understanding specification-to-price relationships
* 📊 Comparing regression models
* 🔎 Exploring smartphone market data
* 🧪 Demonstrating an end-to-end ML workflow
* 🎓 Learning regression and feature engineering
* 📈 Building data-driven pricing tools

---

# 🔮 Future Improvements

The project can be extended significantly:

* [ ] Add XGBoost / Gradient Boosting
* [ ] Hyperparameter tuning
* [ ] Cross-validation
* [ ] Feature importance visualization
* [ ] SHAP-based explainability
* [ ] Price-range classification
* [ ] Automated dataset refresh
* [ ] Scheduled scraping pipeline
* [ ] Price comparison across retailers
* [ ] Smartphone recommendation engine
* [ ] Budget-based phone recommendations
* [ ] "Best phone under ₹X" functionality
* [ ] Interactive price-vs-specification charts
* [ ] REST API for predictions
* [ ] Docker deployment
* [ ] Cloud deployment
* [ ] Model monitoring

---

# 🚀 Future Vision

The project can evolve from a simple price predictor into a complete smartphone intelligence platform:

```text
                 Smartphone Market Data
                         │
                         ▼
                Automated Data Collection
                         │
                         ▼
                   Data Processing
                         │
                         ▼
                Machine Learning Models
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
        Price Prediction       Value Analysis
              │                     │
              └──────────┬──────────┘
                         ▼
                 Recommendation Engine
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
        Best Phone Under       Best Value
           Budget              Smartphone
```

This would allow the system to answer not only:

> **"How much should this smartphone cost?"**

but also:

> **"Which smartphone gives me the best value for ₹30,000?"**

and:

> **"Which phone has the best specifications for my budget?"**

---

# 🎓 Learning Outcomes

This project demonstrates practical experience with:

* End-to-end machine-learning workflows
* Web scraping
* Selenium browser automation
* Data cleaning
* Exploratory Data Analysis
* Feature engineering
* Categorical encoding
* Regression
* Model evaluation
* Scikit-learn pipelines
* Random Forest
* Linear Regression
* Streamlit application development
* Model serialization
* Deployment-oriented ML application development

---

# 👨‍💻 Author

**Ansh Mangukiya**

AI Engineer | Machine Learning | NLP | Generative AI

GitHub:
https://github.com/ANSH1370

---

# ⭐ Project

If you find this project useful or interesting, feel free to explore the source code and give the repository a ⭐.

**Repository:**
https://github.com/ANSH1370/smartphone_price_predictor
