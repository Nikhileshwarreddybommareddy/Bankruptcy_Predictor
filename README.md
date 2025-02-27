# Company Bankruptcy Predictor

![Bankruptcy Predictor Banner](https://img.shields.io/badge/ML-Bankruptcy%20Predictor-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-brightgreen)
![Streamlit](https://img.shields.io/badge/Streamlit-1.10%2B-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange)
![pandas](https://img.shields.io/badge/pandas-1.3%2B-lightblue)
![matplotlib](https://img.shields.io/badge/matplotlib-3.5%2B-purple)
![seaborn](https://img.shields.io/badge/seaborn-0.11%2B-darkblue)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-brightgreen)
![AWS](https://img.shields.io/badge/AWS-Elastic%20Beanstalk-orange)
![Docker](https://img.shields.io/badge/Docker-20.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)

A machine learning application that predicts company bankruptcy risk based on financial ratios.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [CI/CD Pipeline](#cicd-pipeline)
- [Deployment](#deployment)
- [Model Information](#model-information)
- [Financial Ratios Explained](#financial-ratios-explained)
- [Contributing](#contributing)
- [License](#license)

## 🔍 Overview

The Company Bankruptcy Predictor is a machine learning application built with Streamlit that evaluates a company's financial health by analyzing various financial ratios. The application predicts the likelihood of bankruptcy, providing an early warning system for companies to take corrective actions.

## ✨ Features

- **Interactive UI**: User-friendly interface to input financial ratios
- **Real-time Prediction**: Instant bankruptcy risk assessment
- **Visual Results**: Visualizations of risk levels and contributing factors
- **Detailed Analysis**: Identification of key risk factors and strengths
- **Actionable Recommendations**: Suggestions based on financial performance
- **Educational Insights**: Explanation of financial ratios and their significance
- **Containerized Deployment**: Docker support for consistent deployment
- **CI/CD Integration**: Automated testing and deployment pipeline
- **Cloud Deployment**: AWS Elastic Beanstalk integration

## 🛠️ Technologies Used

### Data Science & Machine Learning
- **Python**: Primary programming language
- **pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **scikit-learn**: Machine learning algorithms and preprocessing
- **matplotlib & seaborn**: Data visualization

### Web Application
- **Streamlit**: Interactive web application framework
- **Plotly**: Interactive visualization components

### DevOps & Deployment
- **Docker**: Application containerization
- **GitHub Actions**: CI/CD workflow automation
- **AWS Elastic Beanstalk**: Cloud deployment platform
- **Git**: Version control system

### Development Tools
- **Jupyter Notebooks**: Exploratory data analysis and model development
- **VS Code**: Code editing and development
- **Python virtual environments**: Dependency management

## 📂 Project Structure

```
BANKRUPTCY_PREDICTOR/
│
├── artifacts/                  # Model and preprocessor files
│   ├── best_model.pkl          # Trained ML model
│   └── preprocessor.pkl        # Data preprocessing pipeline
│
├── data/                       # Data directories
│   ├── processed/              # Processed datasets
│   └── raw/                    # Raw datasets
│
├── notebooks/                  # Jupyter notebooks
│   └── Nikhil_Bankruptcy_Predictor.ipynb  # Model development notebook
│
├── src/                        # Source code
│   ├── components/             # Modular components
│   ├── __init__.py             # Package initialization
│   ├── exception.py            # Custom exception handling
│   ├── logger.py               # Logging functionality
│   └── utils.py                # Utility functions
│
├── .ebextensions/              # AWS Elastic Beanstalk configuration
│   └── python.config           # Python environment settings
│
├── .gitignore                  # Git ignore file
├── application.py              # Main Streamlit application
├── Dockerfile                  # Docker configuration
├── requirements.txt            # Python dependencies
├── setup.py                    # Package setup script
└── README.md                   # Project documentation
```

## 🚀 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/bankruptcy-predictor.git
   cd bankruptcy-predictor
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv bankruptcy_venv
   source bankruptcy_venv/bin/activate  # On Windows: bankruptcy_venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

### Local Deployment

1. Run the Streamlit application:
   ```bash
   streamlit run application.py
   ```

2. Open your web browser and navigate to:
   ```
   http://localhost:8501
   ```

3. Enter the financial ratios for the company you wish to analyze and click on "Predict Bankruptcy Risk".

### Docker Deployment

1. Build the Docker image:
   ```bash
   docker build -t bankruptcy-predictor .
   ```

2. Run the Docker container:
   ```bash
   docker run -p 8501:8501 bankruptcy-predictor
   ```

3. Open your web browser and navigate to:
   ```
   http://localhost:8501
   ```

## 🔄 CI/CD Pipeline

This project implements a continuous integration and continuous deployment (CI/CD) pipeline using GitHub Actions to automate testing, building, and deployment processes.

### Workflow Steps

1. **Code Checkout**: Retrieves the latest code from the repository
2. **Environment Setup**: Sets up Python and required dependencies
3. **Code Linting**: Checks code quality and style
4. **Unit Testing**: Runs automated tests to ensure functionality
5. **Build Docker Image**: Creates a containerized application
6. **Push to Registry**: Uploads the Docker image to a container registry
7. **Deploy to AWS**: Deploys the application to AWS Elastic Beanstalk

### Workflow Triggers

- **Push to main branch**: Triggers full CI/CD pipeline
- **Pull Request**: Triggers code linting and testing only
- **Manual Trigger**: Can be initiated through GitHub UI

## 🚀 Deployment

### AWS Elastic Beanstalk

The application is deployed on AWS Elastic Beanstalk, providing a scalable and managed environment.

1. **Environment Configuration**: Configured in `.ebextensions/python.config`
2. **Scaling**: Auto-scaling based on traffic and resource utilization
3. **Monitoring**: AWS CloudWatch integration for performance monitoring
4. **Updates**: Zero-downtime deployments through the CI/CD pipeline

### Accessing the Deployed Application

The application is accessible through the AWS Elastic Beanstalk URL:
```
http://bankruptcy-predictor.elasticbeanstalk.com/
```

## 🧠 Model Information

The bankruptcy prediction model is built using a Random Forest Regressor trained on historical company financial data. The model analyzes various financial ratios to predict the likelihood of bankruptcy.

### Model Performance Metrics
- **R² Score**: 0.XX
- **Mean Absolute Error**: 0.XX
- **Root Mean Squared Error**: 0.XX

### Model Pipeline
1. **Data Preprocessing**: Handling missing values, outlier removal, and feature scaling
2. **Feature Selection**: Identifying the most relevant financial ratios
3. **Model Training**: Training with hyperparameter tuning
4. **Model Evaluation**: Performance assessment using cross-validation
5. **Model Deployment**: Saving the model for production use

## 📊 Financial Ratios Explained

The model uses the following financial ratios as predictors:

1. **Borrowing dependency**: Measures a company's reliance on borrowed funds to finance operations. Higher values indicate greater dependency on external financing.

2. **Current Liability to Current Assets**: Evaluates a company's ability to pay short-term obligations with short-term assets. Values greater than 1 indicate potential liquidity issues.

3. **Debt ratio %**: The percentage of a company's assets financed by debt. Higher values suggest higher financial leverage and risk.

4. **Net Income to Stockholder's Equity**: Also known as Return on Equity (ROE), measures profitability relative to equity investment. Higher values indicate better returns for shareholders.

5. **Net Value Per Share**: The per-share book value of the company. Higher values often suggest a more financially stable company.

6. **Net profit before tax/Paid-in capital**: Measures profitability relative to invested capital. Higher values indicate better utilization of capital.

7. **Operating Gross Margin**: Measures operational efficiency. Higher values indicate better profitability from core operations.

8. **Per Share Net profit before tax**: Earnings per share before tax deductions. Higher values suggest better profitability.

9. **Persistent EPS in the Last Four Seasons**: Consistency of earnings over time. Higher and stable values indicate consistent performance.

10. **ROA before interest and % after tax**: Return on Assets, measuring how efficiently a company uses its assets. Higher values indicate better asset utilization.

11. **Working Capital to Total Assets**: Measures a company's liquidity relative to its size. Higher values indicate better short-term financial health.

Created with ❤️ by Nikhil

For issues or questions, please open an issue on the GitHub repository.
