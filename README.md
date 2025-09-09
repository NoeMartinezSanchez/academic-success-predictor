# 🎓 Academic Success Predictor

**Machine Learning Platform for Predicting Student Success in Online Education**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎯 Project Overview

A comprehensive machine learning solution that predicts student success in online education programs with **89.8% ROC-AUC accuracy**. The platform combines advanced feature engineering, ensemble learning, and an intuitive web interface to help educational institutions proactively identify at-risk students and implement targeted intervention strategies.

### 🏆 Key Achievements
- **89.8% ROC-AUC** and **82.5% accuracy** on 44K+ student predictions
- **32% reduction in demographic bias** through balanced feature importance
- **Real-time predictions** with personalized intervention recommendations
- **Production-ready web application** for non-technical educational staff

## 🚀 Features

### 📊 Advanced Analytics
- **Feature Engineering**: 20+ engineered features from socioeconomic, technological, and educational data
- **Model Optimization**: RandomizedSearchCV hyperparameter tuning with 5-fold cross-validation
- **Interpretable ML**: Balanced Random Forest preventing over-reliance on demographic factors

### 🖥️ Interactive Web Application
- **Real-time Predictions**: Instant success probability assessment
- **Risk Categorization**: Automatic classification (High Risk, Medium Risk, Success Probable)
- **Personalized Recommendations**: Tailored intervention strategies
- **Dynamic Visualizations**: Gauge charts and feature importance plots

### 🔧 Technical Excellence
- **Complete ML Pipeline**: Data preprocessing, model training, and deployment
- **Production Architecture**: Saved model artifacts and automated preprocessing
- **Cross-validation**: Robust model validation with statistical significance testing
- **Error Handling**: Comprehensive validation and user-friendly error messages

## 📈 Model Performance

| Metric | Score |
|--------|-------|
| **ROC-AUC** | 0.8981 |
| **Accuracy** | 82.49% |
| **Precision (Success)** | 80% |
| **Recall (Success)** | 88% |
| **F1-Score** | 0.84 |

### 📋 Classification Report
```
              precision    recall  f1-score   support
    No Éxito       0.86      0.77      0.81     21300
       Éxito       0.80      0.88      0.84     22801
    
    accuracy                           0.82     44101
```

## 🛠️ Technology Stack

- **Machine Learning**: Scikit-learn, Random Forest, Ensemble Methods
- **Data Processing**: Pandas, NumPy, Feature Engineering
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Web Application**: Streamlit, Interactive Dashboards
- **Model Deployment**: Pickle, Production Pipeline
- **Development**: Jupyter Notebooks, Python 3.8+

## 📁 Project Structure

```
academic-success-predictor/
├── 📊 data/
│   ├── datos_sinteticos_prepa_linea_completo.csv
│   └── processed/
├── 📓 notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_optimization.ipynb
├── 🤖 models/
│   ├── modelo_exito_academico_RF_optimizado.pkl
│   └── modelo_metadatos.pkl
├── 🚀 app/
│   ├── streamlit_app.py
│   ├── predictor.py
│   └── utils/
├── 📈 results/
│   ├── model_performance.png
│   ├── feature_importance.png
│   └── confusion_matrix.png
└── 📋 requirements.txt
```

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/yourusername/academic-success-predictor.git
cd academic-success-predictor
pip install -r requirements.txt
```

### Run Web Application
```bash
streamlit run app/streamlit_app.py
```

### Jupyter Notebooks
```bash
jupyter notebook notebooks/
```

## 📊 Key Insights

### Most Important Predictive Factors:
1. **Edad** (54.8%) - Student age
2. **Estudios Previos** (5.1%) - Previous academic experience
3. **Horas de Trabajo** (5.0%) - Work hours per week
4. **Ingresos del Hogar** (3.4%) - Household income
5. **Recursos Tecnológicos** (3.3%) - Technology access score

### Model Advantages:
- **Reduced Bias**: 32% less age-dependency compared to gradient boosting
- **Interpretability**: Clear feature importance for educational stakeholders
- **Balanced Performance**: Consistent results across different student populations
- **Actionable Insights**: Identifies modifiable risk factors

## 🎯 Use Cases

- **Early Warning Systems**: Identify at-risk students in first weeks of semester
- **Resource Allocation**: Prioritize support services for high-risk populations  
- **Intervention Design**: Data-driven strategies for student retention
- **Institutional Analytics**: Program effectiveness and outcome prediction

## 🔬 Methodology

1. **Data Preparation**: Cleaning, feature engineering, and categorical encoding
2. **Model Selection**: Comparison of Random Forest, Gradient Boosting, and Logistic Regression
3. **Optimization**: RandomizedSearchCV with 25 iterations and 3-fold CV
4. **Validation**: Stratified train-test split with cross-validation
5. **Deployment**: Production pipeline with Streamlit web interface

## 📧 Contact & Collaboration

Feel free to reach out for collaborations, questions, or suggestions:

- **LinkedIn**: [Your LinkedIn Profile]
- **Email**: [your.email@domain.com]
- **Portfolio**: [Your Portfolio Website]

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

⭐ **If you found this project helpful, please consider giving it a star!**

*Built with ❤️ for educational impact and student success*
