# Series-de-Tiempo
# Time Series Analysis for Financial Applications

This repository contains the material for a course on **Time Series Analysis**, with a particular focus on **financial time series and volatility modeling**.

> **Note:**  
> Course content (notes, explanations, and examples) is written in **Spanish**, while this README is provided in **English** for broader accessibility.

---

## 📌 Course Overview

The objective of this course is to provide a solid theoretical and practical foundation in time series analysis, emphasizing models commonly used in economics and finance. The course progresses from basic concepts to advanced volatility modeling techniques used in real-world financial applications.

By the end of the course, students will be able to:
- Understand the structure and behavior of time series data  
- Apply ARIMA and SARIMA models for forecasting  
- Use information criteria for model selection  
- Model and interpret volatility using GARCH-type processes  
- Analyze financial time series with leptokurtic distributions  

---

## 📚 Syllabus

### 1. Introduction to Time Series
- Definition and components of time series  
- Stationarity and non-stationarity  
- Trend, seasonality, and noise  
- Practical examples with real data  

---

### 2. Model Identification and Selection

#### Likelihood-Based Methods
- Log-likelihood function  
- Akaike Information Criterion (AIC)  
- Bayesian Information Criterion (BIC)  

#### ARIMA Models
- ARIMA \((p, d, q)\)  
- Interpretation of parameters  
- Model estimation and diagnostics  

#### SARIMA Models
- Seasonal ARIMA \((p, d, q)(P, D, Q)_s\)  
- Seasonal patterns and periodicity  

#### Correlation Functions
- Autocorrelation Function (ACF)  
- Partial Autocorrelation Function (PACF)  
- Model identification using ACF and PACF  

---

### 3. Volatility Modeling and Robust Processes

#### Financial Time Series
- Stylized facts of financial returns  
- Volatility clustering  
- Heavy tails and non-normality  

#### GARCH Models
- ARCH and GARCH processes  
- Conditional heteroskedasticity  
- Volatility forecasting  

#### Extensions and Variants
- EGARCH  
- Other GARCH-family models  
- Asymmetric effects and leverage  

#### Leptokurtic Distributions
- Excess kurtosis in financial returns  
- Implications for risk modeling  

---

## 🛠 Tools & Libraries

Depending on the implementation, the course may use:
- Python (NumPy, pandas, statsmodels, arch)
- R (forecast, tseries, rugarch)
- Jupyter Notebooks / R Markdown

---

## 🎯 Intended Audience

This repository is intended for:
- Students of economics, finance, and quantitative disciplines  
- Practitioners interested in time series and volatility modeling  
- Anyone seeking a structured introduction to ARIMA and GARCH models  

A basic background in statistics and probability is recommended.

---

## 📄 License

This project is intended for educational purposes.  
License information can be added here if applicable.

---

## ✨ Author

Course material developed for academic and applied financial time series analysis.

