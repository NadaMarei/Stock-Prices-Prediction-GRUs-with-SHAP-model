# 📊 Explainable AI for Stock Price Prediction

## Thesis Implementation: GRU Model with SHAP Explanations for Stock Trend Forecasting

This application implements the empirical study from the MSc thesis **"Explainable AI with Deep Learning for Stock Prices Prediction on Time Series Big Data: An Empirical Study"** by Nada Marey Maged Hassan.
[![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg?style=for-the-badge&logo=python)](https://www.python.org)
[![License](https://img.shields.io/badge/License-Academic%20Use-green.svg?style=for-the-badge)](LICENSE)

---
### Try it Now! https://stockgrushapapp.streamlit.app/

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Thesis Results](#-thesis-results)
- [Installation](#-installation)
- [Usage Guide](#-usage-guide)
- [Application Structure](#-application-structure)
- [Model Architecture](#-model-architecture)
- [Technical Details](#-technical-details)
- [Project Structure](#-project-structure)
- [Troubleshooting](#-troubleshooting)
- [Citation](#-citation)
- [Contact](#-contact)

---

## 🎯 Overview

This interactive web application demonstrates a **stacked Gated Recurrent Unit (GRU)** model combined with **SHAP (SHapley Additive exPlanations)** for stock price trend prediction. The system achieves:

- **93.92% to 99.12%** directional accuracy across 1-month to 1-year forecast horizons
- **Superior performance** compared to 5 classical baseline methods
- **Interpretable predictions** through SHAP feature attribution
- **Real-time forecasts** for 10 major S&P 500 stocks

### Research Question
> *Can a GRU-based deep learning model with SHAP explanations provide both strong predictive performance and interpretable decision support for stock trend prediction?*

---

## ✨ Key Features

### 📈 Interactive Dashboard
- **5 Main Sections**: Overview, Model Performance, SHAP Analysis, Forecast Visualization, Literature Results
- **Real-time Ticker Selection**: Choose from 10 S&P 500 stocks
- **Multi-Horizon Forecasting**: 1 month, 3 months, 6 months, and 1 year

### 🔬 Model Capabilities
- **Stacked GRU Architecture**: 2-layer recurrent neural network (128→64 units)
- **15 Engineered Features**: Momentum, volatility, and volume indicators
- **Walk-Forward Validation**: Time-series aware evaluation
- **SHAP Explanations**: Feature importance with economic interpretation

### 📊 Visualization Features
- Interactive performance metrics tables
- ROC curves with AUC values
- Confusion matrices by horizon
- Feature importance plots (bar and beeswarm)
- Forecast vs. actual price charts with confidence intervals
- Per-ticker performance heatmaps

---

## 📊 Thesis Results

### Model Performance Summary

| Horizon | Accuracy | Precision | Recall | F1-Score | MAPE | RMSE |
|---------|----------|-----------|--------|----------|------|------|
| **1 Month** | 93.92% | 94.92% | 93.82% | 94.37% | 2.85% | 3.40% |
| **3 Months** | 95.61% | 96.33% | 95.45% | 95.89% | 2.10% | 2.60% |
| **6 Months** | 97.74% | 98.17% | 97.27% | 97.72% | 1.55% | 1.95% |
| **1 Year** | 99.12% | 99.09% | 99.09% | 99.09% | 1.05% | 1.30% |

### Top 5 Most Important Features (SHAP Analysis)

| Rank | Feature | SHAP Importance |
|------|---------|----------------|
| 1 | Lagged Closing Price (Close_t-1) | 12.02% |
| 2 | On-Balance Volume (OBV) | 9.29% |
| 3 | 50-day Simple Moving Average (SMA_50) | 7.85% |
| 4 | MACD Signal Line | 7.43% |
| 5 | MACD | 7.42% |

### Baseline Comparison

The GRU model significantly outperforms all classical baselines:

| Model | 1M | 3M | 6M | 1Y |
|-------|----|----|----|-----|
| **GRU (Ours)** | **93.92%** | **95.61%** | **97.74%** | **99.12%** |
| ARIMA-like | 56.60% | 51.33% | 44.29% | 40.00% |
| SMA 10/50 | 57.02% | 52.00% | 42.86% | 56.67% |
| Persistence | 56.81% | 52.67% | 51.43% | 56.67% |
| Mean-Reversion | 44.47% | 47.33% | 47.14% | 53.33% |
| Random | 50.85% | 51.33% | 57.14% | 46.67% |

---

## 🚀 Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Step 1: Download the Application

Save the `app.py` file to your local machine.

### Step 2: Create a Virtual Environment (Recommended)

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Dependencies
```
bash
pip install streamlit pandas numpy plotly seaborn matplotlib scikit-learn
Or create a requirements.txt file:

txt
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.17.0
seaborn>=0.12.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
Then install with:

bash
pip install -r requirements.txt
```

### Step 4: Run the Application
bash
streamlit run app.py
The application will automatically open in your default web browser at http://localhost:8501.

📖 Usage Guide
Navigation
The application features a sidebar navigation menu with 5 main sections:

📈 Overview - Research background, objectives, methodology, and key results

📊 Model Performance - Detailed performance metrics, baseline comparisons, confusion matrices

🔍 SHAP Analysis - Feature importance rankings, economic interpretation

📉 Forecast Visualization - Interactive price forecasts for selected stock and horizon

📚 Literature & Results - Research context, hypotheses validation, conclusions

##### Interactive Controls
Stock Ticker Dropdown: Select from AAPL, MSFT, GOOGL, NVDA, AMZN, META, TSLA, JPM, XOM, JNJ

Forecast Horizon Dropdown: Choose between 1 Month, 3 Months, 6 Months, or 1 Year

Understanding the Outputs
Forecast Visualization Page
Blue line: Historical training data

Green dashed line: Actual market prices

Red line: GRU model predictions

Pink shaded area: 95% confidence interval

SHAP Analysis Page
Bar chart: Global feature importance ranking

Beeswarm plot: Distribution of feature impacts

Pie chart: Feature grouping by economic category

🏗️ Application Structure
```text
stock-price-prediction-xai/
│
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
```

🧠 Model Architecture
Network Configuration
text
Input Layer (60 days × 15 features)
    ↓
GRU Layer 1 (128 units)
    ↓
Dropout (0.2)
    ↓
GRU Layer 2 (64 units)
    ↓
Dropout (0.2)
    ↓
Dense Layer (32 units, ReLU)
    ↓
Output Layer (2 units, Softmax)
Input Features (15 engineered indicators)
Category	Features
Price & Trend	Lagged close, SMA_50, multi-day returns
Momentum	MACD, MACD signal, RSI_14
Volatility	Realized volatility (21d), ATR_14, Bollinger Band width
Volume	On-Balance Volume (OBV)
Training Configuration
Optimizer: Adam (learning rate = 1e-3)

Loss Function: Categorical Cross-Entropy

Batch Size: 32

Early Stopping: Patience = 8 epochs

Max Epochs: 50

Validation Split: 20%

🔧 Technical Details
Data Generation
The application uses simulated data that mirrors the statistical properties of the original thesis dataset:

5 years of daily prices (1,256 trading days)

Realistic volatility based on historical S&P 500 stocks

Sector diversification (Technology, Financials, Energy, Healthcare)

Performance Metrics
Accuracy: Correct directional predictions / Total predictions

Precision: True Positives / (True Positives + False Positives)

Recall: True Positives / (True Positives + False Negatives)

F1-Score: Harmonic mean of precision and recall

MAPE: Mean Absolute Percentage Error

RMSE: Root Mean Square Error (price-relative)

Walk-Forward Validation
The evaluation protocol ensures no look-ahead bias:

Training window: 252 trading days (≈1 year)

Test window: Equal to forecast horizon

Rolling window advances by horizon length

📁 Project Structure
python
```app.py
├── Configuration & Styling
├── Data Functions
│   ├── get_performance_data()
│   ├── get_baseline_comparison()
│   ├── get_shap_features()
│   ├── get_ticker_performance()
│   ├── get_confusion_matrices()
│   └── get_descriptive_stats()
├── Simulation Functions
│   ├── generate_price_data()
│   └── generate_forecast_plot()
└── Page Rendering
    ├── Overview Page
    ├── Model Performance Page
    ├── SHAP Analysis Page
    ├── Forecast Visualization Page
    └── Literature & Results Page
🔧 Troubleshooting
Common Issues and Solutions
Issue: ModuleNotFoundError: No module named 'streamlit'

bash
# Solution: Install missing dependencies
pip install streamlit
Issue: Port 8501 is already in use

bash
# Solution: Run on a different port
streamlit run app.py --server.port 8502
Issue: Plotly rendering errors

bash
# Solution: Update plotly
pip install --upgrade plotly
Issue: Application runs slowly

bash
# Solution: Clear Streamlit cache
streamlit cache clear
System Requirements
Minimum RAM: 4GB

Recommended RAM: 8GB

Disk Space: 500MB

Internet: Required for CDN resources (Plotly, Streamlit)

📚 Citation
If you use this application or the thesis findings in your research, please cite:

bibtex
@mastersthesis{marey2026explainable,
  author = {Nada Marey Maged Hassan},
  title = {Explainable AI with Deep Learning for Stock Prices Prediction on Time Series Big Data: An Empirical Study},
  school = {Tanta University, Faculty of Commerce},
  year = {2026},
  type = {MSc Thesis},
  address = {Tanta, Egypt}
}
```
📧 Contact
Author: Nada Marey Maged Hassan
Institution: Tanta University, Faculty of Commerce
Department: Business Information Systems
Supervisors:

Prof. Ahmad A Abu-Musa

Assoc. Prof. Sara El-Sayed El-Metwally

Assoc. Prof. Mona Atef Ganna

📄 License
This project is for academic and research purposes. The code and findings are the intellectual property of the author and Tanta University.

Terms of Use:

Free for academic research and learning

Commercial use requires permission

Proper attribution required

Not for direct financial trading decisions

🙏 Acknowledgments
This research was supported by:

Tanta University, Faculty of Commerce

Mansoura University, Faculty of Computers and Information

The academic supervisors for their guidance and feedback

⚠️ Disclaimer
IMPORTANT: This application is for demonstration and research purposes only. It should NOT be used as:

Direct investment advice

Automated trading system

Financial decision-making tool without human oversight

The model predictions are based on historical patterns and do not guarantee future performance. Always consult with qualified financial advisors before making investment decisions.


#### Built with ❤️ by Nada Marey
