# Random-Forest-Model-applied-to-the-Stock-Market

THis project includes code that was developed from scratch to make Random Forest models (RFM), including a Classifier with replacement spliting criteria (Gini or Entropy) and a Regressor model for continuous price evaluation. 

The structure of the files are as follows: \
\
Stock_Model/ \
├── app/ \
│   ├── data_loader.py \
│   ├── data_processor.py \
│   ├── update_visualiser.py \
│   ├── Random_Forest_Model_Classifier.py \
│   └── Random_Forest_Model_Regressor.py \
├── data/ \
│   └── (cached data files from Alpha Vantage) \
├── images/ \
│   ├── dashboard-overview.png \
│   ├── market-data-tab.png \
│   └── (other screenshots) \
├── main.py \
├── requirements.txt \
├── README.md \

---
## The Project
Interactive financial analysis dashboard featuring custom Random Forest models (RFM) for stock &amp; forex prediction. Compares performance of personally made RFM against sklearn with visualisations, technical indicators, and Alpha Vantage integration, along with Analysis of results of classifier or regressor models.
---

---
### Main Dashboard
![Main Dashboard](Stock_model/images/Front_Page.PNG)
#### Figure 1
* __Real-Time Data__: Fetch and then visualise stock/forex data from Alpha Vantage API.
One can select what stock to look at, either SP500 or Forex, the time span to look over, and the specific ticker.
---
### Market Data Analysis
![Ticker](Stock_model/images/Front_Page_stock.PNG)
#### Figure 2
* __Technical Indicators__: Automatic calculation of MA20, MA50, volatility and, daily returns.
* __Interactive Charts__: Candlestick charts with zoom and pan functionality.
---
### Machine Learning Capabilities
#### Select the next Tab to see the ML page:
Here the chosen model is trained on the data selected by the user.
* __Dual Model Types__: Classification (price direction) and Regression (price value).
* __Feature Selection__: Choose from technical indicators and price data.(see Figure 4)
* __Feature Evaluation__: Correlation and scatter matrices are displayed for selected features. (see Figure 5 & 6)
* __Hyperparameter Tuning__: Adjust model parameters through sliders. (see Figure 7)
The window slider is because the RFM work through bootstrapping, as this is a forecasting problem, the base learners are trained individually on windows of the dataset instead of random sampling. (see Figure 8)
![ML](Stock_model/images/ML_Model_Page.PNG)
#### Figure 3
![ML](Stock_model/images/ML_Model_Page_Features.PNG)
#### Figure 4
![ML](Stock_model/images/ML_Model_Page_Correlation_Matrix.PNG)
#### Figure 5
![ML](Stock_model/images/ML_Model_Page_Scatter_Matrix.PNG)
#### Figure 6
![ML](Stock_model/images/ML_Model_Page_Training.PNG)
#### Figure 7
![ML](Stock_model/images/ML_Model_Page_Bootstrapping.PNG)
#### Figure 8
---
### Perfomance of RFM
