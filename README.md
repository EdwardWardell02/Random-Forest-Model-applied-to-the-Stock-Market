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
* __Real-Time Data__: Fetch and then visualise stock/forex data from Alpha Vantage API.
One can select what stock to look at, either SP500 or Forex, the time span to look over, and the specific ticker.
---
### Market Data Analysis
![Ticker](Stock_model/images/Front_Page_stock.PNG)
* __Technical Indicators__: Automatic calculation of MA20, MA50, volatility and, daily returns.
* __Interactive Charts__: Candlestick charts with zoom and pan functionality.
---
### Machine Learning Capabilities
![ML](Stock_model/images/ML_Model_Page.PNG)

