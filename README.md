# Random-Forest-Model-applied-to-the-Stock-Market

THis project includes code that was developed from scratch to make Random Forest models (RFM), including a Classifier with replacement spliting criteria (Gini or Entropy) and a Regressor model for continuous price evaluation. 

The structure of the files are as follows: \
\\
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
*__Real-Time Data__: Fetch and then visualise stock/forex data from Alpha Vantage API.
