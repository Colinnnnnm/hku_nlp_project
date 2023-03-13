# hku_nlp_project

## data_into_DFv2.py: 
- Class of DataParser that process the tweet and stock price data

## Tool.py:
- Main function for creating train , test data, model
- ETL: subclass of DataParser, with new method to create the time series data for training , validation, testing
- BertPlusTechModel_Classify: A nerual network that accept Text and Price data, output whether the stock price will go up or down

## Ipynb script:
- ModelTrain.ipynb: Notebook for training
- Evalution.ipynb: Notebook for Evaluting the model performance