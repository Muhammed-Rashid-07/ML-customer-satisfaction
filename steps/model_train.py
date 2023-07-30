import logging
import pandas as pd
from zenml import step 
from src.model_dev import LinearRegressionModel
from steps.config import ModelNameConfig
from sklearn.base import RegressorMixin




@step
def train_model(
    X_train:pd.DataFrame,
    y_train:pd.DataFrame,
    config: ModelNameConfig,
    ) -> RegressorMixin:
    '''
    Train the data
    
    Args: 
        df: The ingested data
    Returns:
        trained model : RegressorMixin
    '''
    try:
        model = None
        if config.model_name == "LinearRegression":
            model = LinearRegressionModel()
            trained_model = model.train(X_train,y_train)
            return trained_model
        else:
            raise ValueError("Model {} not supperted".format(config.model_name))
    except Exception as e:
        logging.info("Error in training model {}".format(e))
        raise e