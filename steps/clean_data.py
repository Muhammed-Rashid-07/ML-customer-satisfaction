import logging
from zenml import step
import pandas as pd
from src.data_cleaning import DataCleaning,DataDivideStrategy,DataPreProcessStrategy
from typing import Tuple
from typing_extensions import Annotated

@step
def clean_df(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame,"X_train"],
    Annotated[pd.DataFrame,"X_test"],
    Annotated[pd.Series,"y_train"],
    Annotated[pd.Series,"y_test"],
]:
    '''
        Cleaning the ingested data
        
        Args:
            df: Ingested data
        Returns: 
            X_train = training data
            X_test = testing data
            y_train = training data
            y_test = testing data
    '''
    try:
        #Data cleaning
        process_strategy = DataPreProcessStrategy()
        print(process_strategy)
        data_cleaning = DataCleaning(df,process_strategy)
        processed_data = data_cleaning.handle_data()
        
        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(processed_data,divide_strategy)
        X_train,X_test,y_train,y_test = data_cleaning.handle_data()
        logging.info("Logging data completed")
        return X_train,X_test,y_train,y_test
        
    except Exception as e:
        logging.error("Error in clean_data {}".format(e))
        raise e
        
        



