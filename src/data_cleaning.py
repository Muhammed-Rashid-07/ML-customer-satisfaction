import logging
from abc import ABC,abstractmethod
import pandas as pd
from typing import Union
from sklearn.model_selection import train_test_split
import numpy as np

"""
Design Pattern used: Strategy Pattern
Components:-
1. Abstract Class - must do methods in concrete classes
2. Concrete Class(concrete strategies) - 
3. Context Class
4. Main Code
"""


#Abstract class
class DataStrategy(ABC):
    """
    Abstract class defining for handling data.
    """
    @abstractmethod
    def handle_data(self, data:pd.DataFrame) -> Union[pd.DataFrame,pd.Series]:
        """
        Args:
            pd.Dataframe: Inserting data
            yujitk
        Returns:
            pd.Dataframe , series: Either dataframe or series.
        """
        pass
    
    
#Concrete class
class DataPreProcessStrategy (DataStrategy):
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame,pd.Series]:
        try:
            #dropping unnecessary columns as of now
            data = data.drop([
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
            "order_purchase_timestamp",
            "review_comment_message"
            ],
            axis=1
            )
            data['product_weight_g'].fillna(data["product_weight_g"].median(),inplace=True)
            data['product_length_cm'].fillna(data["product_length_cm"].median(),inplace=True)
            data['product_height_cm'].fillna(data["product_height_cm"].median(),inplace=True)
            data['product_width_cm'].fillna(data["product_width_cm"].median(),inplace=True)
            
            #filtering numeric data.
            data = data.select_dtypes(include=[np.number])
            
            cols_to_drop = ['customer_zip_code_prefix','order_item_id']
            data.drop(
            cols_to_drop
            ,axis=1)
            return data
        except Exception as e:
            logging("Error in processing data: {}".format(e))
            raise e
        
        
            
class DataDivideStrategy(DataStrategy):
    """
    Strategy for dividing data into train and test

    Args:
        np.Dataframe:
    """
    
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame,pd.Series]:
        try:
            X = data.drop(['review_score'],axis=1)
            y = data['review_score']
            X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=1)
            return X_train,X_test,y_train,y_test
        
        except Exception as e:
            logging.error("Error in dividing data: {}".format(e))
            raise e
       
       
#context class 
class DataCleaning:
    """
    Context-class : the "context class" is a class that contains the main business logic and has a dependency on one or more strategies.
    class for cleaning data which preprocess the data and divides it into train and test
    """
    def __init__(self, data: pd.DataFrame, strategy:DataStrategy):
        self.data = data
        self.strategy = strategy
        
    def handle_data(self) -> Union[pd.DataFrame,pd.Series]:
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging("Error in handliong DataCleaning: {}".format(e))
            raise e
            
                 