import logging
from abc import ABC,abstractmethod
from sklearn.linear_model import LinearRegression

class Model(ABC):
    """
    Abstract method

    Args:
        ABC (_type_): _description_
    """
    @abstractmethod
    def train(self, X_train,y_train):
        """
            Trains the model
        Args:
            X_train : training data
            y_train : training labels
        """
        pass
    
    
#Concrete class  
class LinearRegressionModel(Model):
    """Linear regression model

    Args:
        Model (_type_): _description_
    """
    def __init__(self):
        self.model = LinearRegression()
        
    def train(self, X_train, y_train, **kwargs):
        """Training the model

        Args:
            X_train (_type_): _description_
            y_train (_type_): _description_
        """
        try:
            reg = LinearRegression(**kwargs)
            reg.fit(X_train,y_train)
            logging.info("Model training completed")
            return reg
        except Exception as e:
            logging.error("Error in training model {}".format(e))
            raise e