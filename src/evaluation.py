import logging 
from abc import ABC,abstractmethod
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score,mean_squared_error

class Evaluation(ABC):
    """Abstract class for evaluating model

    Args:
        ABC (_type_): _description_
    """
    @abstractmethod
    def calculate_scores(self,y_true:np.ndarray,y_pred:np.ndarray):
        """
        Calculate the score for the model
        Args:
            y_true: True labels
            y_pred: Predicted labels
        """
        pass
    
class MSE(Evaluation):
    """
    Evaluation strategy that uses mean squaured error.
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating MSE")
            mse = mean_squared_error(y_pred=y_pred,y_true=y_true)
            logging.info("MSE: {}".format(mse))
            return mse
        except Exception as e:
            logging.error("Error in calculating MSE, {}".format(e))
            raise e
        
class R2(Evaluation):
    """Evaluating strategy: Rsquare
    Args:
        Evaluation (_type_): _description_
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Calculating Rsquare scores

        Args:
            y_true (ndarray): true labels
            y_pred (ndarray): predictions

        Returns:
            float: rmse
        """
        try:
            logging.info("Calculating Rsquare")
            r2 = r2_score(y_pred=y_pred,y_true=y_true)
            logging.info("R2: {}".format(r2))
            return r2
        except Exception as e:
            logging.error("Error in calculating R2: ".format(r2))
            
            
class RMSE(Evaluation):
    """Calculating RMSE
    """
    
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Calculating RMSE scores

        Args:
            y_true (ndarray): true labels
            y_pred (ndarray): predictions

        Returns:
            float: rmse
        """
        def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
            try:
                logging.info("Calculating MSE")
                rmse = mean_squared_error(y_pred=y_pred,y_true=y_true,squared=False)
                logging.info("MSE: {}".format(rmse))
                return rmse
            except Exception as e:
                logging.error("Error in calculating MSE, {}".format(e))
                raise e