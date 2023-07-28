import logging
from zenml import step
import pandas as pd


@step

def evaluate_model(df:pd.DataFrame) -> None:
    '''
    Evaluating the model.
    
    Args:
        The testing data.
    '''
    pass