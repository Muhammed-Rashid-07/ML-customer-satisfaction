import logging
from zenml import step
import pandas as pd


@step
def clean_df(df: pd.DataFrame) -> None:
    '''
        Cleaning the ingested data
        
        Args:
            df: Ingested data
        Returns: 
            pd.DataFrame: cleaned data.
    '''
    pass

