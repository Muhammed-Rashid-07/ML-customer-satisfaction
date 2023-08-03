import logging

import pandas as pd
from zenml import step

class IngestData:
    '''
    Ingesting data from the data_path
    '''
    def __init__(self,data_path:str) -> None:
        '''
        Args:
            data_path: path to the data
        '''
        self.data_path = data_path
        
    def get_data(self):
        '''
        Ingesting the data from data_path
        Returns: the ingested data.
        '''
        logging.info(f"Ingesting data from {self.data_path}")
        return pd.read_csv(self.data_path)

@step
def ingest_df( data_path:str ) -> pd.DataFrame:
    '''
    Ingesting data from the datapath,
    Args: 
        data path: path to the data.
        
    Returns:
        pd.Dataframe:the ingested data. 
    '''
    try:
        ingest_data = IngestData(data_path)
        df = ingest_data.get_data()
        logging.info("Ingesting data completed.")
        return df
    except Exception as e:
        logging.error("Error while ingesting data")
        raise e            
        
    