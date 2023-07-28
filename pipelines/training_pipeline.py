from zenml import pipeline
from steps.ingest_data import ingest_df
from steps.evaluation import evaluate_model
from steps.clean_data import clean_df
from steps.model_train import train_model

@pipeline
def train_pipeline(data_path: str):
    '''
    Data pipeline for training the model.

    Args:
        data_path: The path to the data to be ingested.
    '''
    # Ingest data using the ingest_df step
    df = ingest_df(data_path=data_path)
    
    # Perform data cleaning using the clean_df step
    df = clean_df(df)
    
    # Train the model using the train_model step
    train_model(df)
    
    # Evaluate the model using the evaluate_model step
    evaluate_model(df)
