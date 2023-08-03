from zenml import pipeline
from steps.ingest_data import ingest_df
from steps.evaluation import evaluate_model
from steps.clean_data import clean_df
from steps.model_train import train_model

@pipeline(enable_cache=False)
def train_pipeline(data_path: str):
    '''
    Data pipeline for training the model.

    Args:
        data_path: The path to the data to be ingested.
    '''
    # Ingest data using the ingest_df step
    df = ingest_df(data_path=data_path)
    
    # Cleaned data by splitting training and testing
    X_train, X_test, y_train, y_test = clean_df(df=df)
    
    #Training the model
    model = train_model(X_train=X_train,y_train=y_train)
    
    #Evaluating the model
    r2_score, rmse = evaluate_model(model=model,X_test=X_test,y_test=y_test)
    