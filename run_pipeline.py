from pipelines.training_pipeline import train_pipeline
from zenml.client import Client

if __name__ == '__main__':
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    train_pipeline(data_path='/mnt/c/Users/MUHAMMED RASHID/desktop/mlops2/Data/olist_customers_dataset.csv')
