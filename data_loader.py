import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import config

def generate_and_scale_data():
    np.random.seed(config.RANDOM_STATE)
    
    normal_telementry = np.random.normal(loc=0.0, scale=1.0, size=(1000,config.INPUT_DIM))
    anomalous_events = np.random.normal(loc=3.0, scale=2.0, size=(100, config.INPUT_DIM))
    
    
    X_train, X_test = train_test_split(normal_telementry, test_size=config.VALIDATION_SPLIT, random_state=config.RANDOM_STATE)
    
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    X_test_scaled = scaler.transform(X_test)
    X_anomaly_scaled = scaler.transform(anomalous_events)
    
    
    return X_train_scaled, X_train_scaled, X_anomaly_scaled

    