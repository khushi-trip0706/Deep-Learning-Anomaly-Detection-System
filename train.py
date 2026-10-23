import numpy as np
import data_loader
import model
import config

def run_pipeline():
    X_train,X_test,X_anomaly = data_loader.generate_and_scale_data()
    
    nn_framework = model.build_autoencoder()
    
    print("Executing Deep Autoencoder Training....")
    nn_framework.fit(
        X_train,X_train,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        validation_data = (X_test,X_test),
        verbose = 1
    )
    
    reconstruction = nn_framework.predict(X_train)
    train_loss = np.mean(np.abs(reconstruction - X_train),axis=1)
    anomaly_threshold = np.mean(train_loss)+(2*np.std(train_loss))
    print(f"\n[SYSTEM DETECTOR LOG] Baseline Anomaly Threshold: {anomaly_threshold:.4f}")
    
    anomaly_preds = nn_framework.predict(X_anomaly)
    anomaly_loss = np.mean(np.abs(anomaly_preds - X_anomaly), axis=1)
    flagged = anomaly_loss > anomaly_threshold
    
    detection_rate = (np.sum(flagged)/len(flagged))*100
    print(f"[SYSTEM DETECTOR LOG] Successfully Flagged) {detection_rate:.1f}% of anomalous variants.")
    
if __name__ == "__main__":
    run_pipeline()