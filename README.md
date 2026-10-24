
Deep Learning Anomaly Detection System

This project builds an AI watchdog that monitors machine data and automatically sounds the alarm if something breaks or behaves unusually.

🎯 How It Works

Instead of teaching the AI what every possible error looks like, it uses a deep learning model called an Autoencoder to learn what perfect, healthy behavior looks like.
• Step 1: Memorizing Normal: The AI trains exclusively on healthy data until it knows the normal patterns by heart.
• Step 2: Testing New Data: When new machine readings come in, the AI tries to copy and recreate them.
• Step 3: Catching Anomalies: If the incoming data is broken or glitchy, the AI will fail to copy it correctly. This recreation error triggers a high statistical score, instantly flagging an anomaly.

📁 Project Files

• config.py: Holds project settings (like training speed and network sizes) in one central place.
• data_loader.py: Cleans the incoming data and prepares it for the AI model.
• model.py: Defines the shape and structure of the Neural Network layers.
• train.py: The main script that runs the data, trains the AI, and outputs the final detection scores.
• requirements.txt: A list of libraries needed to run the project.
