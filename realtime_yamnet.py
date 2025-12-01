import numpy as np
import sounddevice as sd
import tensorflow as tf
import tensorflow_hub as hub 
import time
import json
import paho.mqtt.client as mqtt

# ================= CONFIG =================
SAMPLE_RATE = 16000
DURATION = 1.0
MQTT_BROKER = "172.100.100.100"
#MQTT_BROKER = "192.168.1.36"
MQTT_TOPIC = "forest/wood_cutting_detection_node1"

# Define the local path where you saved the model
LOCAL_MODEL_PATH = "./yamnet_local_dir" 

# ================= LOAD MODELS =================
print(f"Loading YAMNet from local directory: {LOCAL_MODEL_PATH} (Offline)")

# Load the model directly from the local path
try:
    yamnet = hub.load(LOCAL_MODEL_PATH)
except Exception as e:
    print(f"Failed to load YAMNet model from local path {LOCAL_MODEL_PATH}.")
    print(f"Error details: {e}")
    print("Please ensure the 'yamnet_local_dir' folder exists and contains the SavedModel files.")
    raise

print("Loading classifier...")
# Your custom classifier H5 file remains the same
classifier = tf.keras.models.load_model("chainsaw_yamnet_classifier_model.h5")

# ================= MQTT =================
mqtt_client = mqtt.Client()
# This part assumes your local MQTT broker is available on the network 
mqtt_client.connect(MQTT_BROKER, 1883, 60)

# ================= AUDIO LOOP =================
print("Starting Chainsaw Detector with YAMNet Embeddings...")

while True:
    # Record audio
    audio = sd.rec(int(SAMPLE_RATE * DURATION),
                   samplerate=SAMPLE_RATE,
                   channels=1,
                   dtype='float32')
    sd.wait()
    audio = np.squeeze(audio)

    # Run YAMNet to get embeddings
    wave = tf.convert_to_tensor(audio, dtype=tf.float32)
    # It returns scores, embeddings, and log_mel_spectrogram
    scores, embeddings, spectrogram = yamnet(wave)

    # Average all 1-second embeddings
    embedding = np.mean(embeddings.numpy(), axis=0)
    embedding = embedding.reshape(1, 1024)

    # Run your classifier
    pred = classifier.predict(embedding, verbose=0)[0]
    idx = np.argmax(pred)
    classes = ["non-chainsaw", "chainsaw"]
    predicted_class = classes[idx]
    confidence = float(pred[idx])

    print(f"Prediction: {predicted_class} ({confidence*100:.1f}%)")

    # Publish
    mqtt_client.publish(
        MQTT_TOPIC,
        json.dumps({"status": predicted_class, "confidence": confidence})
    )

    time.sleep(0.2)
