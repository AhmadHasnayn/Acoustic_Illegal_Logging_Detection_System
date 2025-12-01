# Acoustic_Illegal_Logging_Detection_System
A wood cutting detection system uses a combination of sensors and AI to identify unauthorized tree felling, typically by detecting the sounds and vibrations of chainsaws and using IoT (Internet of Things) technology to alert authorities. These systems can be integrated with other technologies like drones.

 # Description:
 A small project that uses YAMNet embeddings and a custom classifier to detect chainsaw / wood-cutting events from a microphone in realtime. It also includes utilities to generate embeddings from WAV files (`autoscript.py`)). The realtime detector publishes detection results (status + confidence) to an MQTT broker.

# Repository Structure
- `realtime_yamnet.py`: Realtime detector using YAMNet embeddings + custom classifier; records audio with `sounddevice` and publishes predictions to MQTT.
- `autoscript.py`: Offline script to extract YAMNet embeddings from WAV files and build a dataset (`chainsaw_dataset.npz`) for training a classifier.
- chainsaw_yamnet_classifier_model.h5`: (Not included) This is the trained Keras classifier expected by `realtime_yamnet.py`.
- `yamnet_local_dir/`: (Not included) Optional local SavedModel copy of YAMNet if you cannot download from TF‑Hub at runtime.

# Quick Overview
- `realtime_yamnet.py` records short audio (default 1 second), runs YAMNet to get embeddings, averages/reshapes embeddings, runs a Keras classifier, then publishes the prediction to an MQTT topic.
- `autoscript.py` extracts embeddings frame-by-frame from WAV files placed in `chainsaw_audio/` and `non_chainsaw_audio/` and saves `chainsaw_dataset.npz` (arrays `X` and `y`) for training.

# Prerequisites
- Python 3.8 – 3.11 (match TensorFlow compatibility)
- A working microphone (USB or built-in) recognized by Windows
- Optional: MQTT broker reachable from your machine (local broker IP or cloud)
- If offline: a local YAMNet SavedModel in `yamnet_local_dir/` (see "YAMNet / Model" below)

# Recommended setup (Windows PowerShell)
1. Create and activate a venv
```powershell
Set-Location -Path 'C:\Users\HP\.vscode\chainsaw-detector'
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
2. Install dependencies
```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

YAMNet Model
- The scripts expect YAMNet embeddings for feature extraction. There are two ways to get YAMNet:
  - Online from TF‑Hub: use the handle `https://tfhub.dev/google/yamnet/1` (this requires internet access and `tensorflow_hub` to download the model at runtime).
  - Offline SavedModel: download YAMNet as a SavedModel folder and place it under `yamnet_local_dir/`. Then `realtime_yamnet.py` can load it with `hub.load('./yamnet_local_dir')`.
- If you do not have the SavedModel locally, the scripts will attempt to download YAMNet from TF‑Hub unless they are explicitly configured to load from a local path.

YAMNet TF‑Hub reference: https://tfhub.dev/google/yamnet/1

# Train a simple classifier (example)
- A minimal Keras classifier training flow (not included in repo) using `chainsaw_dataset.npz`:
Run realtime detector (`realtime_yamnet.py`)**
- By default the script loads YAMNet from `./yamnet_local_dir` (offline) — update `LOCAL_MODEL_PATH` if needed.
- Ensure you have `chainsaw_yamnet_classifier_model.h5` in the repository root.
- Example run command (use venv Python):
```powershell
python .\realtime_yamnet.py
- MQTT publishes are sent to the broker configured in `MQTT_BROKER` and `MQTT_TOPIC`.
