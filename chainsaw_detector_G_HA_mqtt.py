#!/usr/bin/env python3
"""
UNIVERSAL Chainsaw Detector Node – Multi-node ready
Just change NODE_ID = X on each Raspberry Pi → everything else auto-configures
Creates entity: sensor.chainsaw_detector_node_X_chainsaw_detector_X
Tested & working on HAOS 2025.11+ with Mosquitto add-on inside HAOS
"""
import tensorflow as tf, numpy as np, sounddevice as sd, time, csv, json, signal, sys
from scipy import signal as sp_signal
import paho.mqtt.client as mqtt

# =================== ONLY CHANGE THIS PER NODE ===================
NODE_ID = 1                                               # ←←← CHANGE THIS ON EACH PI (1, 2, 3, ...)
# =================================================================

NODE_STR = f"node_{NODE_ID}"                  # → node_1, node_2, etc.
DEVICE_NAME = f"Chainsaw Detector Node {NODE_ID}"

MQTT_BROKER = "192.168.1.35"                  # ← Your HAOS IP (Mosquitto add-on)
MQTT_PORT   = 1883
MQTT_USER   = "forest"
MQTT_PASS   = "forest123"

# === AUTO-GENERATED TOPICS (creates exact entity you want) ===
DISCOVERY_TOPIC = f"homeassistant/sensor/chainsaw_detector_{NODE_STR}/chainsaw_detector_{NODE_ID}/config"
DISCOVERY_STATUS = f"homeassistant/sensor/chainsaw_detector_{NODE_ID}/status/config"
STATE_TOPIC     = f"chainsaw_detector/{NODE_STR}/state"
UNIQUE_ID       = f"chainsaw_detector_{NODE_STR}_chainsaw_detector_{NODE_ID}"
#device info added
device_info = {
    "name": DEVICE_NAME,
    "identifiers": [f"chainsaw_detector_{NODE_STR}"],
    "manufacturer": "Pakistan Forest Project",
    "model": "RPi Acoustic Node",
    "sw_version": "3.0-universal"
}

#newly added status topic
STATUS_TOPIC     = f"chainsaw_detector/{NODE_STR}/status"

SAMPLE_RATE = 16000
BUFFER_SEC  = 0.96
DEVICE      = 0                               # Change if needed per node
BLOCKSIZE   = int(SAMPLE_RATE * 0.06)

THRESHOLD           = 0.15
MIN_HA_CONFIDENCE   = 60.0
EMA_ALPHA           = 0.82
DECAY_ALPHA         = 0.15
MIN_CONFIRM_FRAMES  = 2
SILENCE_RMS         = 0.018
first_publish = True
# =================== YAMNET LOAD ===================
print(f"Starting {DEVICE_NAME} – loading YAMNet...")
model = tf.saved_model.load("yamnet_standalone")
infer = model.signatures["serving_default"]
with open("yamnet_class_map.csv") as f:
    reader = csv.reader(f)
    next(reader)
    chainsaw_idx = next(i for i, row in enumerate(reader) if "chainsaw" in row[2].lower())

# =================== FILTERS ===================
nyq = SAMPLE_RATE / 2
b_band, a_band = sp_signal.butter(7, [195/nyq, 6200/nyq], btype='band')
b_notch, a_notch = sp_signal.iirnotch(50.0 / nyq, 45.0)
b_hp, a_hp = sp_signal.butter(7, 155 / nyq, btype='high')

# =================== STATE ===================
buffer = np.zeros(int(SAMPLE_RATE * BUFFER_SEC), dtype=np.float32)
buffer_ptr = 0
chainsaw_ema = 0.0
confirm_count = 0
discovery_sent = False
last_zero_sent = time.time()
last_was_detected = False  # ← NEW: Track if last loop was detected

# =================== MQTT HELPERS ===================
def mqtt_connect():
    c = mqtt.Client()
    c.username_pw_set(MQTT_USER, MQTT_PASS)
    c.connect(MQTT_BROKER, MQTT_PORT, 60)
    return c

def send_discovery():
    global discovery_sent
    if discovery_sent: return
    payload = {
        "name": f"Chainsaw Detector {NODE_ID}",
        "unique_id": UNIQUE_ID,
        "state_topic": STATE_TOPIC,
        "json_attributes_topic": STATE_TOPIC,
        "value_template": "{{ value_json.confidence_ha | round(1) }}",
        "unit_of_measurement": "%",
        "state_class":"measurement",
        "icon": "mdi:saw-blade",
        "device": device_info
    }

    
    c = mqtt_connect()
    c.publish(DISCOVERY_TOPIC, json.dumps(payload), retain=True)
    #chainsaw status discovery 
    c.publish(DISCOVERY_STATUS, json.dumps({
        "name": f"Chainsaw Status Node {NODE_ID}",
        "unique_id": f"chainsaw_status_node_{NODE_ID}",
        "state_topic": STATUS_TOPIC,
        "value_template": "{{ value_json.status }}",
        "entity_category": "diagnostic",
        "device": device_info
    }), retain=True)
    
    c.disconnect()
    discovery_sent = True
    print(f"[Node {NODE_ID}] Discovery sent → sensor.chainsaw_detector_{NODE_STR}_chainsaw_detector_{NODE_ID}")

def publish_state(real_conf_pct: float, detected: bool):
    global last_zero_sent, last_was_detected
    ha_conf = max(MIN_HA_CONFIDENCE, real_conf_pct) if detected else 0.0
    ha_conf = round(ha_conf, 1)

    # Always publish if detected
    if detected:
        last_was_detected = True
    # If not detected...
    else:
        if last_was_detected:
            # Transition from detected → quiet: Send 0% IMMEDIATELY to reset HA
            last_was_detected = False
            last_zero_sent = time.time()  # Reset timer
        else:
            # Already quiet: Send only once/hour
            if time.time() - last_zero_sent < 3600:
                return
            last_zero_sent = time.time()

    payload = {
        "confidence_ha": ha_conf,
        "confidence_real": round(real_conf_pct, 2),
        "confidence_internal": round(chainsaw_ema, 4),
        "status": "chainsaw" if detected else "quiet",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z")
    }
    c = mqtt_connect()
    c.publish(STATE_TOPIC, json.dumps(payload),retain=True)
    #chainsaw status 
    c.publish(STATUS_TOPIC, json.dumps({
    "status": "chainsaw" if detected else "quiet",
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z")
      }), retain=True)
    
    c.disconnect()

    status = "CHAINSAW!!!" if detected else "quiet"
    print(f"[Node {NODE_ID}] {status} → HA:{ha_conf}%  Real:{real_conf_pct:.2f}%")

# =================== AUDIO CALLBACK ===================
def audio_callback(indata, frames, time_info, status):
    global buffer, buffer_ptr
    if status: return
    mono = indata[:, 0].astype(np.float32)
    mono = sp_signal.lfilter(b_band, a_band, mono)
    mono = sp_signal.lfilter(b_notch, a_notch, mono)
    mono = sp_signal.lfilter(b_hp, a_hp, mono)
    rms = np.sqrt(np.mean(mono**2) + 1e-12)
    if rms < 0.05:
        mono *= min(0.48 / rms, 22.0)
    end = buffer_ptr + len(mono)
    if end <= len(buffer):
        buffer[buffer_ptr:end] = mono
    else:
        split = len(buffer) - buffer_ptr
        buffer[buffer_ptr:] = mono[:split]
        buffer[:end % len(buffer)] = mono[split:]
    buffer_ptr = end % len(buffer)

# =================== MAIN ===================
send_discovery()
#publish_state(12.3, True)
#time.sleep(1)
#publish_state(0.0, False)
stream = sd.InputStream(samplerate=SAMPLE_RATE, device=DEVICE, channels=1,
                        callback=audio_callback, blocksize=BLOCKSIZE, dtype='float32')
stream.start()

last_process = time.time()
try:
    while True:
        if time.time() - last_process >= 0.5:
            last_process = time.time()
            audio = np.copy(buffer)
            rms = np.sqrt(np.mean(audio**2))
            if rms < SILENCE_RMS:
                chainsaw_ema *= 0.9
                confirm_count = 0
                publish_state(0.0, False)
            else:
                scores = infer(waveform=audio)["output_0"].numpy()
                raw = float(np.max(scores[-4:, chainsaw_idx]))
                chainsaw_ema = EMA_ALPHA * raw + (1-EMA_ALPHA)*chainsaw_ema if raw > chainsaw_ema \
                              else DECAY_ALPHA * raw + (1-DECAY_ALPHA)*chainsaw_ema
                if chainsaw_ema >= THRESHOLD:
                    confirm_count = min(confirm_count + 1, 20)  # ← NEW: Higher cap for smoother decay
                else:
                    confirm_count = max(confirm_count - 1, 0)
                publish_state(chainsaw_ema * 100, confirm_count >= MIN_CONFIRM_FRAMES)
        time.sleep(0.01)
except KeyboardInterrupt:
    print(f"\n[Node {NODE_ID}] Stopped.")
finally:
    stream.stop(); stream.close()
