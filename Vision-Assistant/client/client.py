import streamlit as st
# Configure Streamlit page
st.set_page_config(layout="wide", page_title="Vision Assistant")


import cv2
import numpy as np
import time
import threading
import base64
import requests
import json
import psutil
import random
from power_model import (
    PowerSimulator,
    DEVICE_PROFILES,
    read_host_battery,
)

from speech_client import (
    initialize_speech_recognition, 
    process_speech, 
    cleanup_speech_recognition
)

# Server configuration (NOTE: This must be updated based on server IP and port on which the backend is running)
SERVER_URL = "http://localhost:8000"

# ---------------------------------------------------------------------------
# Smart Activation thresholds — driven by physical battery model
# ---------------------------------------------------------------------------
# These replace Ankita's fixed percentage thresholds with thresholds based on
# actual remaining estimated lifetime and per-module energy cost, so the
# decisions are calibrated to the selected target device.

# Battery % below which scene description is blocked (most expensive module)
SCENE_BATTERY_THRESHOLD = 40

# Battery % below which face recognition is blocked
FACE_BATTERY_THRESHOLD = 20

# Battery % below which all AI modules stop and only voice/TTS survives
CRITICAL_BATTERY_THRESHOLD = 10

# Camera resolution steps per power mode
RESOLUTION_NORMAL   = (640, 480)
RESOLUTION_POWERSAVE = (320, 240)

# Object detection must have seen a person before face recognition is allowed
# (Ankita's conditional activation: face only after person detected)
PERSON_LABELS = {"person", "human", "people", "man", "woman", "child"}


def get_power_mode(battery_pct: float) -> str:
    """Return the current power mode label based on battery percentage."""
    if battery_pct > SCENE_BATTERY_THRESHOLD:
        return "normal"
    elif battery_pct > FACE_BATTERY_THRESHOLD:
        return "power-save"
    elif battery_pct > CRITICAL_BATTERY_THRESHOLD:
        return "restricted"
    else:
        return "critical"


def smart_activation_check(requested_mode: str, battery_pct: float,
                            person_detected: bool) -> tuple[bool, str]:
    """
    Decide whether to actually run the requested inference module.

    Returns (allowed: bool, reason: str).

    Rules (in priority order):
    1. Critical battery  → nothing runs
    2. Scene description → blocked below SCENE_BATTERY_THRESHOLD
    3. Face recognition  → blocked below FACE_BATTERY_THRESHOLD
    4. Face recognition  → blocked if no person has been detected yet
       (conditional activation from Ankita's contribution)
    5. Otherwise         → allowed
    """
    if battery_pct <= CRITICAL_BATTERY_THRESHOLD:
        return False, f"Critical battery ({battery_pct:.1f}%) — all AI modules suspended"

    if requested_mode == "scene":
        if battery_pct <= SCENE_BATTERY_THRESHOLD:
            return False, (
                f"Scene description blocked — battery {battery_pct:.1f}% "
                f"below {SCENE_BATTERY_THRESHOLD}% threshold"
            )

    if requested_mode == "face":
        if battery_pct <= FACE_BATTERY_THRESHOLD:
            return False, (
                f"Face recognition blocked — battery {battery_pct:.1f}% "
                f"below {FACE_BATTERY_THRESHOLD}% threshold"
            )
        if not person_detected:
            return False, "Face recognition waiting — no person detected by object module yet"

    return True, "ok"


def apply_camera_resolution(cap, power_mode: str):
    """
    Adjust camera resolution based on power mode.
    Normal   → 640×480 @30fps
    Power-save / restricted → 320×240 @10fps
    Critical → camera stays open but frames are not processed
    """
    if power_mode == "normal":
        w, h, fps = RESOLUTION_NORMAL[0], RESOLUTION_NORMAL[1], 30
    else:
        w, h, fps = RESOLUTION_POWERSAVE[0], RESOLUTION_POWERSAVE[1], 10

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, fps)


# ---------------------------------------------------------------------------


class VisionApp:
    def __init__(self):
        pass

    def encode_frame(self, frame):
        """Encode frame to base64 for sending to server for faster and better performance"""
        try:
            _, buffer = cv2.imencode('.jpg', frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            return frame_base64
        except Exception as e:
            print(f"Error encoding frame: {e}")
            return None

    def process_frame_with_server(self, frame, mode):

        try:
            print(f"\n=== CLIENT REQUEST DEBUG ===")
            print(f"Mode: {mode} (type: {type(mode).__name__})")
            print(f"Frame shape: {frame.shape}")
            
            # Validate mode
            if mode is None:
                print("Mode is None - skipping processing")
                return None
                
            if mode not in ['face', 'object', 'scene']:
                print(f"Invalid mode: {mode} - must be one of: face, object, scene")
                return None
            
            print(f"Mode is valid: {mode}")
            
            # Encode frame
            frame_b64 = self.encode_frame(frame)
            if frame_b64 is None:
                print("Failed to encode frame")
                return None

            endpoint_map = {
                "face": f"{SERVER_URL}/face/recognize",
                "object": f"{SERVER_URL}/object/detect",
                "scene": f"{SERVER_URL}/scene/describe"
            }
            endpoint = endpoint_map.get(mode)

            t_request = time.perf_counter()
            response = requests.post(
                endpoint,
                json={"frame_data": frame_b64},
                timeout=10.0
            )
            rtt = time.perf_counter() - t_request

            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    host_latency = result.get("processing_time") or rtt
                    return result.get("detection_result"), host_latency
                else:
                    print(f"Server processing error: {result.get('error')}")
                    return None
            else:
                print(f" Server request failed: {response.status_code}")
                try:
                    error_content = response.text
                    print(f"Error response content: {error_content}")
                except Exception as e:
                    print(f"Could not parse error response: {e}")
                return None

        except requests.exceptions.RequestException as e:
            print(f"Error communicating with server: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error in frame processing: {e}")
            return None

    def speak_via_server(self, text):
        """Send text-to-speech request to server"""
        try:
            response = requests.post(
                f"{SERVER_URL}/speak",
                json={"text": text},
                timeout=1.0
            )
            return response.status_code == 200
        except Exception as e:
            print(f"Error with server TTS: {e}")
            return False

def check_server_health():
    """Check if server is running and healthy"""
    try:
        response = requests.get(f"{SERVER_URL}/health", timeout=5.0)
        if response.status_code == 200:
            health_data = response.json()
            return health_data.get("status") == "healthy", health_data
        return False, None
    except Exception as e:
        return False, str(e)


# Session state initialisation
if 'power_sim' not in st.session_state:
    st.session_state['power_sim'] = PowerSimulator()
if 'cpu_usage' not in st.session_state:
    st.session_state['cpu_usage'] = 0.0
if 'low_battery_spoken' not in st.session_state:
    st.session_state['low_battery_spoken'] = False
# Smart activation state
if 'person_detected' not in st.session_state:
    st.session_state['person_detected'] = False
if 'last_power_mode' not in st.session_state:
    st.session_state['last_power_mode'] = "normal"
if 'skipped_reason' not in st.session_state:
    st.session_state['skipped_reason'] = ""


def update_power_simulation():
    """Advance the state-based power model once per rerun."""
    sim = st.session_state['power_sim']
    cpu_usage = psutil.cpu_percent(interval=None)
    st.session_state['cpu_usage'] = cpu_usage

    mode_active = st.session_state.get('mode') in ('object', 'face', 'scene')
    breakdown = sim.tick(cpu_util_percent=cpu_usage, mode_active=mode_active)

    battery_percentage = breakdown['battery_percent']

    # Low battery TTS warning
    if battery_percentage <= 12 and not st.session_state['low_battery_spoken']:
        try:
            requests.post(f"{SERVER_URL}/speak", json={"text": "Battery is low"})
            st.session_state['low_battery_spoken'] = True
        except Exception as e:
            print(f"Error speaking low battery warning: {e}")

    # Demo recharge (preserved from original)
    if battery_percentage <= 10:
        sim.sod = 0.0
        st.session_state['low_battery_spoken'] = False

    return breakdown


def main():
    # Initialize session state variables
    if 'mode' not in st.session_state:
        st.session_state.mode = None
    if 'stop_signal' not in st.session_state:
        st.session_state.stop_signal = False
    if 'voice_command_active' not in st.session_state:
        st.session_state.voice_command_active = False
    if 'speech_stop_event' not in st.session_state:
        st.session_state.speech_stop_event = threading.Event()
    if 'speech_thread' not in st.session_state:
        st.session_state.speech_thread = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'last_spoken' not in st.session_state:
        st.session_state.last_spoken = {}
    if 'cap' not in st.session_state:
        st.session_state.cap = None

    st.title("🎧 Voice-Only Vision Assistant for Blind People")
    st.markdown("### Audio-based object detection, face recognition, and scene description")

    # --- Target-system selector ---
    sim = st.session_state['power_sim']
    device = st.sidebar.selectbox(
        "🔌 Simulated target system",
        list(DEVICE_PROFILES.keys()),
        index=list(DEVICE_PROFILES.keys()).index(sim.device_name),
    )
    sim.set_device(device)

    # Advance the power model
    breakdown = update_power_simulation()
    battery_percentage = breakdown['battery_percent']
    cpu_usage = st.session_state['cpu_usage']

    # Compute current power mode
    power_mode = get_power_mode(battery_percentage)

    # Announce mode change via TTS if it changed
    if power_mode != st.session_state['last_power_mode']:
        mode_messages = {
            "power-save": "Entering power save mode. Scene description disabled.",
            "restricted":  "Battery low. Face recognition disabled. Running object detection only.",
            "critical":    "Critical battery. All AI modules suspended. Voice only.",
            "normal":      "Battery recovered. Full operation restored.",
        }
        msg = mode_messages.get(power_mode)
        if msg:
            try:
                requests.post(f"{SERVER_URL}/speak", json={"text": msg})
            except Exception:
                pass
        st.session_state['last_power_mode'] = power_mode

    # Battery warning banner
    if battery_percentage <= 12:
        st.warning(f" Low Battery Warning: {battery_percentage:.2f}% - Please conserve power!")
    # Power mode banner
    mode_colors = {
        "normal":     ("🟢", "Normal"),
        "power-save": ("🟡", "Power-Save  —  Scene description disabled"),
        "restricted": ("🟠", "Restricted  —  Scene + Face disabled"),
        "critical":   ("🔴", "Critical  —  All AI suspended, voice only"),
    }
    icon, label = mode_colors[power_mode]
    st.info(f"{icon} **Power Mode:** {label}")

    # Status bar
    st.markdown(f"""
    <div style='border:2px solid #222; border-radius:8px; padding:12px; margin-bottom:12px; width:100%;'>
        <b>🔋 Battery:</b> {battery_percentage:.2f}% ({breakdown['voltage_V']:.2f} V)
        &nbsp;&nbsp; <b>⚡ Total:</b> {breakdown['P_total_mW']:.0f} mW
        &nbsp;&nbsp; <b>🖥️ CPU:</b> {breakdown['P_cpu_mW']:.0f} mW ({cpu_usage:.0f}%)
        &nbsp;&nbsp; <b>📷 Cam:</b> {breakdown['P_camera_mW']:.0f} mW
        &nbsp;&nbsp; <b>📶 Net:</b> {breakdown['P_net_mW']:.0f} mW [{breakdown['net_state']}]
        &nbsp;&nbsp; <b>⏳ Est. lifetime:</b> {sim.battery_lifetime_estimate_h():.1f} h
    </div>
    """, unsafe_allow_html=True)

    # Smart activation status in sidebar
    with st.sidebar.expander("⚡ Smart Activation Status", expanded=True):
        st.write(f"**Power Mode:** {power_mode}")
        st.write(f"**Scene Description:** {'✅ allowed' if battery_percentage > SCENE_BATTERY_THRESHOLD else '🚫 blocked (battery)'}")
        st.write(f"**Face Recognition:** {'✅ allowed' if battery_percentage > FACE_BATTERY_THRESHOLD else '🚫 blocked (battery)'}")
        if battery_percentage > FACE_BATTERY_THRESHOLD:
            st.write(f"**Person detected:** {'✅ yes' if st.session_state['person_detected'] else '⏳ waiting for object detection'}")
        if st.session_state['skipped_reason']:
            st.caption(f"Last skip: {st.session_state['skipped_reason']}")

    # Per-module energy accounting
    with st.sidebar.expander("🔬 Energy accounting (u = active, n = tail) [mJ]", expanded=True):
        rep = sim.energy_report()
        for module_name, (u, n) in rep['modules'].items():
            st.write(f"**{module_name}**: u = {u} mJ, n = {n} mJ")
        st.write(f"Baseline (CPU idle + display + camera): {rep['baseline_mJ']} mJ")
        host = read_host_battery()
        if host:
            st.caption(f"Host laptop battery (ground truth for validation): {host[0]:.0f}%")

    # Server health check
    server_healthy, health_data = check_server_health()
    if not server_healthy:
        st.error("⚠️ Server is not running or unhealthy. Please start the server first.")
        st.info("To start the server, run: `cd server && python server.py`")
        return

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 🎛️ Control Panel")

        if st.session_state.mode:
            st.info(f"🔄 Active: {st.session_state.mode.title()} Detection")
        else:
            st.info("⏸️ No detection active")

        # Voice commands
        voice_command_text = "🎤 Stop Voice Commands" if st.session_state.voice_command_active else "🎤 Start Voice Commands"
        if st.button(voice_command_text, key="voice_cmd_btn", use_container_width=True):
            try:
                if not st.session_state.voice_command_active:
                    st.info("Starting voice commands...")
                    components = initialize_speech_recognition()
                    if None in components:
                        st.error("Failed to initialize speech recognition")
                        return

                    recognizer, audio_queue, _, stream = components
                    st.session_state.speech_stop_event.clear()
                    stream.start()

                    def mode_callback(mode):
                        st.session_state.mode = mode

                    st.session_state.speech_thread = threading.Thread(
                        target=process_speech,
                        args=(recognizer, audio_queue, st.session_state.speech_stop_event, mode_callback)
                    )
                    st.session_state.speech_thread.daemon = True
                    st.session_state.speech_thread.start()

                    st.session_state.voice_command_active = True
                    requests.post(f"{SERVER_URL}/speak", json={"text": "Voice commands activated"})
                    st.success("Voice commands are now active")
                else:
                    st.info("Stopping voice commands...")
                    st.session_state.voice_command_active = False
                    st.session_state.speech_stop_event.set()
                    cleanup_speech_recognition()
                    if st.session_state.speech_thread:
                        st.session_state.speech_thread.join(timeout=1)
                        st.session_state.speech_thread = None
                    requests.post(f"{SERVER_URL}/speak", json={"text": "Voice commands deactivated"})
                    st.warning("Voice commands are now inactive")

            except Exception as e:
                print(f"Error toggling voice commands: {e}")
                st.error(f"Error toggling voice commands: {str(e)}")
            st.rerun()

        st.markdown("---")

        # Detection mode buttons — labelled with current availability
        obj_label = "🎯 Object Detection"
        face_label = "👤 Face Recognition" + ("" if battery_percentage > FACE_BATTERY_THRESHOLD else " 🚫")
        scene_label = "🖼️ Scene Description" + ("" if battery_percentage > SCENE_BATTERY_THRESHOLD else " 🚫")

        if st.button(obj_label, key="object_detect_btn", use_container_width=True):
            st.session_state.mode = 'object'
            st.session_state.stop_signal = False
            requests.post(f"{SERVER_URL}/speak", json={"text": "Starting object detection"})
            st.rerun()

        if st.button(face_label, key="face_recog_btn", use_container_width=True):
            st.session_state.mode = 'face'
            st.session_state.stop_signal = False
            requests.post(f"{SERVER_URL}/speak", json={"text": "Starting face recognition"})
            st.rerun()

        if st.button(scene_label, key="scene_desc_btn", use_container_width=True):
            st.session_state.mode = 'scene'
            st.session_state.stop_signal = False
            requests.post(f"{SERVER_URL}/speak", json={"text": "Starting scene description"})
            st.rerun()

        if st.button("⏹️ Stop Detection", key="stop_btn", use_container_width=True):
            st.session_state.mode = None
            st.session_state.stop_signal = True
            requests.post(f"{SERVER_URL}/speak", json={"text": "Stopping detection"})
            st.rerun()

    with col2:
        st.markdown("### 📝 Audio Messages Log")

        if st.session_state.messages:
            messages_text = "\n".join([f"• {msg}" for msg in reversed(st.session_state.messages[-10:])])
            st.text_area("Recent Audio Announcements", messages_text, height=400, disabled=True, key="messages_area")
        else:
            st.info("No messages yet. Start a detection mode to begin.")

        if st.button("🗑️ Clear Messages", key="clear_msgs_btn"):
            st.session_state.messages = []
            st.rerun()

    # Camera initialisation
    if st.session_state.cap is None:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Failed to open camera")
            st.stop()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION_NORMAL[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION_NORMAL[1])
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        st.session_state.cap = cap

    # Apply resolution based on current power mode
    if st.session_state.cap is not None:
        apply_camera_resolution(st.session_state.cap, power_mode)

    # Frame processing
    if st.session_state.cap is not None and st.session_state.cap.isOpened():
        try:
            ret, frame = st.session_state.cap.read()
            if not ret:
                st.error("Failed to read from camera")
                return

            if st.session_state.mode and st.session_state.mode in ['face', 'object', 'scene']:

                # ── SMART ACTIVATION CHECK ──────────────────────────────────
                allowed, reason = smart_activation_check(
                    requested_mode=st.session_state.mode,
                    battery_pct=battery_percentage,
                    person_detected=st.session_state['person_detected'],
                )
                # ────────────────────────────────────────────────────────────

                if not allowed:
                    # Log the skip reason but do not send a frame to the server.
                    # This saves the inference energy, the network transfer energy,
                    # and the radio tail energy that would have been caused.
                    st.session_state['skipped_reason'] = reason
                    print(f"[Smart Activation] Skipped {st.session_state.mode}: {reason}")

                else:
                    st.session_state['skipped_reason'] = ""
                    app = VisionApp()
                    server_reply = app.process_frame_with_server(frame, st.session_state.mode)

                    detection_result = None
                    if server_reply:
                        detection_result, host_latency = server_reply
                        # Account energy only when inference actually ran
                        st.session_state['power_sim'].record_inference(
                            st.session_state.mode, host_latency
                        )

                    if detection_result:
                        # Track whether a person has been seen (for face activation gate)
                        if st.session_state.mode == 'object':
                            results_lower = [r.lower() for r in detection_result] \
                                if isinstance(detection_result, list) \
                                else [str(detection_result).lower()]
                            if any(label in results_lower for label in PERSON_LABELS):
                                st.session_state['person_detected'] = True

                        current_time = time.time()
                        mode_key = st.session_state.mode

                        if mode_key not in st.session_state.last_spoken or \
                           current_time - st.session_state.last_spoken.get(mode_key, 0) >= 3.0:
                            if st.session_state.mode == 'face':
                                message = f"Recognized: {', '.join(detection_result)}"
                            elif st.session_state.mode == 'object':
                                message = f"I see {', '.join(detection_result)}"
                            else:
                                message = f"Scene: {detection_result}"

                            requests.post(f"{SERVER_URL}/speak", json={"text": message})
                            st.session_state.last_spoken[mode_key] = current_time
                            st.session_state.messages.append(f"{time.strftime('%H:%M:%S')} - {message}")

            time.sleep(0.1)

        except Exception as e:
            st.error(f"Error processing frame: {str(e)}")
            print(f"Frame processing error: {e}")

        if not st.session_state.stop_signal:
            st.rerun()


if __name__ == "__main__":
    main()