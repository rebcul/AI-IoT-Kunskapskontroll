# =========================================================
# SafeWatch
# Kunskapskontroll 1 – AI och IoT
# =========================================================
#
# SafeWatch är ett säkerhetssystem som kombinerar
# objektdetektion, tracking och enkel larmlogik.
#
# Två modeller används:
# - weapon model för farliga föremål
# - item model för väskor och liknande objekt
#
# Systemet kan ge två typer av larm:
# 1. Person nära farligt föremål
# 2. Lämnat föremål
# =========================================================

import os
import time
import tempfile
from datetime import datetime

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# =========================================================
# SIDINSTÄLLNINGAR
# =========================================================

st.set_page_config(
    page_title="SafeWatch",
    page_icon="🛡️",
    layout="wide"
)

# =========================================================
# CSS
# =========================================================


def inject_css():
    st.markdown(
        """
        <style>
        .main {
            background: linear-gradient(180deg, #0b1220 0%, #111827 100%);
        }
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }
        .hero-box {
            background: linear-gradient(135deg, #111827 0%, #182235 100%);
            border: 1px solid #2b3648;
            border-radius: 18px;
            padding: 22px 26px;
            margin-bottom: 18px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.18);
        }
        .hero-title {
            color: #f9fafb;
            font-size: 30px;
            font-weight: 800;
            margin-bottom: 8px;
        }
        .hero-sub {
            color: #cbd5e1;
            font-size: 15px;
            line-height: 1.6;
        }
        .system-strip {
            display: flex;
            gap: 14px;
            flex-wrap: wrap;
            margin-bottom: 12px;
        }
        .system-pill {
            background: #0f172a;
            border: 1px solid #243041;
            color: #cbd5e1;
            padding: 8px 12px;
            border-radius: 999px;
            font-size: 12px;
            letter-spacing: 0.02em;
        }
        .alert-danger {
            background: #2a1515;
            border: 1px solid #8b2c2c;
            border-radius: 12px;
            padding: 12px 16px;
            margin-bottom: 10px;
            box-shadow: 0 8px 20px rgba(120, 28, 28, 0.18);
        }
        .alert-warning {
            background: #2a2110;
            border: 1px solid #8a5a12;
            border-radius: 12px;
            padding: 12px 16px;
            margin-bottom: 10px;
            box-shadow: 0 8px 20px rgba(138, 90, 18, 0.15);
        }
        .alert-info {
            background: #10202a;
            border: 1px solid #235a85;
            border-radius: 12px;
            padding: 12px 16px;
            margin-bottom: 10px;
            box-shadow: 0 8px 20px rgba(35, 90, 133, 0.15);
        }
        .alert-title {
            color: #f8fafc;
            font-weight: 700;
            font-size: 14px;
            margin-bottom: 4px;
        }
        .alert-sub {
            color: #cbd5e1;
            font-size: 12px;
        }
        .critical-banner {
            background: linear-gradient(90deg, #7f1d1d 0%, #991b1b 50%, #7f1d1d 100%);
            border: 1px solid #ef4444;
            color: #fff7f7;
            border-radius: 14px;
            padding: 16px 18px;
            font-size: 18px;
            font-weight: 800;
            margin-bottom: 16px;
            animation: pulseAlert 1s infinite;
            box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.55);
        }
        @keyframes pulseAlert {
            0% { transform: scale(1); box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.50); }
            70% { transform: scale(1.01); box-shadow: 0 0 0 16px rgba(239, 68, 68, 0.00); }
            100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.00); }
        }
        .status-ok {
            background: #0f2a1d;
            border: 1px solid #1f8a4c;
            color: #dcfce7;
            border-radius: 12px;
            padding: 12px 16px;
            margin-bottom: 12px;
            font-weight: 600;
        }
        .event-log-box {
            background: #0f172a;
            border: 1px solid #233044;
            border-radius: 14px;
            padding: 14px 16px;
            margin-top: 10px;
        }
        .event-log-title {
            color: #f8fafc;
            font-size: 14px;
            font-weight: 700;
            margin-bottom: 10px;
        }
        .event-row {
            border-bottom: 1px solid #1e293b;
            padding: 8px 0;
            color: #cbd5e1;
            font-size: 13px;
        }
        .event-row:last-child {
            border-bottom: none;
        }
        .rec-indicator {
            position: fixed;
            top: 18px;
            right: 22px;
            z-index: 9999;
            background: rgba(127, 29, 29, 0.95);
            border: 1px solid #ef4444;
            color: #fff;
            border-radius: 999px;
            padding: 8px 14px;
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 0.08em;
            box-shadow: 0 8px 20px rgba(239, 68, 68, 0.22);
            animation: recBlink 1s infinite;
        }
        .rec-dot {
            display: inline-block;
            width: 9px;
            height: 9px;
            background: #ff3b3b;
            border-radius: 50%;
            margin-right: 8px;
            box-shadow: 0 0 10px rgba(255, 59, 59, 0.9);
        }
        @keyframes recBlink {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.55; }
        }
        div[data-testid="stFileUploader"] {
            border: 1px dashed #334155;
            border-radius: 12px;
            padding: 10px;
            background: rgba(17, 24, 39, 0.45);
        }
        div[data-testid="stMetric"] {
            background: #111827;
            border: 1px solid #243041;
            border-radius: 12px;
            padding: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


inject_css()

st.markdown(
    """
    <div class="rec-indicator">
        <span class="rec-dot"></span>REC / ALERT ACTIVE
    </div>
    """,
    unsafe_allow_html=True
)

# =========================================================
# MODELLSÖKVÄGAR
# =========================================================

WEAPON_MODEL_PATH = r"C:\temp\SafeWatch\runs\detect\weapon_finetune_test\weights\best.pt"
ITEM_MODEL_PATH = r"C:\temp\SafeWatch\runs\detect\item_finetune1\weights\best.pt"

# =========================================================
# KLASSER
# =========================================================

WEAPON_CLASSES = {
    "dangerous weapon": "Dangerous weapon",
    "pistol": "Pistol",
    "gun": "Gun",
    "knife": "Knife",
    "rifle": "Rifle",
    "axe": "Axe",
    "hammer": "Hammer"
}

ITEM_CLASSES = {
    "backpack": "Backpack",
    "empty_tray": "Empty Tray",
    "filled_tray": "Filled Tray",
    "handbag": "Handbag",
    "suitcase": "Suitcase"
}

PERSON_CLASS = "person"

# =========================================================
# FÄRGER
# =========================================================

COLOR_DANGER = (0, 0, 255)
COLOR_LEFT = (0, 165, 255)
COLOR_PERSON = (0, 255, 0)
COLOR_ITEM = (0, 255, 255)

# =========================================================
# INSTÄLLNINGAR
# =========================================================

IMAGE_CONF = 0.40
WEAPON_VIDEO_CONF = 0.45
ITEM_VIDEO_CONF = 0.25
WEAPON_ALERT_CONFIDENCE = 65.0
PERSON_WEAPON_DISTANCE = 120
LEFT_OBJECT_SECONDS_DEFAULT = 5
THROWN_SPEED_THRESHOLD_DEFAULT = 35
PERSON_NEAR_ITEM_DISTANCE_DEFAULT = 220
MIN_MOVEMENT_RESET = 45
FRAME_SKIP = 5

# =========================================================
# LADDA MODELLER
# =========================================================


@st.cache_resource
def load_models():
    weapon_model = YOLO(WEAPON_MODEL_PATH)
    item_model = YOLO(ITEM_MODEL_PATH)
    return weapon_model, item_model


weapon_model, item_model = load_models()

# =========================================================
# HJÄLPFUNKTIONER
# =========================================================


def now_str():
    return datetime.now().strftime("%H:%M:%S")


def euclidean_distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def box_center(x1, y1, x2, y2):
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def normalize_class_name(name):
    return str(name).strip().lower()


def is_reasonable_box(x1, y1, x2, y2, image_shape, max_area_ratio=0.75):
    img_h, img_w = image_shape[:2]
    box_area = max(0, x2 - x1) * max(0, y2 - y1)
    image_area = img_h * img_w

    if image_area == 0:
        return False

    return (box_area / image_area) <= max_area_ratio


def get_label(class_name):
    class_name = normalize_class_name(class_name)
    if class_name in WEAPON_CLASSES:
        return WEAPON_CLASSES[class_name]
    if class_name in ITEM_CLASSES:
        return ITEM_CLASSES[class_name]
    return class_name.title()


def parse_yolo_results(results, source_model, image_shape):
    detections = []

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = normalize_class_name(source_model.names[class_id])

            if source_model is weapon_model and class_name == "helmet":
                continue

            confidence = round(float(box.conf[0]) * 100, 1)
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if not is_reasonable_box(x1, y1, x2, y2, image_shape):
                continue

            track_id = None
            if hasattr(box, "id") and box.id is not None:
                track_id = int(box.id[0])

            detections.append({
                "class_name": class_name,
                "label": get_label(class_name),
                "confidence": confidence,
                "bbox": (x1, y1, x2, y2),
                "center": box_center(x1, y1, x2, y2),
                "track_id": track_id,
                "is_person": class_name == PERSON_CLASS,
                "is_weapon": class_name in WEAPON_CLASSES,
                "is_item": class_name in ITEM_CLASSES,
            })

    return detections


def merge_detections(detections_a, detections_b):
    return detections_a + detections_b

# =========================================================
# LARMLOGIK
# =========================================================


def detect_person_with_weapon(
    detections,
    threshold_px=PERSON_WEAPON_DISTANCE,
    active_pairs=None,
    min_confidence=WEAPON_ALERT_CONFIDENCE
):
    if active_pairs is None:
        active_pairs = set()

    alerts = []
    new_active_pairs = set()

    persons = [d for d in detections if d["is_person"]]
    weapons = [
        d for d in detections
        if normalize_class_name(d["class_name"]) in WEAPON_CLASSES
        and d["confidence"] >= min_confidence
    ]

    for person in persons:
        for weapon in weapons:
            distance = euclidean_distance(person["center"], weapon["center"])
            if distance <= threshold_px:
                pair_id = (
                    person.get("track_id"),
                    weapon.get("track_id"),
                    weapon["label"]
                )
                new_active_pairs.add(pair_id)

                if pair_id not in active_pairs:
                    alerts.append({
                        "type": "Person holding dangerous object",
                        "object": weapon["label"],
                        "distance": round(distance),
                        "time": now_str(),
                        "person_center": person["center"],
                        "weapon_center": weapon["center"]
                    })

    return alerts, new_active_pairs


class UnattendedTracker:
    def __init__(self, alert_seconds=20, speed_threshold=35, person_distance=150):
        self.alert_seconds = alert_seconds
        self.speed_threshold = speed_threshold
        self.person_distance = person_distance
        self.positions = {}
        self.alone_since_frame = {}
        self.class_names = {}
        self.alert_history = []
        self.alerted_ids = set()

    def update(self, object_id, position, class_name, person_positions, frame_number, fps):
        alert = None

        person_near = False
        for px, py in person_positions:
            if euclidean_distance((px, py), position) < self.person_distance:
                person_near = True
                break

        if object_id not in self.positions:
            self.positions[object_id] = position
            self.alone_since_frame[object_id] = None
            self.class_names[object_id] = class_name
            return None

        old_position = self.positions[object_id]
        movement = euclidean_distance(old_position, position)
        self.class_names[object_id] = class_name
        self.positions[object_id] = position

        if movement > MIN_MOVEMENT_RESET:
            self.alone_since_frame[object_id] = None
            if object_id in self.alerted_ids:
                self.alerted_ids.remove(object_id)
            return None

        if class_name in ITEM_CLASSES:
            if person_near:
                self.alone_since_frame[object_id] = None
            else:
                if self.alone_since_frame[object_id] is None:
                    self.alone_since_frame[object_id] = frame_number

                still_seconds = (
                    frame_number - self.alone_since_frame[object_id]
                ) / fps

                if still_seconds > self.alert_seconds and object_id not in self.alerted_ids:
                    alert = {
                        "type": "Left object",
                        "object": get_label(class_name),
                        "id": object_id,
                        "seconds": round(still_seconds, 1),
                        "time": now_str()
                    }
                    self.alert_history.append(alert)
                    self.alerted_ids.add(object_id)

        return alert

    def cleanup(self, active_ids):
        for object_id in list(self.positions.keys()):
            if object_id not in active_ids:
                del self.positions[object_id]
                del self.alone_since_frame[object_id]
                del self.class_names[object_id]
                if object_id in self.alerted_ids:
                    self.alerted_ids.remove(object_id)

# =========================================================
# RITA RESULTAT
# =========================================================


def draw_detections(image, detections, person_weapon_alerts=None, left_ids=None):
    output = image.copy()
    person_weapon_alerts = person_weapon_alerts or []
    left_ids = left_ids or set()

    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]

        if detection["is_person"]:
            color = COLOR_PERSON
        elif detection["track_id"] in left_ids:
            color = COLOR_LEFT
        elif normalize_class_name(detection["class_name"]) in WEAPON_CLASSES:
            color = COLOR_DANGER
        elif detection["is_item"]:
            color = COLOR_ITEM
        else:
            color = COLOR_PERSON

        text = detection["label"]
        if detection["track_id"] is not None:
            text = f"ID-{detection['track_id']} {text}"

        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            output,
            text,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2
        )

    for alert in person_weapon_alerts:
        cv2.line(
            output,
            alert["person_center"],
            alert["weapon_center"],
            COLOR_DANGER,
            2
        )

    return output


def show_alert(alert_type, title, detail=""):
    class_name = {
        "danger": "alert-danger",
        "warning": "alert-warning",
        "info": "alert-info"
    }.get(alert_type, "alert-info")

    st.markdown(
        f'<div class="{class_name}">'
        f'<div class="alert-title">{title}</div>'
        f'<div class="alert-sub">{detail}</div>'
        f'</div>',
        unsafe_allow_html=True
    )


def show_critical_banner(message):
    st.markdown(
        f'<div class="critical-banner">⚠ SECURITY ALERT — {message}</div>',
        unsafe_allow_html=True
    )


def show_status_ok(message):
    st.markdown(
        f'<div class="status-ok">✅ {message}</div>',
        unsafe_allow_html=True
    )


def render_event_log(title, events):
    if not events:
        return

    rows = []
    for event in events[-8:][::-1]:
        rows.append(f'<div class="event-row">{event}</div>')

    st.markdown(
        f'<div class="event-log-box">'
        f'<div class="event-log-title">{title}</div>'
        f'{"".join(rows)}'
        f'</div>',
        unsafe_allow_html=True
    )


def play_alert_sound():
    st.components.v1.html(
        """
        <script>
        (function() {
            const ctx = new (window.AudioContext || window.webkitAudioContext)();
            const playBeep = (freq, duration, delay) => {
                const osc = ctx.createOscillator();
                const gain = ctx.createGain();
                osc.type = 'sine';
                osc.frequency.value = freq;
                gain.gain.value = 0.04;
                osc.connect(gain);
                gain.connect(ctx.destination);
                const start = ctx.currentTime + delay;
                osc.start(start);
                osc.stop(start + duration);
            };
            playBeep(880, 0.18, 0.00);
            playBeep(660, 0.18, 0.22);
            playBeep(880, 0.22, 0.46);
        })();
        </script>
        """,
        height=0,
    )

# =========================================================
# BILDANALYS
# =========================================================


def analyze_image(image_array):
    weapon_results = weapon_model(image_array, conf=IMAGE_CONF)
    item_results = item_model(image_array, conf=IMAGE_CONF)

    weapon_detections = parse_yolo_results(
        weapon_results, weapon_model, image_array.shape)
    item_detections = parse_yolo_results(
        item_results, item_model, image_array.shape)

    detections = merge_detections(weapon_detections, item_detections)
    person_weapon_alerts, _ = detect_person_with_weapon(detections)

    annotated = draw_detections(
        image_array,
        detections,
        person_weapon_alerts=person_weapon_alerts
    )

    return annotated, detections, person_weapon_alerts

# =========================================================
# VIDEOANALYS
# =========================================================


def analyze_video(video_path, alert_seconds, speed_threshold, person_distance):
    tracker = UnattendedTracker(
        alert_seconds=alert_seconds,
        speed_threshold=speed_threshold,
        person_distance=person_distance
    )

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25

    preview_frames = []
    frame_number = 0
    person_weapon_alerts_all = []
    active_pairs = set()

    while cap.isOpened():
        ret, frame_bgr = cap.read()
        if not ret:
            break

        if frame_number % FRAME_SKIP != 0:
            frame_number += 1
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        weapon_results = weapon_model.track(
            frame_rgb,
            conf=WEAPON_VIDEO_CONF,
            persist=True
        )
        item_results = item_model.track(
            frame_rgb,
            conf=ITEM_VIDEO_CONF,
            persist=True
        )

        weapon_detections = parse_yolo_results(
            weapon_results, weapon_model, frame_rgb.shape)
        item_detections = parse_yolo_results(
            item_results, item_model, frame_rgb.shape)
        detections = merge_detections(weapon_detections, item_detections)

        person_positions = [d["center"] for d in detections if d["is_person"]]
        active_ids = set()
        left_ids = set()

        for detection in item_detections:
            if detection["track_id"] is None:
                continue

            active_ids.add(detection["track_id"])

            tracker.update(
                detection["track_id"],
                detection["center"],
                detection["class_name"],
                person_positions,
                frame_number,
                fps
            )

            if detection["track_id"] in tracker.alerted_ids:
                left_ids.add(detection["track_id"])

        tracker.cleanup(active_ids)

        person_weapon_alerts, active_pairs = detect_person_with_weapon(
            detections,
            active_pairs=active_pairs
        )

        for alert in person_weapon_alerts:
            alert["frame"] = frame_number
            alert["video_time"] = round(frame_number / fps, 1)
            person_weapon_alerts_all.append(alert)

        annotated = draw_detections(
            frame_rgb,
            detections,
            person_weapon_alerts=person_weapon_alerts,
            left_ids=left_ids
        )

        preview_frames.append(annotated)
        frame_number += 1

    cap.release()
    return preview_frames, tracker.alert_history, person_weapon_alerts_all

# =========================================================
# LIVEKAMERA
# =========================================================


def analyze_camera_frame(frame_bgr, tracker, active_pairs, camera_frame_number):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    weapon_results = weapon_model.track(
        frame_rgb,
        conf=WEAPON_VIDEO_CONF,
        persist=True
    )
    item_results = item_model.track(
        frame_rgb,
        conf=ITEM_VIDEO_CONF,
        persist=True
    )

    weapon_detections = parse_yolo_results(
        weapon_results, weapon_model, frame_rgb.shape)
    item_detections = parse_yolo_results(
        item_results, item_model, frame_rgb.shape)
    detections = merge_detections(weapon_detections, item_detections)

    person_positions = [d["center"] for d in detections if d["is_person"]]
    active_ids = set()
    left_ids = set()

    for detection in item_detections:
        if detection["track_id"] is None:
            continue

        active_ids.add(detection["track_id"])
        tracker.update(
            detection["track_id"],
            detection["center"],
            detection["class_name"],
            person_positions,
            camera_frame_number,
            20
        )

        if detection["track_id"] in tracker.alerted_ids:
            left_ids.add(detection["track_id"])

    tracker.cleanup(active_ids)

    person_weapon_alerts, active_pairs = detect_person_with_weapon(
        detections,
        active_pairs=active_pairs
    )

    annotated = draw_detections(
        frame_rgb,
        detections,
        person_weapon_alerts=person_weapon_alerts,
        left_ids=left_ids
    )

    return annotated, detections, person_weapon_alerts, active_pairs, tracker.alert_history

# =========================================================
# GRÄNSSNITT
# =========================================================


st.markdown(
    """
    <div class="system-strip">
        <div class="system-pill">CAM 01</div>
        <div class="system-pill">AI SECURITY MODE</div>
        <div class="system-pill">REC ● ACTIVE</div>
        <div class="system-pill">LAYERED DETECTION</div>
    </div>
    <div class="hero-box">
        <div class="hero-title">SafeWatch – AI-driven detektering av farliga föremål</div>
        <div class="hero-sub">
            Ladda upp en bild, video eller använd livekamera för att analysera farliga föremål,
            personer nära vapen och obevakade objekt. Systemet kombinerar objektdetektion,
            tracking och enkel larmlogik.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    st.header("About SafeWatch")
    st.write(
        "Systemet använder två modeller och enkel tracking-logik. "
        "En modell fokuserar på farliga föremål och en annan på "
        "objekt som kan ha lämnats kvar."
    )

    st.divider()
    st.subheader("Alert types")
    st.write("- Person holding dangerous object")
    st.write("- Left unattended object")

    st.divider()
    st.subheader("Settings")

    alert_seconds = st.slider(
        "Seconds before left-object alert",
        min_value=5,
        max_value=120,
        value=LEFT_OBJECT_SECONDS_DEFAULT
    )

    speed_threshold = st.slider(
        "Thrown object speed (px/frame)",
        min_value=10,
        max_value=100,
        value=THROWN_SPEED_THRESHOLD_DEFAULT
    )

    person_distance = st.slider(
        "Person distance for item supervision (px)",
        min_value=50,
        max_value=300,
        value=PERSON_NEAR_ITEM_DISTANCE_DEFAULT
    )

analysis_type = st.radio(
    "Choose analysis type:",
    ["Image", "Video with tracking", "Live camera"],
    horizontal=True
)

st.divider()

# =========================================================
# BILDLÄGE
# =========================================================

if analysis_type == "Image":
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        image_array = np.array(image)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original image")
            st.image(image, use_container_width=True)

        with st.spinner("Analyzing image..."):
            annotated, detections, person_weapon_alerts = analyze_image(
                image_array)

        with col2:
            st.subheader("Analysis result")
            st.image(annotated, use_container_width=True)

        st.divider()

        dangerous_detections = [
            d for d in detections
            if normalize_class_name(d["class_name"]) in WEAPON_CLASSES
        ]

        item_detections = [
            d for d in detections
            if normalize_class_name(d["class_name"]) in ITEM_CLASSES
        ]

        image_event_rows = []

        if person_weapon_alerts:
            play_alert_sound()
            show_critical_banner("person holding dangerous object")
            for alert in person_weapon_alerts:
                show_alert(
                    "danger",
                    f"ALERT – Person holding {alert['object']}",
                    f"Distance: {alert['distance']} px — Detected at {alert['time']}"
                )
                image_event_rows.append(
                    f"{alert['time']} — Person holding {alert['object']} ({alert['distance']} px)"
                )
        elif dangerous_detections:
            play_alert_sound()
            show_critical_banner("dangerous object detected")
            for detection in dangerous_detections:
                show_alert(
                    "warning",
                    f"Warning – Dangerous object detected: {detection['label']}",
                    f"Confidence: {detection['confidence']}%"
                )
                image_event_rows.append(
                    f"{now_str()} — Dangerous object detected: {detection['label']} ({detection['confidence']}%)"
                )
        else:
            show_status_ok("No dangerous objects were detected.")

        if item_detections:
            for detection in item_detections:
                show_alert(
                    "info",
                    f"Monitored item detected: {detection['label']}",
                    f"Confidence: {detection['confidence']}% — Left-object logic is evaluated in video and live mode"
                )
                image_event_rows.append(
                    f"{now_str()} — Monitored item detected: {detection['label']} ({detection['confidence']}%)"
                )

        render_event_log("Recent image events", image_event_rows)

        if detections:
            with st.expander("Show all detections"):
                for detection in detections:
                    st.write(
                        f"- {detection['label']}: {detection['confidence']}%")

        st.caption(
            f"Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

# =========================================================
# VIDEOLÄGE
# =========================================================

elif analysis_type == "Video with tracking":
    uploaded_video = st.file_uploader(
        "Upload a video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video is not None:
        st.video(uploaded_video)
        st.divider()

        if st.button("Start analysis"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(uploaded_video.read())
                tmp_path = tmp.name

            with st.spinner("Analyzing video with tracking..."):
                frames, history, person_weapon_alerts = analyze_video(
                    tmp_path,
                    alert_seconds,
                    speed_threshold,
                    person_distance
                )

            os.unlink(tmp_path)

            left_alerts = [a for a in history if a["type"] == "Left object"]
            all_alerts = history + person_weapon_alerts

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Frames analyzed", len(frames))
            with c2:
                st.metric("Person + weapon", len(person_weapon_alerts))
            with c3:
                st.metric("Left objects", len(left_alerts))

            st.divider()
            video_event_rows = []

            if all_alerts:
                play_alert_sound()
                show_critical_banner(f"{len(all_alerts)} event(s) detected")
                show_alert(
                    "danger",
                    f"ALERT – {len(all_alerts)} event(s) detected",
                    "Review the event history below for details"
                )
                st.subheader("Alert history")

                for alert in person_weapon_alerts:
                    show_alert(
                        "danger",
                        f"Person holding {alert['object']}",
                        f"At {alert.get('video_time', '?')} sec"
                    )
                    video_event_rows.append(
                        f"{alert.get('video_time', '?')} sec — Person holding {alert['object']}"
                    )

                for alert in history:
                    if alert["type"] == "Left object":
                        show_alert(
                            "warning",
                            f"Left object: {alert['object']}",
                            f"Stationary for {alert['seconds']} sec"
                        )
                        video_event_rows.append(
                            f"{alert['time']} — Left object: {alert['object']} ({alert['seconds']} sec)"
                        )
            else:
                show_status_ok("No suspicious events were detected.")

            render_event_log("Recent video events", video_event_rows)

            if frames:
                st.divider()
                st.subheader("Analyzed frames")
                cols = st.columns(3)
                sample_frames = frames[::max(1, len(frames) // 9)][:9]
                for i, frame in enumerate(sample_frames):
                    with cols[i % 3]:
                        st.image(
                            frame,
                            caption=f"Sample {i + 1}",
                            use_container_width=True
                        )

            st.caption(
                f"Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )

# =========================================================
# LIVEKAMERA
# =========================================================

elif analysis_type == "Live camera":
    st.info(
        "Live camera mode analyzes webcam frames in real time. "
        "Press Start to begin and Stop to end the session."
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        camera_placeholder = st.empty()
    with col2:
        st.subheader("Live alerts")
        live_alert_placeholder = st.empty()

    if "camera_active" not in st.session_state:
        st.session_state["camera_active"] = False

    start_button = st.button("Start camera")
    stop_button = st.button("Stop camera")

    if start_button:
        st.session_state["camera_active"] = True

    if stop_button:
        st.session_state["camera_active"] = False

    if st.session_state.get("camera_active", False):
        tracker = UnattendedTracker(
            alert_seconds=alert_seconds,
            speed_threshold=speed_threshold,
            person_distance=person_distance
        )
        active_pairs = set()
        cap = cv2.VideoCapture(0)
        camera_frame_number = 0

        if not cap.isOpened():
            st.error("Could not open camera. Make sure a webcam is connected.")
        else:
            st.success("Camera active — analyzing in real time...")

            while st.session_state.get("camera_active", False):
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to read camera frame.")
                    break

                annotated, detections, person_weapon_alerts, active_pairs, history = analyze_camera_frame(
                    frame,
                    tracker,
                    active_pairs,
                    camera_frame_number
                )
                camera_frame_number += 1

                camera_placeholder.image(
                    annotated,
                    channels="RGB",
                    use_container_width=True
                )

                latest_messages = []

                for alert in person_weapon_alerts[-3:]:
                    latest_messages.append((
                        "danger",
                        f"Person holding {alert['object']}",
                        f"Detected at {alert['time']}"
                    ))

                for alert in history[-3:]:
                    if alert["type"] == "Left object":
                        latest_messages.append((
                            "warning",
                            f"Left object: {alert['object']}",
                            f"Stationary for {alert['seconds']} sec"
                        ))

                with live_alert_placeholder.container():
                    if latest_messages:
                        play_alert_sound()
                        for level, title, detail in latest_messages[-5:]:
                            show_alert(level, title, detail)
                    else:
                        st.write("No active alerts.")

                time.sleep(0.05)

            cap.release()
            st.info("Camera stopped.")

# =========================================================
# REFLEKTION
# =========================================================
# SafeWatch visar hur ett mer realistiskt säkerhetssystem kan
# byggas genom att kombinera objektdetektion, tracking och logik.
# Det viktigaste jag lärt mig är att modellens output inte räcker
# ensam. Det behövs också tydliga regler och en bra struktur runt
# lösningen för att systemet ska fungera i praktiken.

#
# Etiskt perspektiv:
# Ett system som analyserar video måste användas ansvarsfullt.
# I den här lösningen ligger fokus på objekt och beteenden
# istället för identitet, men det är fortfarande viktigt att tänka
# på integritet och tydlig information till användare.
#
# Vidareutveckling:
# Projektet kan utvecklas vidare med bättre träningsdata,
# stöd för fler kameratyper och export av larmhistorik.
# I den ursprungliga planen ingick även att upptäcka kastade objekt.
# Den funktionen hann jag inte färdigställa i denna version, men vissa
# delar finns kvar i koden eftersom det är något jag vill utveckla vidare.
