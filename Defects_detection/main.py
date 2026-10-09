
import os
import sys
import json
import time
import uuid
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights


# =========================================================
# CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Fabric Anomaly Detection",
    page_icon="🔍",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

CAMERA_INDEX = 0
INSPECTION_INTERVAL = 5

CAMERA_DIR = BASE_DIR / "camera_runtime"
INCOMING_DIR = CAMERA_DIR / "incoming"
STOP_FILE = CAMERA_DIR / "stop_camera.flag"
STATUS_FILE = CAMERA_DIR / "camera_status.json"

RESULTS_DIR = BASE_DIR / "results"
OK_DIR = RESULTS_DIR / "OK"
NOK_DIR = RESULTS_DIR / "NOK"

# Fixed guide coordinates as proportions of the frame.
# Adjust these four values to change the guide size.
GUIDE_LEFT = 0.20
GUIDE_TOP = 0.18
GUIDE_RIGHT = 0.80
GUIDE_BOTTOM = 0.82

# Basic blank/low-texture rejection, not fabric-edge validation.
MIN_GRAY_STD = 4.0


# =========================================================
# RESNET50 FEATURE EXTRACTOR
# =========================================================

class resnet_feature_extractor(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = resnet50(
            weights=ResNet50_Weights.DEFAULT
        )

        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.layer2[-1].register_forward_hook(
            self._hook
        )

        self.model.layer3[-1].register_forward_hook(
            self._hook
        )

    def _hook(self, module, inputs, output):
        self.features.append(output)

    def forward(self, input):
        self.features = []

        with torch.no_grad():
            self.model(input)

        avg = nn.AvgPool2d(3, stride=1)
        fmap_size = self.features[0].shape[-2]

        resize = nn.AdaptiveAvgPool2d(fmap_size)

        resized_maps = [
            resize(avg(fmap))
            for fmap in self.features
        ]

        return torch.cat(resized_maps, dim=1)


# =========================================================
# AUTOENCODER
# =========================================================

class FeatCAE(nn.Module):

    def __init__(
        self,
        in_channels=1536,
        latent_dim=100,
        is_bn=True
    ):
        super().__init__()

        middle_channels = (
            in_channels + 2 * latent_dim
        ) // 2

        encoder_layers = [
            nn.Conv2d(
                in_channels,
                middle_channels,
                kernel_size=1
            )
        ]

        if is_bn:
            encoder_layers.append(
                nn.BatchNorm2d(middle_channels)
            )

        encoder_layers.append(nn.ReLU())

        encoder_layers.append(
            nn.Conv2d(
                middle_channels,
                2 * latent_dim,
                kernel_size=1
            )
        )

        if is_bn:
            encoder_layers.append(
                nn.BatchNorm2d(2 * latent_dim)
            )

        encoder_layers.append(nn.ReLU())

        encoder_layers.append(
            nn.Conv2d(
                2 * latent_dim,
                latent_dim,
                kernel_size=1
            )
        )

        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = [
            nn.Conv2d(
                latent_dim,
                2 * latent_dim,
                kernel_size=1
            )
        ]

        if is_bn:
            decoder_layers.append(
                nn.BatchNorm2d(2 * latent_dim)
            )

        decoder_layers.append(nn.ReLU())

        decoder_layers.append(
            nn.Conv2d(
                2 * latent_dim,
                middle_channels,
                kernel_size=1
            )
        )

        if is_bn:
            decoder_layers.append(
                nn.BatchNorm2d(middle_channels)
            )

        decoder_layers.append(nn.ReLU())

        decoder_layers.append(
            nn.Conv2d(
                middle_channels,
                in_channels,
                kernel_size=1
            )
        )

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        return self.decoder(self.encoder(x))


# =========================================================
# ANOMALY SCORING
# =========================================================

def decision_function(segm_map):
    scores = []

    for anomaly_map in segm_map:
        flattened = anomaly_map.reshape(-1)

        sorted_values, _ = torch.sort(
            flattened,
            descending=True
        )

        scores.append(sorted_values[:10].mean())

    return torch.stack(scores)


# =========================================================
# MODEL DISCOVERY
# =========================================================

def get_model_folders(base_path="models"):

    path = Path(base_path)

    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        return []

    folders = []

    for folder_path in path.iterdir():

        if not folder_path.is_dir():
            continue

        checkpoint = (
            folder_path / "anomaly_model_complete.pth"
        )

        config = folder_path / "threshold_config.json"

        if checkpoint.is_file() and config.is_file():
            folders.append(folder_path.name)

    return sorted(folders)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_models(model_folder_path):

    checkpoint_path = os.path.join(
        model_folder_path,
        "anomaly_model_complete.pth"
    )

    threshold_path = os.path.join(
        model_folder_path,
        "threshold_config.json"
    )

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)

    if not os.path.isfile(threshold_path):
        raise FileNotFoundError(threshold_path)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu"
    )

    with open(
        threshold_path,
        "r",
        encoding="utf-8"
    ) as file:
        threshold_config = json.load(file)

    backbone = resnet_feature_extractor()
    backbone.eval()

    model_config = checkpoint["model_config"]

    model = FeatCAE(
        in_channels=model_config["in_channels"],
        latent_dim=model_config["latent_dim"],
        is_bn=model_config["is_bn"]
    )

    model.load_state_dict(
        checkpoint["autoencoder_state_dict"]
    )

    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    return backbone, model, transform, threshold_config


# =========================================================
# ANOMALY DETECTION
# =========================================================

def tensor_to_array(tensor):

    return np.asarray(
        tensor.detach().cpu().tolist(),
        dtype=np.float32
    )


def detect_anomaly(
    image,
    backbone,
    model,
    transform,
    best_threshold
):

    image = image.convert("RGB")
    test_image = transform(image).unsqueeze(0)

    with torch.no_grad():

        features = backbone(test_image)
        reconstruction = model(features)

        segm_map = (
            (features - reconstruction) ** 2
        ).mean(dim=1)

        image_score = decision_function(segm_map)

        score = float(image_score[0].item())

        is_anomaly = score >= best_threshold

        heat_map = tensor_to_array(
            segm_map.squeeze()
        )

        heat_map_resized = cv2.resize(
            heat_map,
            (128, 128)
        )

        # Preserves the segmentation rule from your original code.
        # If masks remain empty, this rule needs calibration.
        segmentation_mask = (
            heat_map_resized
            > float(heat_map_resized.max()) * 9
        ).astype(np.uint8)

        original_image = tensor_to_array(
            test_image.squeeze().permute(1, 2, 0)
        )

    return {
        "original_image": original_image,
        "heat_map": heat_map_resized,
        "segmentation_mask": segmentation_mask,
        "anomaly_score": score,
        "is_anomaly": bool(is_anomaly),
        "normalized_score": score / max(
            float(best_threshold),
            1e-12
        )
    }


# =========================================================
# CAMERA STATUS
# =========================================================

def write_camera_status(status, message=""):

    CAMERA_DIR.mkdir(parents=True, exist_ok=True)

    temp_path = CAMERA_DIR / "camera_status.tmp"

    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "status": status,
                "message": message,
                "time": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            file
        )

    os.replace(temp_path, STATUS_FILE)


def read_camera_status():

    try:
        with open(STATUS_FILE, "r", encoding="utf-8") as file:
            return json.load(file)

    except (OSError, json.JSONDecodeError):
        return {
            "status": "unknown",
            "message": "Waiting for camera status."
        }


# =========================================================
# SEPARATE FULL-SCREEN CAMERA WINDOW
# =========================================================

def run_camera_window(camera_index=CAMERA_INDEX):

    CAMERA_DIR.mkdir(parents=True, exist_ok=True)
    INCOMING_DIR.mkdir(parents=True, exist_ok=True)

    STOP_FILE.unlink(missing_ok=True)

    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened() and os.name == "nt":
        cap.release()
        cap = cv2.VideoCapture(
            camera_index,
            cv2.CAP_DSHOW
        )

    if not cap.isOpened():

        write_camera_status(
            "error",
            f"Cannot open camera index {camera_index}"
        )

        return

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    window_name = "Fabric Inspection Camera - Q to close"

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL
    )

    cv2.setWindowProperty(
        window_name,
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN
    )

    next_capture = (
        time.monotonic() + INSPECTION_INTERVAL
    )

    write_camera_status(
        "running",
        "Place fabric inside the yellow guide."
    )

    try:

        while True:

            if STOP_FILE.exists():
                break

            success, frame = cap.read()

            if not success or frame is None:
                write_camera_status(
                    "error",
                    "Failed to read camera frame."
                )
                time.sleep(0.1)
                continue

            height, width = frame.shape[:2]

            x1 = int(width * GUIDE_LEFT)
            y1 = int(height * GUIDE_TOP)
            x2 = int(width * GUIDE_RIGHT)
            y2 = int(height * GUIDE_BOTTOM)

            # Fixed yellow rectangular guide.
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 255),
                3
            )

            cv2.putText(
                frame,
                "KEEP FABRIC INSIDE YELLOW GUIDE",
                (x1, max(30, y1 - 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA
            )

            remaining = max(
                0,
                int(next_capture - time.monotonic())
            )

            cv2.putText(
                frame,
                f"Next inspection: {remaining}s",
                (25, height - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            now = time.monotonic()

            if now >= next_capture:

                crop = frame[y1:y2, x1:x2].copy()

                if crop.size:

                    gray = cv2.cvtColor(
                        crop,
                        cv2.COLOR_BGR2GRAY
                    )

                    # Rejects only nearly blank / low-texture crops.
                    # It does not prove that fabric edges are inside the guide.
                    if float(np.std(gray)) >= MIN_GRAY_STD:

                        filename = (
                            "capture_"
                            + time.strftime("%Y%m%d_%H%M%S")
                            + "_"
                            + uuid.uuid4().hex[:6]
                            + ".jpg"
                        )

                        final_path = INCOMING_DIR / filename

                        temp_path = INCOMING_DIR / (
                            filename + ".tmp.jpg"
                        )

                        if cv2.imwrite(str(temp_path), crop):

                            os.replace(
                                temp_path,
                                final_path
                            )

                            write_camera_status(
                                "running",
                                f"Captured {filename}"
                            )

                            cv2.putText(
                                frame,
                                "CAPTURED",
                                (x1, min(height - 10, y2 + 30)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (0, 255, 0),
                                2
                            )

                    else:

                        write_camera_status(
                            "running",
                            "Skipped blank or low-texture crop."
                        )

                next_capture = (
                    time.monotonic() + INSPECTION_INTERVAL
                )

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

    finally:

        cap.release()
        cv2.destroyAllWindows()

        write_camera_status(
            "stopped",
            "Camera window closed."
        )


# =========================================================
# CAMERA PROCESS CONTROLS
# =========================================================

def start_camera_process():

    CAMERA_DIR.mkdir(parents=True, exist_ok=True)
    INCOMING_DIR.mkdir(parents=True, exist_ok=True)

    STOP_FILE.unlink(missing_ok=True)

    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--camera-window",
        "--camera-index",
        str(CAMERA_INDEX)
    ]

    return subprocess.Popen(
        command,
        cwd=str(BASE_DIR),
        creationflags=(
            subprocess.CREATE_NEW_PROCESS_GROUP
            if os.name == "nt"
            else 0
        )
    )


def stop_camera_process():

    CAMERA_DIR.mkdir(parents=True, exist_ok=True)

    STOP_FILE.write_text(
        "stop",
        encoding="utf-8"
    )

    process = st.session_state.get("camera_process")

    if process is not None and process.poll() is None:

        try:
            process.terminate()
            process.wait(timeout=3)

        except Exception:

            try:
                process.kill()
            except Exception:
                pass

    st.session_state["camera_process"] = None
    st.session_state["camera_running"] = False


# =========================================================
# RESULTS DISPLAY
# =========================================================

def display_results(
    image,
    result,
    best_threshold,
    heat_map_min,
    heat_map_max,
    heatmap_multiplier,
    overlay_alpha
):

    st.markdown("---")

    if result["is_anomaly"]:
        st.error("🚨 DEFECT / ANOMALY DETECTED")
    else:
        st.success("✅ FABRIC OK / NORMAL")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📷 Captured Fabric")
        st.image(image, use_container_width=True)

    display_max = max(
        heat_map_max * heatmap_multiplier,
        1e-12
    )

    with col2:

        st.subheader("🔥 Anomaly Heatmap")

        fig, ax = plt.subplots(figsize=(5, 5))

        im = ax.imshow(
            result["heat_map"],
            cmap="jet",
            vmin=heat_map_min,
            vmax=display_max
        )

        ax.axis("off")

        ax.set_title(
            f"Score: {result['normalized_score']:.4f}"
        )

        fig.colorbar(
            im,
            ax=ax,
            fraction=0.046,
            pad=0.04
        )

        st.pyplot(fig)
        plt.close(fig)

    with col3:

        st.subheader("🎯 Segmentation Mask")

        fig, ax = plt.subplots(figsize=(5, 5))

        ax.imshow(
            result["segmentation_mask"],
            cmap="gray",
            vmin=0,
            vmax=1
        )

        ax.axis("off")

        st.pyplot(fig)
        plt.close(fig)

    st.subheader("📊 Inspection Details")

    m1, m2, m3, m4 = st.columns(4)

    m1.metric(
        "Anomaly Score",
        f"{result['anomaly_score']:.6f}"
    )

    m2.metric(
        "Threshold",
        f"{best_threshold:.6f}"
    )

    m3.metric(
        "Normalized Score",
        f"{result['normalized_score']:.4f}"
    )

    m4.metric(
        "Classification",
        "NOK / DEFECT"
        if result["is_anomaly"]
        else "OK"
    )

    st.subheader("🖼️ Heatmap Overlay")

    original_resized = cv2.resize(
        np.asarray(image.convert("RGB")),
        (128, 128)
    )

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.imshow(original_resized)

    ax.imshow(
        result["heat_map"],
        cmap="jet",
        alpha=overlay_alpha,
        vmin=heat_map_min,
        vmax=display_max
    )

    ax.axis("off")
    ax.set_title("Fabric Anomaly Overlay")

    st.pyplot(fig)
    plt.close(fig)


# =========================================================
# SESSION STATE
# =========================================================

def initialize_session_state():

    defaults = {
        "camera_process": None,
        "camera_running": False,
        "last_result": None,
        "last_capture_time": None,
        "last_processed_file": None,
        "camera_error": None
    }

    for key, value in defaults.items():

        if key not in st.session_state:
            st.session_state[key] = value


# =========================================================
# MAIN STREAMLIT DASHBOARD
# =========================================================

def main():

    initialize_session_state()

    st.title("🔍 Fabric Anomaly Detection Dashboard")

    st.caption(
        "ResNet-50 + Autoencoder | Separate full-screen camera | "
        "Automatic inspection every 5 seconds"
    )

    # -----------------------------------------------------
    # MODEL SELECTION
    # -----------------------------------------------------

    st.sidebar.header("🗂️ Model Selection")

    base_models_path = st.sidebar.text_input(
        "Models Base Path",
        value="models"
    )

    model_folders = get_model_folders(
        base_models_path
    )

    if not model_folders:

        st.warning(
            f"No valid model folders found in `{base_models_path}`."
        )

        st.code(
            "models/\n"
            "└── carpet_model/\n"
            "    ├── anomaly_model_complete.pth\n"
            "    └── threshold_config.json"
        )

        return

    selected_model = st.sidebar.selectbox(
        "Select Model",
        options=model_folders
    )

    model_folder_path = os.path.join(
        base_models_path,
        selected_model
    )

    st.sidebar.success(
        f"Selected: {selected_model}"
    )

    # -----------------------------------------------------
    # LOAD MODEL
    # -----------------------------------------------------

    try:

        with st.spinner("Loading AI model..."):

            (
                backbone,
                model,
                transform,
                threshold_config
            ) = load_models(model_folder_path)

    except Exception as error:

        st.error(f"Could not load model: {error}")
        return

    st.sidebar.success(
        "AI model loaded successfully."
    )

    # -----------------------------------------------------
    # THRESHOLD SETTINGS
    # -----------------------------------------------------

    st.sidebar.header("⚙️ Settings")
    st.sidebar.subheader("Threshold Configuration")

    default_threshold = float(
        threshold_config.get("threshold", 0.30)
    )

    use_custom_threshold = st.sidebar.checkbox(
        "Use Custom Threshold",
        value=False,
        key=f"use_custom_threshold_{selected_model}"
    )

    if use_custom_threshold:

        best_threshold = st.sidebar.slider(
            "Anomaly Threshold",
            min_value=0.000001,
            max_value=0.5,
            value=min(
                max(default_threshold, 0.000001),
                0.5
            ),
            format="%.6f",
            key=f"threshold_slider_{selected_model}"
        )

        if st.sidebar.button(
            "💾 Save as Default Threshold",
            key=f"save_threshold_{selected_model}"
        ):

            try:

                threshold_config["threshold"] = float(
                    best_threshold
                )

                threshold_path = os.path.join(
                    model_folder_path,
                    "threshold_config.json"
                )

                with open(
                    threshold_path,
                    "w",
                    encoding="utf-8"
                ) as file:

                    json.dump(
                        threshold_config,
                        file,
                        indent=4
                    )

                st.sidebar.success("Threshold saved.")

            except Exception as error:

                st.sidebar.error(
                    f"Could not save threshold: {error}"
                )

    else:

        best_threshold = default_threshold

        st.sidebar.info(
            f"Default threshold: {best_threshold:.6f}"
        )

    # -----------------------------------------------------
    # HEATMAP SETTINGS
    # -----------------------------------------------------

    st.sidebar.subheader("Heatmap Display Range")

    heat_map_min = float(
        threshold_config.get("min_error", 0.0)
    )

    heat_map_max = float(
        threshold_config.get("max_error", 1.0)
    )

    if st.sidebar.checkbox(
        "Customize Heatmap Range",
        value=False
    ):

        heat_map_min = st.sidebar.number_input(
            "Min Value",
            value=heat_map_min,
            format="%.6f"
        )

        heat_map_max = st.sidebar.number_input(
            "Max Value",
            value=heat_map_max,
            format="%.6f"
        )

    heatmap_multiplier = st.sidebar.slider(
        "Heatmap Intensity Multiplier",
        min_value=1,
        max_value=20,
        value=10
    )

    overlay_alpha = st.sidebar.slider(
        "Heatmap Opacity",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05
    )

    # -----------------------------------------------------
    # MODEL STATISTICS
    # -----------------------------------------------------

    with st.sidebar.expander("📊 Model Statistics"):

        st.write(f"**Model:** {selected_model}")

        for key in [
            "mean_error",
            "std_error",
            "min_error",
            "max_error"
        ]:

            if key in threshold_config:

                st.write(
                    f"**{key.replace('_', ' ').title()}:** "
                    f"{float(threshold_config[key]):.6f}"
                )

        st.write(
            f"**Current Threshold:** {best_threshold:.6f}"
        )

        if "training_info" in threshold_config:

            st.write("**Training Info:**")

            for key, value in threshold_config[
                "training_info"
            ].items():

                st.write(f"- {key}: {value}")

    # -----------------------------------------------------
    # CAMERA CONTROLS
    # -----------------------------------------------------

    st.markdown("---")
    st.header("📷 Camera Inspection")

    st.write(
        "Start Camera par separate full-screen window khulegi. "
        "Fabric ko yellow guide ke andar position karein."
    )

    start_col, stop_col, status_col = st.columns([1, 1, 2])

    with start_col:

        start_clicked = st.button(
            "▶️ Start Camera",
            disabled=st.session_state["camera_running"],
            use_container_width=True
        )

    with stop_col:

        stop_clicked = st.button(
            "⏹️ Stop Camera",
            disabled=not st.session_state["camera_running"],
            use_container_width=True
        )

    with status_col:

        status = read_camera_status()

        st.write(
            f"**Camera status:** {status.get('status', 'unknown')}"
        )

        st.caption(status.get("message", ""))

    if start_clicked:

        try:

            st.session_state["camera_process"] = (
                start_camera_process()
            )

            st.session_state["camera_running"] = True
            st.session_state["last_processed_file"] = None
            st.session_state["last_result"] = None
            st.session_state["camera_error"] = None

            st.rerun()

        except Exception as error:

            st.session_state["camera_error"] = str(error)

    if stop_clicked:

        stop_camera_process()
        st.rerun()

    if st.session_state["camera_error"]:
        st.error(st.session_state["camera_error"])

    process = st.session_state.get("camera_process")

    if (
        st.session_state["camera_running"]
        and process is not None
        and process.poll() is not None
    ):

        st.session_state["camera_running"] = False

        st.warning(
            "Camera window has closed. Press Start Camera to reopen it."
        )

    st.info(
        "Camera har 5 seconds mein yellow guide ka crop capture karti hai. "
        "Basic blank-frame rejection enabled hai. Fabric ke actual edges "
        "guide ke andar hain ya nahi, iski reliable validation abhi "
        "included nahi hai."
    )

    # -----------------------------------------------------
    # AUTOMATIC INSPECTION
    # -----------------------------------------------------

    @st.fragment(run_every=1)
    def inspect_new_captures():

        INCOMING_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        files = sorted(
            INCOMING_DIR.glob("capture_*.jpg"),
            key=lambda path: path.stat().st_mtime
        )

        if not files:

            st.caption(
                "Waiting for the next camera capture..."
            )

            return

        # Process oldest unprocessed image first.
        image_path = next(
            (
                path for path in files
                if str(path)
                != st.session_state.get("last_processed_file")
            ),
            None
        )

        if image_path is None:

            st.caption(
                "Latest capture has already been analyzed. "
                f"Last inspection: "
                f"{st.session_state.get('last_capture_time', 'N/A')}"
            )

            return

        try:

            with Image.open(image_path) as opened:
                fabric_image = opened.convert("RGB").copy()

            with st.spinner("Analyzing captured fabric..."):

                result = detect_anomaly(
                    fabric_image,
                    backbone,
                    model,
                    transform,
                    best_threshold
                )

            # Save the original captured crop in the matching folder.
            target_dir = (
                NOK_DIR
                if result["is_anomaly"]
                else OK_DIR
            )

            target_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            destination = target_dir / image_path.name

            shutil.copy2(
                image_path,
                destination
            )

            st.session_state["last_result"] = result

            st.session_state["last_processed_file"] = str(
                image_path
            )

            st.session_state["last_capture_time"] = (
                time.strftime("%Y-%m-%d %H:%M:%S")
            )

            classification = (
                "NOK / DEFECT"
                if result["is_anomaly"]
                else "OK"
            )

            st.success(
                f"Classification: {classification} | "
                f"Saved: {destination.relative_to(BASE_DIR)}"
            )

            st.caption(
                "Last inspection: "
                + st.session_state["last_capture_time"]
            )

            display_results(
                fabric_image,
                result,
                best_threshold,
                heat_map_min,
                heat_map_max,
                heatmap_multiplier,
                overlay_alpha
            )

        except Exception as error:

            st.error(
                f"Anomaly analysis failed: {error}"
            )

            st.info(
                "If the error persists, share the complete traceback."
            )

            # Avoid retrying the same failing capture every second.
            st.session_state["last_processed_file"] = str(
                image_path
            )

    inspect_new_captures()

    # -----------------------------------------------------
    # SAVED INSPECTIONS
    # -----------------------------------------------------

    st.markdown("---")
    st.subheader("📁 Saved Inspections")

    ok_count = (
        len(list(OK_DIR.glob("*.jpg")))
        if OK_DIR.exists()
        else 0
    )

    nok_count = (
        len(list(NOK_DIR.glob("*.jpg")))
        if NOK_DIR.exists()
        else 0
    )

    count1, count2 = st.columns(2)

    count1.metric("Saved OK Images", ok_count)
    count2.metric("Saved NOK Images", nok_count)


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    # A separate process runs the OpenCV camera window.
    if "--camera-window" in sys.argv:

        camera_index = CAMERA_INDEX

        if "--camera-index" in sys.argv:

            try:
                index = sys.argv.index("--camera-index")
                camera_index = int(sys.argv[index + 1])

            except (ValueError, IndexError):
                pass

        run_camera_window(camera_index)

    else:
        main()