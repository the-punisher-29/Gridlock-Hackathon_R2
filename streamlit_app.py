import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import streamlit as st
import cv2
import tempfile
import pandas as pd
from PIL import Image

from app.rules.zone_config import ZoneConfig
from app.pipeline import TrafficViolationPipeline
from app.db import Session, Violation

st.set_page_config(page_title="Traffic Violation Detector", layout="wide")
st.title("🚦 Automated Traffic Violation Detection")

# ------------------------------------------------------------------
# Session state init — MUST run before anything else touches these keys
# ------------------------------------------------------------------
DEFAULTS = {
    "sample_frame_path": None,
    "video_path": None,
    "pipeline_done": False,
    "output_video_path": None,
}
for key, default_value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

tab1, tab2, tab3 = st.tabs(["1️⃣ Upload & Calibrate", "2️⃣ Run Detection", "3️⃣ Results & Reports"])

# ------------------------------------------------------------------
# TAB 1 — Upload video + set stop line (no canvas, numeric input version)
# ------------------------------------------------------------------
with tab1:
    st.subheader("Upload Traffic Footage")
    uploaded_file = st.file_uploader("Upload a video (mp4/avi/mov)", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_video.write(uploaded_file.read())
        temp_video.flush()
        st.session_state["video_path"] = temp_video.name

        cap = cv2.VideoCapture(st.session_state["video_path"])
        ret, frame = cap.read()
        cap.release()

        if ret:
            frame_path = os.path.join(tempfile.gettempdir(), "sample_frame.jpg")
            cv2.imwrite(frame_path, frame)
            st.session_state["sample_frame_path"] = frame_path
            st.success("Video uploaded successfully.")
        else:
            st.error("Could not read a frame from this video. Try a different file.")

    if st.session_state.get("sample_frame_path"):
        st.subheader("Set Stop Line Coordinates")
        st.image(st.session_state["sample_frame_path"], caption="Reference frame")

        img = Image.open(st.session_state["sample_frame_path"])
        img_w, img_h = img.size
        st.caption(f"Frame size: {img_w} x {img_h} px")

        col1, col2 = st.columns(2)
        with col1:
            x1 = st.number_input("Point 1 - X", min_value=0, max_value=img_w, value=int(img_w * 0.1))
            y1 = st.number_input("Point 1 - Y", min_value=0, max_value=img_h, value=int(img_h * 0.7))
        with col2:
            x2 = st.number_input("Point 2 - X", min_value=0, max_value=img_w, value=int(img_w * 0.9))
            y2 = st.number_input("Point 2 - Y", min_value=0, max_value=img_h, value=int(img_h * 0.7))

        light_state = st.selectbox("Simulate traffic light state", ["red", "yellow", "green", "unknown"], index=0)

        preview = cv2.imread(st.session_state["sample_frame_path"])
        cv2.line(preview, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)
        preview_path = os.path.join(tempfile.gettempdir(), "preview_line.jpg")
        cv2.imwrite(preview_path, preview)
        st.image(preview_path, caption="Preview of stop line placement")

        if st.button("Save Zone Configuration"):
            zc = ZoneConfig()
            zc.set_stop_line((x1, y1), (x2, y2))
            zc.set_light_state(light_state)
            zc.save()
            st.success(f"Stop line saved: ({x1},{y1}) -> ({x2},{y2}), light={light_state}")
    else:
        st.info("Upload a video above to continue.")

# ------------------------------------------------------------------
# TAB 2 — Run pipeline
# ------------------------------------------------------------------
with tab2:
    st.subheader("Run Violation Detection")

    if not st.session_state.get("video_path"):
        st.info("Upload a video in Tab 1 first.")
    else:
        sample_rate = st.slider("Process every Nth frame (higher = faster, less accurate)", 1, 10, 2)
        camera_id = st.text_input("Camera / Location ID", value="cam1")

        if st.button("▶️ Run Detection Pipeline"):
            output_path = os.path.join(tempfile.gettempdir(), "output_annotated.mp4")

            with st.spinner("Processing video... this may take a few minutes."):
                pipeline = TrafficViolationPipeline(camera_id=camera_id)
                pipeline.process_video(
                    video_path=st.session_state["video_path"],
                    output_path=output_path,
                    sample_every_n=sample_rate
                )

            st.session_state["pipeline_done"] = True
            st.session_state["output_video_path"] = output_path
            st.success("Done! Check the Results tab.")

        if st.session_state.get("pipeline_done") and st.session_state.get("output_video_path"):
            st.subheader("Annotated Output Preview")
            st.video(st.session_state["output_video_path"])

# ------------------------------------------------------------------
# TAB 3 — Results & Reports
# ------------------------------------------------------------------
with tab3:
    st.subheader("Violation Records")

    session = Session()
    violations = session.query(Violation).order_by(Violation.timestamp.desc()).all()
    session.close()

    if not violations:
        st.info("No violations recorded yet. Run the pipeline in Tab 2.")
    else:
        df = pd.DataFrame([{
            "ID": v.id,
            "Timestamp": v.timestamp,
            "Camera": v.camera_id,
            "Vehicle Track ID": v.tracker_id,
            "Violation Type": v.violation_type,
            "Confidence": round(v.confidence, 2),
            "Plate Number": v.plate_number or "Unreadable",
            "Plate Confidence": round(v.plate_confidence, 2) if v.plate_confidence else 0,
            "Evidence Path": v.evidence_image_path
        } for v in violations])

        col1, col2 = st.columns(2)
        with col1:
            violation_filter = st.multiselect("Filter by Violation Type", options=df["Violation Type"].unique().tolist())
        with col2:
            plate_search = st.text_input("Search Plate Number")

        filtered_df = df.copy()
        if violation_filter:
            filtered_df = filtered_df[filtered_df["Violation Type"].isin(violation_filter)]
        if plate_search:
            filtered_df = filtered_df[filtered_df["Plate Number"].str.contains(plate_search.upper(), na=False)]

        st.dataframe(filtered_df.drop(columns=["Evidence Path"]), use_container_width=True)

        st.subheader("Violation Summary")
        summary = df["Violation Type"].value_counts()
        st.bar_chart(summary)

        st.subheader("Evidence Images")
        for _, row in filtered_df.head(20).iterrows():
            if os.path.exists(row["Evidence Path"]):
                col_img, col_info = st.columns([1, 2])
                with col_img:
                    st.image(row["Evidence Path"], width=300)
                with col_info:
                    st.write(f"**Violation:** {row['Violation Type']}")
                    st.write(f"**Plate:** {row['Plate Number']}")
                    st.write(f"**Confidence:** {row['Confidence']}")
                    st.write(f"**Time:** {row['Timestamp']}")
                st.divider()

        csv = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download CSV Report", csv, "violations_report.csv", "text/csv")