# Enhancing Customer Experience with AI-Driven Insights

This Streamlit dashboard uses OpenCV and the included OpenVINO YOLOv8 model to
detect and anonymously track people in a recorded video or an IP camera stream.
Analysis covers the entire frame; no zones need to be selected or configured.

## Run

From the project directory, activate the existing virtual environment or create
one, install the dependencies, and start Streamlit:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Use **Recorded Video** to upload a supported video file, or select
**IP Camera / Live Stream**, enter an RTSP/RTSPS or HTTP/HTTPS (including
MJPEG) stream URL, and select **Connect Stream**. Stream URLs are entered into
a password-masked field and are not written to logs or connection errors. For
deployment, `CAMERA_STREAM_URL` can be supplied through the process environment
instead of typing the URL into the app. Do not commit stream credentials or
place them in source files.

The live dashboard displays the stream alongside current and total customer
counts, entries/exits, dwell-time estimates, peak activity, busy/low-traffic
periods, a movement heatmap, and high-volume alerts. Stream and tracking
availability depend on the camera/network, OpenCV video backend, and model
detections. Tracking IDs are temporary and anonymous; brief occlusion or
missed detections can affect counts and dwell-time estimates.

The existing processed-video preview, heatmap, summary, and generated CSV/text
reports remain available for recorded-video analysis. Analysis output is saved
under `output/`.
