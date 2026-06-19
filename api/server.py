import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from vision.frame_extractor import extract_frames
from vision.face_analyzer import analyze_frames
from llm.feedback_generator import generate_feedback
from audio.transcriber import transcribe

app = FastAPI(title="InterviewCoach AI", version="0.1.0")

HTML_PAGE = """<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>InterviewCoach AI</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0f1117; color: #e2e8f0; min-height: 100vh; }
    .container { max-width: 800px; margin: 0 auto; padding: 40px 20px; }
    h1 { font-size: 2rem; font-weight: 700; margin-bottom: 8px; background: linear-gradient(135deg, #667eea, #764ba2); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .subtitle { color: #718096; margin-bottom: 40px; }
    .upload-zone { border: 2px dashed #4a5568; border-radius: 16px; padding: 48px; text-align: center; transition: all 0.2s; cursor: pointer; background: #1a202c; }
    .upload-zone:hover { border-color: #667eea; background: #1e2535; }
    .upload-zone input { display: none; }
    .upload-icon { font-size: 3rem; margin-bottom: 16px; }
    .upload-text { color: #a0aec0; margin-bottom: 8px; }
    .upload-hint { color: #4a5568; font-size: 0.85rem; }
    .btn { display: inline-block; padding: 12px 32px; border-radius: 8px; border: none; cursor: pointer; font-size: 1rem; font-weight: 600; transition: all 0.2s; }
    .btn-primary { background: linear-gradient(135deg, #667eea, #764ba2); color: white; width: 100%; margin-top: 20px; }
    .btn-primary:hover { opacity: 0.9; transform: translateY(-1px); }
    .btn-primary:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
    .results { margin-top: 40px; display: none; }
    .metrics-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; margin-bottom: 24px; }
    .metric-card { background: #1a202c; border: 1px solid #2d3748; border-radius: 12px; padding: 20px; }
    .metric-label { color: #718096; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px; }
    .metric-value { font-size: 2rem; font-weight: 700; color: #e2e8f0; }
    .metric-unit { font-size: 0.9rem; color: #718096; font-weight: 400; }
    .feedback-box { background: #1a202c; border: 1px solid #2d3748; border-radius: 12px; padding: 28px; white-space: pre-wrap; line-height: 1.7; color: #cbd5e0; }
    .status { padding: 16px; border-radius: 8px; margin-top: 16px; text-align: center; font-weight: 500; }
    .status.loading { background: #1e3a5f; color: #63b3ed; border: 1px solid #2b6cb0; }
    .status.error { background: #3d1a1a; color: #fc8181; border: 1px solid #742a2a; }
    .section-title { font-size: 1.1rem; font-weight: 600; color: #a0aec0; margin-bottom: 16px; text-transform: uppercase; letter-spacing: 0.05em; }
    .file-name { margin-top: 12px; color: #667eea; font-size: 0.9rem; }
    .progress { display: none; margin-top: 16px; }
    .progress-bar { height: 4px; background: #2d3748; border-radius: 2px; overflow: hidden; }
    .progress-fill { height: 100%; background: linear-gradient(90deg, #667eea, #764ba2); border-radius: 2px; animation: progress 2s ease-in-out infinite; }
    @keyframes progress { 0% { width: 0%; } 50% { width: 70%; } 100% { width: 90%; } }
  </style>
</head>
<body>
  <div class="container">
    <h1>🎤 InterviewCoach AI</h1>
    <p class="subtitle">Upload video phỏng vấn → AI phân tích facial expression + nội dung → feedback chi tiết</p>

    <div class="upload-zone" onclick="document.getElementById('videoInput').click()">
      <div class="upload-icon">🎬</div>
      <p class="upload-text">Click để chọn video phỏng vấn</p>
      <p class="upload-hint">Hỗ trợ MP4, MOV, AVI, WEBM</p>
      <input type="file" id="videoInput" accept="video/*" onchange="handleFileSelect(event)">
      <div class="file-name" id="fileName"></div>
    </div>

    <button class="btn btn-primary" id="analyzeBtn" onclick="analyzeVideo()" disabled>
      Phân tích phỏng vấn →
    </button>

    <div class="progress" id="progress">
      <div class="progress-bar"><div class="progress-fill"></div></div>
    </div>

    <div id="status"></div>

    <div class="results" id="results">
      <div class="section-title">📊 Metrics</div>
      <div class="metrics-grid">
        <div class="metric-card">
          <div class="metric-label">Eye Contact</div>
          <div class="metric-value" id="metricEye">—</div>
          <div class="metric-unit">% thời gian</div>
        </div>
        <div class="metric-card">
          <div class="metric-label">Nụ Cười</div>
          <div class="metric-value" id="metricSmile">—</div>
          <div class="metric-unit">% thời gian</div>
        </div>
        <div class="metric-card">
          <div class="metric-label">Tốc độ nói</div>
          <div class="metric-value" id="metricWpm">—</div>
          <div class="metric-unit">từ / phút</div>
        </div>
        <div class="metric-card">
          <div class="metric-label">Filler words</div>
          <div class="metric-value" id="metricFiller">—</div>
          <div class="metric-unit">lần</div>
        </div>
      </div>

      <div class="section-title">💬 Transcript</div>
      <div class="feedback-box" id="transcriptBox" style="margin-bottom:24px; max-height:200px; overflow-y:auto;"></div>

      <div class="section-title">🤖 AI Feedback</div>
      <div class="feedback-box" id="feedbackBox"></div>
    </div>
  </div>

  <script>
    let selectedFile = null;

    function handleFileSelect(event) {
      selectedFile = event.target.files[0];
      if (selectedFile) {
        document.getElementById('fileName').textContent = '✓ ' + selectedFile.name;
        document.getElementById('analyzeBtn').disabled = false;
      }
    }

    async function analyzeVideo() {
      if (!selectedFile) return;

      const btn = document.getElementById('analyzeBtn');
      const statusEl = document.getElementById('status');
      const progressEl = document.getElementById('progress');

      btn.disabled = true;
      btn.textContent = 'Đang phân tích...';
      progressEl.style.display = 'block';
      document.getElementById('results').style.display = 'none';
      statusEl.innerHTML = '<div class="status loading">⏳ Đang xử lý video... (có thể mất 1-3 phút tùy độ dài video)</div>';

      const formData = new FormData();
      formData.append('video', selectedFile);

      try {
        const response = await fetch('/analyze', { method: 'POST', body: formData });
        const data = await response.json();

        if (!response.ok) {
          throw new Error(data.detail || 'Lỗi server');
        }

        document.getElementById('metricEye').textContent = data.eye_contact_percent + '%';
        document.getElementById('metricSmile').textContent = data.smile_rate_percent + '%';
        document.getElementById('metricWpm').textContent = data.wpm;
        document.getElementById('metricFiller').textContent = data.filler_count;
        document.getElementById('transcriptBox').textContent = data.transcript || '(không có transcript)';
        document.getElementById('feedbackBox').textContent = data.feedback;

        statusEl.innerHTML = '';
        document.getElementById('results').style.display = 'block';
      } catch (err) {
        statusEl.innerHTML = '<div class="status error">❌ ' + err.message + '</div>';
      } finally {
        btn.disabled = false;
        btn.textContent = 'Phân tích phỏng vấn →';
        progressEl.style.display = 'none';
      }
    }
  </script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE


@app.post("/analyze")
async def analyze(video: UploadFile = File(...)):
    if not video.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File phải là video")

    with tempfile.TemporaryDirectory() as tmp_dir:
        video_path = os.path.join(tmp_dir, video.filename)
        with open(video_path, "wb") as f:
            content = await video.read()
            f.write(content)

        try:
            frames = extract_frames(video_path, sample_rate=5)
            face_metrics = analyze_frames(frames)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Lỗi phân tích video: {e}")

        audio_path = os.path.join(tmp_dir, "audio.wav")
        try:
            import subprocess
            subprocess.run(
                ["ffmpeg", "-i", video_path, "-vn", "-ar", "16000", "-ac", "1", audio_path, "-y", "-loglevel", "error"],
                check=True,
            )
            transcript_result = transcribe(audio_path)
        except Exception as e:
            from audio.transcriber import TranscriptResult
            transcript_result = TranscriptResult(
                text="(không trích xuất được audio)",
                filler_words={},
                filler_count=0,
                word_count=0,
                wpm=0.0,
                duration_seconds=0.0,
            )

        try:
            feedback = generate_feedback(transcript_result, face_metrics)
        except Exception as e:
            feedback = f"(Lỗi tạo feedback: {e})"

    return JSONResponse({
        "eye_contact_percent": face_metrics.eye_contact_percent,
        "smile_rate_percent": face_metrics.smile_rate_percent,
        "frames_analyzed": face_metrics.frames_analyzed,
        "frames_with_face": face_metrics.frames_with_face,
        "transcript": transcript_result.text,
        "wpm": transcript_result.wpm,
        "filler_count": transcript_result.filler_count,
        "filler_words": transcript_result.filler_words,
        "feedback": feedback,
    })
