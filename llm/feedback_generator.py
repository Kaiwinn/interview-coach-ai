import os
import anthropic
from dotenv import load_dotenv

from audio.transcriber import TranscriptResult
from vision.face_analyzer import FaceMetrics

load_dotenv()

MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """Bạn là interview coach chuyên nghiệp. Phân tích buổi phỏng vấn và đưa ra feedback cụ thể, có thể hành động ngay.
Phản hồi bằng tiếng Việt, ngắn gọn, thực tế."""

FEEDBACK_TEMPLATE = """
## Dữ liệu phỏng vấn

**Giọng nói & Nội dung:**
- Transcript: {transcript}
- Tốc độ nói: {wpm} từ/phút (lý tưởng: 120–160)
- Từ filler dùng: {filler_count} lần → {filler_words}
- Thời lượng: {duration}s

**Ngôn ngữ cơ thể:**
- Eye contact: {eye_contact}% (lý tưởng: >70%)
- Nụ cười: {smile_rate}% thời gian
- Frames phân tích: {frames_analyzed} ({frames_with_face} có khuôn mặt)

---

Hãy đưa ra feedback theo cấu trúc:
1. **Điểm mạnh** (2–3 điểm cụ thể)
2. **Cần cải thiện** (2–3 điểm với hành động cụ thể)
3. **Score tổng** (0–10) và 1 câu tóm tắt
"""


def generate_feedback(transcript: TranscriptResult, metrics: FaceMetrics) -> str:
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    prompt = FEEDBACK_TEMPLATE.format(
        transcript=transcript.text[:2000],
        wpm=transcript.wpm,
        filler_count=transcript.filler_count,
        filler_words=transcript.filler_words,
        duration=transcript.duration_seconds,
        eye_contact=metrics.eye_contact_percent,
        smile_rate=metrics.smile_rate_percent,
        frames_analyzed=metrics.frames_analyzed,
        frames_with_face=metrics.frames_with_face,
    )

    message = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )

    return message.content[0].text
