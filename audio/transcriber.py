import re
from dataclasses import dataclass

import whisper


FILLER_WORDS = {"uh", "um", "erm", "ah", "like", "you know", "basically", "actually", "literally"}


@dataclass
class TranscriptResult:
    text: str
    filler_words: dict[str, int]
    filler_count: int
    word_count: int
    wpm: float
    duration_seconds: float


def transcribe(audio_path: str, model_size: str = "base") -> TranscriptResult:
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path, language="vi", fp16=False)

    text: str = result["text"].strip()
    duration: float = result["segments"][-1]["end"] if result["segments"] else 0.0

    words = text.lower().split()
    word_count = len(words)
    wpm = round(word_count / (duration / 60), 1) if duration > 0 else 0.0

    filler_counts: dict[str, int] = {}
    for word in FILLER_WORDS:
        count = len(re.findall(r"\b" + re.escape(word) + r"\b", text.lower()))
        if count > 0:
            filler_counts[word] = count

    return TranscriptResult(
        text=text,
        filler_words=filler_counts,
        filler_count=sum(filler_counts.values()),
        word_count=word_count,
        wpm=wpm,
        duration_seconds=round(duration, 1),
    )
