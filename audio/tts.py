import asyncio
import edge_tts


VOICE = "vi-VN-HoaiMyNeural"


def speak(text: str, output_path: str = "feedback_audio.mp3") -> str:
    """Convert text to speech using Edge-TTS. Returns path to output file."""
    asyncio.run(_synthesize(text, output_path))
    return output_path


async def _synthesize(text: str, output_path: str) -> None:
    communicate = edge_tts.Communicate(text, VOICE)
    await communicate.save(output_path)
