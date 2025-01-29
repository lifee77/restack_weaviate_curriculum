import os
from restack_ai.function import function, log
from pydantic import BaseModel
from google import genai

class AudioInput(BaseModel):
    text: str

class AudioOutput(BaseModel):
    audio_file: str

@function.defn()
async def text_to_audio(input: AudioInput) -> AudioOutput:
    """
    Convert text summary into an audio file using Gemini's text-to-speech capabilities.
    """
    try:
        log.info("Starting text-to-audio conversion...")
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

        response = client.models.generate_content(
            model='gemini-1.5-flash',
            contents=[input.text],
            config={
                "response_mime_type": "audio/wav"
            }
        )

        # Save the audio file
        audio_filename = "summary_audio.wav"
        with open(audio_filename, "wb") as audio_file:
            audio_file.write(response.audio)  # Assuming response.audio contains the audio bytes

        log.info(f"Audio file saved: {audio_filename}")
        return AudioOutput(audio_file=audio_filename)

    except Exception as e:
        log.error("Error in text_to_audio function", error=e)
        raise e
