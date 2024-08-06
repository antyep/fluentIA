import gradio as gr
import whisper
from translate import Translator
from dotenv import dotenv_values
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings

config = dotenv_values(".env")

ELEVENLABS_API_KEY = config ("ELEVENLABS_API_KEY")

def translator(audio_file):

    model = whisper.load_model("base")
    result = model.transcribe(audio_file)
    transcription = result["text"]

    en_transcription = Translator(to_lang="en").translate(transcription)

    client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

    client.text_to_speech.convert(
                response = client.text_to_speech.convert(
            voice_id="pNInz6obpgDQGcFmaJgB",
            optimize_streaming_latency="0",
            output_format="mp3_22050_32",
            text=en_transcription,
            model_id="eleven_turbo_v2",
            voice_settings=VoiceSettings(
                stability=0.0,
                similarity_boost=0.0,
                style=0.0,
                use_speaker_boost=True,
            ),
        )
    )

web = gr.Interface(
    fn=translator,
    inputs=gr.Audio(
        sources=["microphone"],
        type="filepath"
    ),
    outputs=[],
    title="Voice Translator",
    description="Voice Translator with IA"
)

web.launch()

