import gradio as gr
import whisper
from translate import Translator
from dotenv import dotenv_values
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings

config = dotenv_values(".env")

ELEVENLABS_API_KEY = config["ELEVENLABS_API_KEY"]

def translator(audio_file):

    try:
        model = whisper.load_model("base")
        result = model.transcribe(audio_file, language="English", fp16=False)

        if result is None or "text" not in result:
            raise ValueError("Error while transcribing text")

        transcription = result["text"]
    except Exception as e:
        raise gr.Error(f"There was an error while transcribing the text: {str(e)}")

    print(f"Original text: {transcription}")

    try:
        es_transcription = Translator(from_lang="en", to_lang="es").translate(transcription)
        it_transcription = Translator(from_lang="en", to_lang="it").translate(transcription)
        fr_transcription = Translator(from_lang="en", to_lang="fr").translate(transcription)
        ja_transcription = Translator(from_lang="en", to_lang="ja").translate(transcription)
        de_transcription = Translator(from_lang="en", to_lang="de").translate(transcription)

    except Exception as e:
        raise gr.Error(f"There was a mistake while creating the text: {str(e)}")

    try:
        en_save_file_path = text_to_speech(es_transcription, "es")
        it_save_file_path = text_to_speech(it_transcription, "it")
        fr_save_file_path = text_to_speech(fr_transcription, "fr")
        ja_save_file_path = text_to_speech(ja_transcription, "ja")
        de_save_file_path = text_to_speech(de_transcription, "de")

    except Exception as e:
        raise gr.Error(f"There was a mistake during audio creation: {str(e)}")

    return en_save_file_path, it_save_file_path, fr_save_file_path, ja_save_file_path, de_save_file_path

def text_to_speech(text: str, language: str) -> str:

    try:
        client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

        response = client.text_to_speech.convert(
            voice_id="2EiwWnXFnvU5JabPnv8n", 
            optimize_streaming_latency="0",
            output_format="mp3_22050_32",
            text=text,
            model_id="eleven_turbo_v2",
            voice_settings=VoiceSettings(
                stability=0.0,
                similarity_boost=0.0,
                style=0.0,
                use_speaker_boost=True,
            ),
        )

        save_file_path = f"{language}.mp3"

        with open(save_file_path, "wb") as f:
            for chunk in response:
                if chunk:
                    f.write(chunk)

    except Exception as e:
        raise gr.Error(f"There was an error while creating audio: {str(e)}")

    return save_file_path

web = gr.Interface(
    fn=translator,
    inputs=gr.Audio(
        sources=["microphone"],
        type="filepath",
        label="English"
    ),
    outputs=[
        gr.Audio(label="Spanish"),
        gr.Audio(label="Italiano"),
        gr.Audio(label="Française"),
        gr.Audio(label="Japanese"),
        gr.Audio(label="German")
    ],
    title="Voice Translator",
    description="Voice Translator from spoken English with IA"
)

web.launch()
