import gradio as gr
import whisper
from translate import Translator

def translator(audio_file):

    model = whisper.load_model("base")
    result = model.transcribe(audio_file)
    transcription = result["text"]

    en_transcription = Translator(to_lang="en").translate(transcription)

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

