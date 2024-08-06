import gradio as gr
import whisper

def translator(audio_file):

    model = whisper.load_model("base")
    result = model.transcribe(audio_file)
    transcription = result["text"]

    transcription 

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

