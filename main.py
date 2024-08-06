import gradio as gr

def translator():
    pass

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

