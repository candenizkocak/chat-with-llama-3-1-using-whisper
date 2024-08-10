import os
import gradio as gr
from groq import Groq

api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)

if not api_key:
    raise ValueError("API key not found. Please set the GROQ_API_KEY environment variable.")

def transcribe_audio(file_path):
    with open(file_path, "rb") as file:
        transcription = client.audio.transcriptions.create(
            file=(file_path, file.read()),
            model="whisper-large-v3",
            response_format="verbose_json",
        )
        return transcription.text

def get_chat_completion(prompt):
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=1,
        max_tokens=1024,
        top_p=1,
        stream=True,
        stop=None,
    )

    response = ""
    for chunk in completion:
        response += chunk.choices[0].delta.content or ""
    return response

def process_input(audio_file, text_input, chat_history):
    if audio_file is not None:
        transcription_text = transcribe_audio(audio_file)
    else:
        transcription_text = text_input

    chat_response = get_chat_completion(transcription_text)
    chat_history.append(("👤", transcription_text))
    chat_history.append(("🤖", chat_response))

    formatted_history = "\n".join([f"{role}: {content}\n" for role, content in chat_history])

    return formatted_history, gr.update(value=None), gr.update(value=''), chat_history

# Create Gradio interface
interface = gr.Interface(
    fn=process_input,
    inputs=[
        gr.Audio(type="filepath", label="Upload Audio or Record"),
        gr.Textbox(lines=2, placeholder="Or type text here", label="Text Input"),
        gr.State([])
    ],
    outputs=[
        gr.Textbox(label="Chat History", lines=20),
        gr.Audio(visible=False),
        gr.Textbox(visible=False),
        gr.State()
    ],
    title="Chat with Llama 3.1-8B With Text or Voice (Whisper Large-v3)",
    description="Upload an audio file or type text to get a chat response based on the transcription.",
    allow_flagging='never'  # Prevent flagging to isolate sessions
)

if __name__ == "__main__":
    interface.launch()
