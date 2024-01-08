import torch
import os

import gradio as gr
import pytube as pt
from speechbox import ASRDiarizationPipeline

from string import punctuation
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')

MODEL_NAME = "openai/whisper-large-v3"

device = 0 if torch.cuda.is_available() else "cpu"
HF_TOKEN = os.environ.get("HF_TOKEN")

pipe = ASRDiarizationPipeline.from_pretrained(
    asr_model=MODEL_NAME,
    device=device,
    use_auth_token=HF_TOKEN,
)

def tuple_to_string(start_end_tuple, ndigits=1):
    return str((round(start_end_tuple[0], ndigits), round(start_end_tuple[1], ndigits)))


def format_as_transcription(raw_segments, with_timestamps=False):
    if with_timestamps:
        return "\n\n".join([chunk["speaker"] + " " + tuple_to_string(chunk["timestamp"]) +  chunk["text"] for chunk in raw_segments])
    else:
        return "\n\n".join([chunk["speaker"] + chunk["text"] for chunk in raw_segments])


def transcribe(file_upload, with_timestamps):
    if file_upload is None:
        raise gr.Error("No audio file submitted! Please upload an audio file before submitting your request.")
    raw_segments = pipe(file_upload)
    transcription = format_as_transcription(raw_segments, with_timestamps=with_timestamps)
    return f"##Transcription:\n\n###JSON Format:\n{raw_segments}\n\n###Transcribed Script:\n{transcription}"


def _return_yt_html_embed(yt_url):
    video_id = yt_url.split("?v=")[-1]
    HTML_str = (
        f'<center> <iframe width="500" height="320" src="https://www.youtube.com/embed/{video_id}"> </iframe>'
        " </center>"
    )
    return HTML_str


def yt_transcribe(yt_url, with_timestamps):
    yt = pt.YouTube(yt_url)
    html_embed_str = _return_yt_html_embed(yt_url)
    stream = yt.streams.filter(only_audio=True)[0]
    stream.download(filename="audio.mp3")

    text = pipe("audio.mp3")

    return html_embed_str, format_as_transcription(text, with_timestamps=with_timestamps)


demo = gr.Blocks()

mf_transcribe = gr.Interface(
    fn=transcribe,
    inputs=[
        gr.Audio(sources=["upload", "microphone"], type="filepath"),
        gr.Checkbox(label="With timestamps?", value=True),
    ],
    outputs="text",
    #layout="horizontal",
    theme="huggingface",
    title="WSIB Customer Service Transcription",
    description=(
        "Transcribe audio files with speaker diarization using [🤗 Speechbox](https://github.com/huggingface/speechbox/). "
        "Demo uses the pre-trained checkpoint [Whisper Small](https://huggingface.co/openai/whisper-small) for the ASR "
        "transcriptions and [pyannote.audio](https://huggingface.co/pyannote/speaker-diarization) to label the speakers."
        "\n\n"
        "Check out the repo here: https://github.com/huggingface/speechbox/"
    ),
    #examples=[
    #    ["./processed.wav", True],
    #    ["./processed.wav", False],
    #],
    allow_flagging="never",
)

yt_transcribe = gr.Interface(
    fn=yt_transcribe,
    inputs=[
        gr.Textbox(lines=1, placeholder="Paste the URL to a YouTube video here", label="YouTube URL"),
        gr.Checkbox(label="With timestamps?", value=True),
    ],
    outputs=["html", "text"],
    #layout="horizontal",
    theme="huggingface",
    title="Whisper Speaker Diarization: Transcribe YouTube",
    description=(
        "Transcribe YouTube videos with speaker diarization using [🤗 Speechbox](https://github.com/huggingface/speechbox/). "
        "Demo uses the pre-trained checkpoint [Whisper Tiny](https://huggingface.co/openai/whisper-tiny) for the ASR "
        "transcriptions and [pyannote.audio](https://huggingface.co/pyannote/speaker-diarization) to label the speakers."
        "\n\n"
        "Check out the repo here: https://github.com/huggingface/speechbox/"
    ),
    examples=[
        ["https://www.youtube.com/watch?v=9dAWIPixYxc", True],
        ["https://www.youtube.com/watch?v=9dAWIPixYxc", False],
    ],
    allow_flagging="never",
)

with demo:
    gr.TabbedInterface([mf_transcribe, yt_transcribe], ["Transcribe Audio", "Transcribe YouTube"])

demo.launch()