#!/usr/bin/env python
#
# Whisper Integration: Audio To Text support
# *EXPERIMENTAL*
#

try:
    import streamlit as st
    from dotenv import dotenv_values
    from threading import RLock

    with st.spinner("** LOADING TORCH AND TRANSFORMERS... **"):
        from torchcodec.decoders import AudioDecoder
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
        import matplotlib.pyplot as plt

    with st.spinner("** LOADING INTERFACE... **"):
        # local imports
        import libs.shared.huggingface as hf
        from libs.shared.settings import Properties
        from libs.shared.session import Session
        from libs.shared.utils import detect_accelerator
        from libs.utils.audio_pipelines import resample, waveform, spectrum
except Exception as e:
    print(f"Caught fatal exception: {e}")

# whisper inference settings
INFERENCE_SAMPLE_RATE = 16_000
INFERENCE_CHANNELS = 1

# load environment
config_env: dict = dotenv_values(".env")

# load app settings
config_filename: str = config_env.get("CONFIG_FILE", "parameters.yaml")
appSettings = Properties(config_file=config_filename)

# initialize streamlit session
stSession = Session(st.session_state)

# build streamlit UI
st.set_page_config(
    page_title="🧠 RedHat Agentic AI Assistant",
    initial_sidebar_state="collapsed",
    layout="wide",
)
st.html("assets/whisper.html")

# detect acceleration device
device, dtype = detect_accelerator()

# whisper model
supported_models = appSettings.config_parameters.data.get("whisper").get(
    "supported_models"
)
model = appSettings.config_parameters.whisper.model

# pyplot lock
_pyplot_lock = RLock()


@st.cache_resource
def loadWhisperModel(low_vram: bool, device: str) -> pipeline:
    with st.spinner("**Loading model into memory...**"):
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            mp,
            device_map="auto",
            torch_dtype=dtype,
            low_cpu_mem_usage=low_vram,
            use_safetensors=True,
        )
        # model.to(device)
        processor = AutoProcessor.from_pretrained(
            mp, device_map="auto", use_safetensors=True
        )

        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=dtype,
        )
    return pipe


# populate sidebar options
with st.sidebar:
    # reset function
    def reset_agent():
        st.cache_resource.clear()

    st.header("🛠 Whisper Control Panel")

    st.markdown("** Model Selection **")
    selected_model = st.selectbox(
        label="Select Whisper Model",
        options=supported_models.keys(),
        index=[i for i, x in enumerate(supported_models.keys()) if x == model].pop(),
        on_change=reset_agent,
    )

    st.markdown(f"**Model: `{selected_model}`**")
    st.markdown(f"**Device: `{device}/{dtype}`**")

    with st.expander("Options"):
        low_vram = st.checkbox(
            "Use low VRAM Settings", value=True, on_change=reset_agent
        )

# file uploader
with st.spinner("** GETTING WHISPER MODEL FROM HUGGINGFACE... **"):
    try:
        # download model from huggingface
        mp = hf.downloadFromHuggingFace(
            repo_id=supported_models.get(selected_model),
            local_dir=appSettings.config_parameters.huggingface.local_dir,
            cache_dir=appSettings.config_parameters.huggingface.cache_dir,
            apitoken=appSettings.config_parameters.huggingface.apitoken,
            revision=appSettings.config_parameters.whisper.revision,
        )
    except Exception as e:
        st.markdown(f"🚨 Error downloading model from HuggingFace: {e}")

st.subheader("Upload an Audio File", divider=True)
uploaded_files = st.file_uploader(
    "Upload file(s) or directory",
    accept_multiple_files=False,
    type=appSettings.config_parameters.features.supported_audio_formats,  # Add more file types as needed
)

# ok, check if we got files...
if uploaded_files:
    st.success(f"Upload Successful: {uploaded_files.name}.")

    # input columns
    audio_data, parameters = st.columns([2, 2], vertical_alignment="top")
    audio_data.subheader("File Properties")
    parameters.subheader("Generation Parameters")

    with st.spinner("** Load Samples from File... **"):
        # fetch file info
        decodedAudioFile = AudioDecoder(uploaded_files)
        metadata = decodedAudioFile.metadata
        clipInfoJson = {
            "name": uploaded_files.name,
            "channels": metadata.num_channels,
            "samples": metadata.sample_rate * metadata.duration_seconds_from_header,
            "duration": metadata.duration_seconds_from_header,
            "encoding": {
                "format": metadata.codec,
                "sample_rate": metadata.sample_rate,
                "bitrate": metadata.bit_rate,
            },
        }

    # display info
    audio_data.json(clipInfoJson)

    # add supported language options
    supported_languages = ["English", "German", "French", "Italian", "Spanish"]
    audio_language = parameters.segmented_control(
        "Source Language",
        supported_languages,
        selection_mode="single",
        default="English",
        on_change=reset_agent,
        key="audio_language",
    )
    task = parameters.selectbox("Task", ("Transcribe", "Translate"), index=0)
    if task == "Translate":
        target_language = parameters.segmented_control(
            "Target Language",
            supported_languages,
            selection_mode="single",
            default="English",
            on_change=reset_agent,
            key="target_language",
        )

    # return timestamps if len(audio)>30s
    if clipInfoJson.get("duration") > 30.0:
        return_timestamps = parameters.checkbox("Return Timestamps", value=True)
    else:
        return_timestamps = parameters.checkbox("Return Timestamps", value=False)

    # inference parameters
    generate_kwargs = {
        "language": audio_language,
        "task": task.lower(),
        "return_timestamps": bool(return_timestamps),
    }

    # load samples
    with st.spinner("** Load Samples... **"):
        audio_samples = resample(
            decodedAudioFile,
            target_sample_rate=INFERENCE_SAMPLE_RATE,
            target_num_channels=INFERENCE_CHANNELS,
        ).get_all_samples()

        # display info
        samplesJson = {
            "sample_rate": audio_samples.sample_rate,
            "data": {
                "duration_s": audio_samples.duration_seconds,
                "samples": audio_samples.data.shape[1],
                "channels": audio_samples.data.shape[0],
            },
            "inference": generate_kwargs,
        }

    with st.expander("Clip Information", expanded=False):
        wavepanel, spectrumpanel, infopanel = st.columns([1, 1, 1])
        wavepanel.subheader("Waveform")
        spectrumpanel.subheader("Spectrum")
        with _pyplot_lock:
            plt.subplots(2, 1)
            spec, _ = spectrum(decodedAudioFile)
            wave, _ = waveform(decodedAudioFile)
            wavepanel.pyplot(wave)
            spectrumpanel.pyplot(spec)
        infopanel.subheader("Preview")
        infopanel.audio(audio_samples.data.numpy(), sample_rate=INFERENCE_SAMPLE_RATE)
        infopanel.subheader("Converted For Inference")
        infopanel.json(samplesJson)

    # load model
    whisperPipeline = loadWhisperModel(low_vram=low_vram, device=device)

    # begin conversion
    if samplesJson.get("data").get("channels") > 1:
        st.error("Stereo audio is not supported yet. Please use mono audio.")
    else:
        if parameters.button("Transcribe Audio...", type="primary"):
            # transcribe
            with st.spinner("** TRANSCRIBING AUDIO, PLEASE WAIT ... **"):
                # tensor to numpy..
                samples_array = audio_samples.data.squeeze().numpy()
                prediction = whisperPipeline(
                    samples_array,
                    return_timestamps=return_timestamps,
                    generate_kwargs=generate_kwargs,
                )

            # finished
            st.badge("Success", icon=":material/check:", color="green")

            with st.container(border=True):
                st.subheader("Process Output", divider=True)
                transcribed_text, download_button = st.columns([3, 1])
                transcribed_text.subheader("Transcription")
                transcribed_text.markdown(prediction.get("text"))
                if return_timestamps:
                    with transcribed_text.expander("Timeline"):
                        transcribed_text.json(prediction.get("chunks"))

                # download button
                download_button.download_button(
                    label="Download Transctription",
                    type="primary",
                    data=prediction.get("text"),
                    mime="plain/text",
                    icon=":material/download:",
                )
