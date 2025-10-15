#!/usr/bin/env python
#
# Kokoro TTS Integration: Text To Audio support
# *EXPERIMENTAL*
#

try:
    import streamlit as st
    from dotenv import dotenv_values
    from threading import RLock

    with st.spinner("** LOADING TTS PIPELINE COMPONENTS... **"):
        from torchcodec.decoders import AudioDecoder
        from torchcodec.encoders import AudioEncoder
        from kokoro import KPipeline
        import matplotlib.pyplot as plt

    with st.spinner("** LOADING INTERFACE... **"):
        # local imports
        from uuid import uuid4
        import libs.shared.huggingface as hf
        from libs.shared.settings import Properties
        from libs.shared.session import Session
        from libs.shared.utils import detect_accelerator
        from libs.utils.audio_pipelines import resample, waveform, spectrum, tensorToBytes
except Exception as e:
    print(f"Caught fatal exception: {e}")

# whisper inference settings
INFERENCE_SAMPLE_RATE = 22050
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
st.html("assets/tts.html")

# get models
supported_models = appSettings.config_parameters.data.get("kokoro").get(
    "supported_models"
)
model = appSettings.config_parameters.kokoro.model

# pyplot lock
_pyplot_lock = RLock()

# detect acceleration device
device, dtype = detect_accelerator()

@st.cache_resource
def loadKokoroPipeline(repo_id: str, lang_code: str) -> KPipeline:
    with st.spinner("**Loading model into memory...**"):
        pipeline = KPipeline(lang_code=lang_code, repo_id=repo_id)
    return pipeline

# populate sidebar
with st.sidebar:
    # reset function
    def reset_agent():
        st.cache_resource.clear()

    st.header("🛠 TTS Control Panel")

    st.markdown("**Text-To-Speech Setting Selection**")
    selected_model = st.selectbox(
        label="Select TTS Model",
        options=supported_models.keys(),
        index=[i for i, x in enumerate(supported_models.keys()) if x == model].pop(),
        on_change=reset_agent,
    )

    st.markdown("**Audio Settings**")
    selected_language = st.selectbox(
        label="Supported Languages",
        options=[x.get("type") for x in appSettings.config_parameters.kokoro.supported_languages],
        index=0,
        on_change=reset_agent,
    )
    for i, x in enumerate(appSettings.config_parameters.kokoro.supported_languages):
        if x.get("type") == selected_language:
            lang_params = x

    st.markdown(f"**Model: `{selected_model}`**")
    st.markdown(f"**Device: `{device}/{dtype}`**")
    st.markdown(f"**Language: `{lang_params.get("lang_id")}/{lang_params.get('voice_id')}`**")

# file uploader
with st.spinner("** GETTING TTS MODEL FROM HUGGINGFACE... **"):
    try:
        # download model from huggingface
        mp = hf.downloadFromHuggingFace(
            repo_id=supported_models.get(selected_model),
            local_dir=appSettings.config_parameters.huggingface.local_dir,
            cache_dir=appSettings.config_parameters.huggingface.cache_dir,
            apitoken=appSettings.config_parameters.huggingface.apitoken,
            revision=appSettings.config_parameters.kokoro.revision,
        )
    except Exception as e:
        st.markdown(f"🚨 Error downloading model from HuggingFace: {e}")


# input area
text_area, upload_area = st.tabs(["Text Input", "File Input"])

# file input section
with upload_area:
    st.subheader("Upload an Audio File", divider=True)
    uploaded_files = st.file_uploader(
        "Upload Text Files",
        accept_multiple_files=False,
        type=["txt", "md"] # Add more file types as needed
    )

    # ok, check if we got files...
    if uploaded_files:
        # read file...
        text_content = uploaded_files.getvalue()
        st.success(f"Upload Successful: {uploaded_files.name} contains {len(text_content.decode('utf-8').split(' '))} words.")

        # begin inference
        OUTPUT_SAMPLES: list = []
        # create audio pipeline
        try:
            k_pipe = loadKokoroPipeline(repo_id=supported_models.get(selected_model), lang_code=lang_params.get("lang_id"))

            # generate audio
            OUTPUT_SAMPLES.append(k_pipe(text=text_content.decode('utf-8'), voice=lang_params.get("voice_id")))
        except Exception as e:
            st.error(f"Error during inference: {e}")

        # output generated audio
        with st.spinner(text="Rendering.....", show_time=True):
            ENCODED_AUDIO: list = []
            for i, output in enumerate(OUTPUT_SAMPLES):
                for index, (gs, ps, audio) in enumerate(output):
                    # encode audio
                    filename: str = f"/tmp/output_{i}.mp3"
                    e: AudioEncoder = AudioEncoder(samples=audio, sample_rate=INFERENCE_SAMPLE_RATE)
                    e.to_file(filename)
                    ENCODED_AUDIO.append({
                        "audioencoder": e,
                        "graphemes": gs,
                        "phonemes": ps, 
                        "filename": filename
                    })

        # display generated waveforms
        for item in ENCODED_AUDIO:
            # get audio decoder
            d: AudioDecoder = AudioDecoder(item.get("audioencoder").to_tensor(format="mp3"))

            # hear it!
            audio_samples = resample(d, target_sample_rate=INFERENCE_SAMPLE_RATE, target_num_channels=INFERENCE_CHANNELS, target_format="mp3").get_all_samples()
            
            # display info
            samplesJson = {
                "sample_rate": audio_samples.sample_rate,
                "data": {
                    "duration_s": audio_samples.duration_seconds,
                    "samples": audio_samples.data.shape[1],
                    "channels": audio_samples.data.shape[0],
                },
                "inference": {
                    "gs": item.get("graphemes"),
                    "ps": item.get("phonemes"),
                    "output_file": item.get("filename")
                },
            }

            clip_info, download_area = st.columns([3,1])

            with clip_info:
                # display information
                with st.expander("Clip Information", expanded=False):
                    wavepanel, spectrumpanel, infopanel = st.columns([1, 1, 1])
                    wavepanel.subheader("Waveform")
                    spectrumpanel.subheader("Spectrum")
                    with _pyplot_lock:
                        plt.subplots(2, 1)
                        spec, _ = spectrum(d)
                        wave, _ = waveform(d)
                        wavepanel.pyplot(wave)
                        spectrumpanel.pyplot(spec)
                    infopanel.subheader("Preview")
                    infopanel.audio(audio_samples.data.numpy(), sample_rate=INFERENCE_SAMPLE_RATE)
                    infopanel.subheader("Inference Output")
                    infopanel.json(samplesJson)

# text input section
# TODO: refactor....
with text_area:
    st.subheader("Write text that you want to convert to audio", divider=True)
    text_in, button_in = st.columns([3,1], vertical_alignment="center")
    with text_in:
        input_text_to_convert = st.text_input(label="Input Text")

    with button_in:
        # add conversion button
        submit = st.button(label="Text-To-Speech", type="primary", disabled=not input_text_to_convert)

    # convert text to audio
    if submit:
        OUTPUT_SAMPLES: list = []
        # create audio pipeline
        try:
            k_pipe = loadKokoroPipeline(repo_id=supported_models.get(selected_model), lang_code=lang_params.get("lang_id"))

            # generate audio
            OUTPUT_SAMPLES.append(k_pipe(text=input_text_to_convert, voice=lang_params.get("voice_id")))
        except Exception as e:
            st.error(f"Error during inference: {e}")

        # output generated audio
        with st.spinner(text="Rendering.....", show_time=True):
            ENCODED_AUDIO: list = []
            for i, output in enumerate(OUTPUT_SAMPLES):
                for index, (gs, ps, audio) in enumerate(output):
                    # encode audio
                    filename: str = f"/tmp/output_{i}.mp3"
                    e: AudioEncoder = AudioEncoder(samples=audio, sample_rate=INFERENCE_SAMPLE_RATE)
                    e.to_file(filename)
                    ENCODED_AUDIO.append({
                        "audioencoder": e,
                        "graphemes": gs,
                        "phonemes": ps, 
                        "filename": filename
                    })

        # display generated waveforms
        for item in ENCODED_AUDIO:
            # get audio decoder
            d: AudioDecoder = AudioDecoder(item.get("audioencoder").to_tensor(format="mp3"))

            # hear it!
            audio_samples = resample(d, target_sample_rate=INFERENCE_SAMPLE_RATE, target_num_channels=INFERENCE_CHANNELS, target_format="mp3").get_all_samples()
            
            # display info
            samplesJson = {
                "sample_rate": audio_samples.sample_rate,
                "data": {
                    "duration_s": audio_samples.duration_seconds,
                    "samples": audio_samples.data.shape[1],
                    "channels": audio_samples.data.shape[0],
                },
                "inference": {
                    "gs": item.get("graphemes"),
                    "ps": item.get("phonemes"),
                    "output_file": item.get("filename")
                },
            }

            clip_info, download_area = st.columns([3,1])

            with clip_info:
                # display information
                with st.expander("Clip Information", expanded=False):
                    wavepanel, spectrumpanel, infopanel = st.columns([1, 1, 1])
                    wavepanel.subheader("Waveform")
                    spectrumpanel.subheader("Spectrum")
                    with _pyplot_lock:
                        plt.subplots(2, 1)
                        spec, _ = spectrum(d)
                        wave, _ = waveform(d)
                        wavepanel.pyplot(wave)
                        spectrumpanel.pyplot(spec)
                    infopanel.subheader("Preview")
                    infopanel.audio(audio_samples.data.numpy(), sample_rate=INFERENCE_SAMPLE_RATE)
                    infopanel.subheader("Inference Output")
                    infopanel.json(samplesJson)

            with download_area:
                @st.fragment
                def downloadButton(data, filename: str):
                    st.download_button(label="Download Audio File", type="primary",
                                    data=data,
                                    file_name=filename,
                                    icon=":material/download:")

                try:
                    with open(f"{samplesJson['inference'].get('output_file')}", "rb") as f:
                        audio_bytes = f.read()

                    downloadButton(data=audio_bytes, filename=f"{str(uuid4())}.mp3")
                except AssertionError as e:
                    st.error(f"Type Assertion Error {e}")