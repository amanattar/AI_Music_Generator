import numpy as np
import pretty_midi
import streamlit as st
from tensorflow.keras.models import load_model

# Load the trained model
@st.cache(allow_output_mutation=True)
def load_trained_model():
    return load_model('rnn_music_best_model.keras')

model = load_trained_model()

# Generate music
def generate_music(model, start_sequence, seq_length=100, output_length=500, temperature=1.0):
    generated = list(start_sequence)
    for _ in range(output_length):
        input_seq = np.expand_dims(generated[-seq_length:], axis=0)
        predictions = model.predict(input_seq, verbose=0)[0]
        predictions = np.log(predictions + 1e-8) / temperature
        predictions = np.exp(predictions) / np.sum(np.exp(predictions))
        next_note = np.random.choice(range(len(predictions)), p=predictions)
        generated.append(next_note)
    return generated

# Convert sequence to MIDI with a selected instrument
def sequence_to_midi(sequence, output_file, instrument_program=0):
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=instrument_program)
    start_time = 0
    duration = 0.5
    for pitch in sequence:
        note = pretty_midi.Note(velocity=100, pitch=int(pitch), start=start_time, end=start_time + duration)
        instrument.notes.append(note)
        start_time += duration
    midi.instruments.append(instrument)
    midi.write(output_file)

# Streamlit App UI
st.title("AI Music Generator 🎵")
st.write("Generate personalized music by selecting parameters below:")

# Input Options
st.subheader("1. Starting Sequence")
sequence_option = st.radio(
    "Choose starting sequence:",
    ("Random Sequence", "Custom Sequence")
)

if sequence_option == "Random Sequence":
    start_sequence = np.random.choice(range(60, 72), size=100).tolist()
    st.write("A new random starting sequence will be used each time you click Generate.")
else:
    custom_sequence = st.text_input(
        "Enter a custom sequence (comma-separated note values, e.g., 60,62,64):",
        "60,62,64,65,67,69,71,72"
    )
    start_sequence = list(map(int, custom_sequence.split(',')))

# Generation Parameters
st.subheader("2. Customize Generation Parameters")
temperature = st.slider("Temperature (randomness):", 0.1, 2.0, 1.0)
output_length = st.slider("Output Length (number of notes):", 100, 1000, 500)

# Instrument Selection
st.subheader("3. Select Instrument")
instrument_dict = {
    "Acoustic Grand Piano": 0,
    "Bright Acoustic Piano": 1,
    "Electric Grand Piano": 2,
    "Honky-Tonk Piano": 3,
    "Electric Guitar (Clean)": 27,
    "Electric Bass (Finger)": 33,
    "Violin": 40,
    "Trumpet": 56,
    "Saxophone": 65,
    "Flute": 74,
    "Steel Drums": 114,
    "Synth Lead": 80,
    "Tuba": 58
}
instrument_name = st.selectbox("Choose an instrument:", list(instrument_dict.keys()))
instrument_program = instrument_dict[instrument_name]

# Generate Button
if st.button("Generate Music"):
    st.write("Generating music... 🎶")
    # Generate a new random starting sequence if Random Sequence is selected
    if sequence_option == "Random Sequence":
        start_sequence = np.random.choice(range(60, 72), size=100).tolist()

    generated_sequence = generate_music(
        model, start_sequence, seq_length=100, output_length=output_length, temperature=temperature
    )
    output_file = "generated_music.mid"
    sequence_to_midi(generated_sequence, output_file, instrument_program=instrument_program)
    st.write(f"Music generated with {instrument_name}.")
    
    # Provide download link and audio playback
    st.audio(output_file, format="audio/midi")
    st.download_button(
        label="Download MIDI File",
        data=open(output_file, "rb"),
        file_name="generated_music.mid",
        mime="audio/midi"
    )
