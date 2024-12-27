AI Music Generator 🎵
=====================

This project is an AI-powered music generator that uses a Recurrent Neural Network (RNN) trained on the **Maestro Dataset** to create music tracks. Users can interact with the generator via a web interface to customize music generation, select instruments, and download MIDI files.

Features
--------

*   **Interactive Web Interface**: Built with Streamlit, allowing users to:
    *   Select a starting sequence (random or custom).
    *   Customize generation parameters (e.g., temperature, output length).
    *   Choose an instrument for the generated music.
    *   Download or play the generated music.
*   **RNN-based Music Generation**:
    *   Utilizes LSTM layers to learn and generate note sequences.
    *   Trained on the Maestro v3.0.0 MIDI dataset.

* * *

File Structure
--------------

```graphql
AI_Music_Generator
├── generate_music.ipynb         # Jupyter notebook for music generation
├── LICENSE                      # Project license
├── maestro_sequences.pkl        # Preprocessed Maestro dataset
├── model_architecture.json      # JSON representation of the RNN model
├── Model_training.ipynb         # Notebook for training the RNN model
├── music_generator_app.py       # Streamlit app for user interaction
├── preprocessing.ipynb          # Notebook for preprocessing the Maestro dataset
├── README.md                    # Project README file
├── requirements.txt             # Required Python packages
├── rnn_model_architecture.json  # JSON representation of RNN architecture
├── rnn_model_architecture.png   # Visual diagram of the RNN architecture
├── rnn_music_best_model.keras   # Best-trained model saved during training
├── rnn_music_final_model.h5     # Final trained RNN model
├── training.ipynb               # Notebook for model training and evaluation
```

* * *

Installation and Setup
----------------------

Follow these steps to set up and run the project:

### 1\. Clone the Repository

```bash
git clone https://github.com/your-repo/AI_Music_Generator.git
cd AI_Music_Generator
```

### 2\. Set Up the Environment

Use the provided `requirements.txt` file to set up the Conda environment:

```bash
conda create -n ai_music_generator python=3.9
conda activate ai_music_generator
pip install -r requirements.txt
```

### 3\. Download the Dataset

Download the **Maestro Dataset** using `wget`:

```bash
wget https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip
```

Unzip the dataset into the `maestro_dataset` folder:

```bash
unzip maestro-v3.0.0-midi.zip -d maestro_dataset
```

* * *

Usage
-----

### 1\. Preprocess the Dataset

Run the `preprocessing.ipynb` notebook to preprocess the Maestro dataset and generate sequences:


### 2\. Train the Model

Train the RNN model by running the `Model_training.ipynb` notebook:


The trained model will be saved as `rnn_music_best_model.keras`.

### 3\. Generate Music

Use the `generate_music.ipynb` notebook to generate music using the trained model:



### 4\. Interactive Web App

Run the Streamlit app to interact with the music generator:

```bash
streamlit run music_generator_app.py
```

*   Customize generation parameters (e.g., starting sequence, temperature, instrument).
*   Play or download the generated MIDI file.

* * *

Examples
--------

### Random Starting Sequence

*   **Starting Sequence**: Random
*   **Temperature**: 1.2
*   **Instrument**: Synth Lead
*   **Output Length**: 500 notes

Generated music can be played or downloaded as a MIDI file.

### Custom Starting Sequence

*   **Starting Sequence**: `60,62,64,65,67`
*   **Temperature**: 1.0
*   **Instrument**: Violin
*   **Output Length**: 300 notes

* * *

Instruments
-----------

Here are some available instruments:

*   Acoustic Grand Piano (0)
*   Electric Guitar (Clean) (27)
*   Violin (40)
*   Trumpet (56)
*   Flute (74)
*   Steel Drums (114)
*   Synth Lead (80)

Refer to the full [General MIDI Instrument List](https://en.wikipedia.org/wiki/General_MIDI#Program_change_events) for additional options.

* * *

Credits
-------

*   **Maestro Dataset**: [Magenta Project](https://magenta.tensorflow.org/datasets/maestro)
*   **Streamlit**: Interactive web app framework
*   **TensorFlow**: Deep learning library

* * *

License
-------

This project is licensed under the [MIT License](LICENSE).

* * *

