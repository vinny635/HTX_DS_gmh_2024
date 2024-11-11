# Third-party Libraries
from flask import Flask, request, jsonify
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import librosa
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h").to(device) # device_map='cuda' doesn't work, logits end up all nan
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")

# Define the `/asr` route that accepts a file upload and returns the transcription and duration:
app = Flask(__name__)


@app.route('/ping', methods=['GET'])
def ping():
    '''
    Call ping API to return pong
    '''
    return 'pong'


@app.route('/asr', methods=['POST'])
def asr():
    '''
    Call asr API to do speech recognition to retun text and duration
    
    Returns:
        transcription (str): Predicted text
        duration (str): duration of audio in seconds
    '''
    # Get the uploaded audio file
    audio_file = request.files['file']

    # Load the audio file and resample to 16kHz
    audio, sample_rate = librosa.load(audio_file, sr=16000)

    # Preprocess the audio
    input_values = processor(audio, return_tensors="pt", padding="longest").input_values

    # Generate the transcription
    logits = model(input_values.to(device)).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])

    # Calculate the duration
    duration = len(audio) / sample_rate

    # Return the response
    return jsonify({
        "transcription": transcription,
        "duration": f"{duration:.1f}"
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001, debug=True)