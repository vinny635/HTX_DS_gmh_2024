# Native Libraries
import requests
import glob
from tqdm import tqdm

# Third-party Libraries
import pandas as pd

# # Local files
# from asr_api import asr

def audio_decode(audio_dir: str):
    '''
    Call asr_api API to convert mp3 files from speech to text iteratively in a folder

    Parameters:
        audio_dir (str): folder name containing audio files

    Returns:
        csv file: updated csv file with predicted text (same name as audio_dir) 
    '''
    asr_url = 'http://localhost:8001/asr'

    generated_text_arr = []
    audio_file_arr = sorted(glob.glob(f'{audio_dir}/*.mp3'))

    for audio_file in tqdm(audio_file_arr):

        files = {'file': open(audio_file, 'rb')}
        response = requests.post(asr_url, files=files)

        generated_text_arr.append(response.json()['transcription'])


    valid_dev_df = pd.read_csv(f"{audio_dir}.csv")

    valid_dev_df = pd.concat([valid_dev_df, pd.DataFrame(data={'generated_text':generated_text_arr})],axis=1)
    valid_dev_df.to_csv(f"{audio_dir}.csv", index=False)
    

if __name__ == '__main__':
    audio_dir = "common_voice/cv-valid-dev"

    audio_decode(audio_dir)