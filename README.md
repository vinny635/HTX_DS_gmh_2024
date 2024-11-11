
# SystemCode Structure

## Directory Layout
```
HTX_DS_gmh_2024/
│
├── README.md
├── requirements.txt
├── HTX xData Technical Test Questions (Data Scientist).pdf
├── training-report.pdf
├── essay-ssl.pdf
│
├── common_voice/
│   ├── cv-valid-dev.csv
│   ├── cv-valid-train.csv
│   ├── cv-valid-dev-q4.csv
│   ├── cv-valid-dev-q5.csv
│   └── cv-valid-train-q2.csv
│
├── asr/
│   ├── requirements_asr.txt
│   ├── Dockerfile
│   ├── .dockerignore
│   ├── asr_api.py
│   └── cv-decoder.py
│
├── asr-train/
│   ├── logs/
│   ├── wav2vec2-finetuned/
│   └── cv-train-2a.ipynb
│   
│
└── hotword-detection/
    ├── cv-hotword-5a.ipynb
    ├── cv-similarity-5b.ipynb
    └── detected.txt
    


```

## Brief Description
`HTX_DS_gmh_2024` repository answers the `HTX xData Technical Test Questions (Data Scientist).pdf`. Repository includes containerising an ASR model using Docker, finetuning the ASR model called Wav2Vec2 model (facebook/wav2vec2-large-960h), similar words search as well as literature review about self-supervised learning for ASR model. 

## Instructions on How to Run the Code

1. Navigate to the HTX_DS_gmh_2024 directory:
   ```bash
   cd HTX_DS_gmh_2024/
   ```
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Login to HuggingFace by typing this into the command terminal while having the virtual environment activated:
   ```bash
   huggingface-cli login
   ```
   Subsequently, enter your HuggingFace token when being asked for it.
4. From [Kaggle](https://www.kaggle.com/datasets/mozillaorg/common-voice), download cv-valid-train.csv  and cv-valid-dev.csv and put in common_voice folder.
5. (Optional) if you want to preprocess the raw dataset manually, download cv-valid-train folder and cv-valid-dev folder from the same link.
6. Everytime you want to run the code, ensure to go into the folder e.g. `cd {folder-name}` before running.


## Additional Resources

- **Dataset**: From [Kaggle](https://www.kaggle.com/datasets/mozillaorg/common-voice), download cv-valid-train folder, cv-valid-dev folder and their respective csv files and put in common_voice folder.
- **Training Dataset**: The preprocessed training dataset used to fine-tune the Wav2Vec2 model is available [here](https://huggingface.co/datasets/gmh98/common_voice_train_ds_v1).
- **Validation Dataset**: The preprocessed validation dataset used to fine-tune the Wav2Vec2 model is available [here](https://huggingface.co/datasets/gmh98/common_voice_valid_ds_v1).


## Note
- Due to huggingface limited to only 10000 files in a dataset, only a subset of training and validation dataset are used for finetuning