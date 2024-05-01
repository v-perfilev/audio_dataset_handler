## Audio Dataset Handler

This project contains custom PyTorch datasets and utility scripts designed for audio data management. Developed for use
in my other audio-related projects, it offers convenient tools for dataset handling, preprocessing, and analysis.

### Data Preparation

Before running the project, it's essential to download and organize the necessary datasets.
This project relies on audio data from Mozilla Common Voice and UrbanSound8K.
Please follow the steps below to prepare your data:

#### 1. Mozilla Common Voice

The Mozilla Common Voice dataset is used for multiple languages in this project.
You need to download the datasets for English (en), German (de), and Russian (ru).
After downloading, the data should be organized into specific directories according to the language.

```
../_audio_data/cv-en/    # For English
../_audio_data/cv-de/    # For German
../_audio_data/cv-ru/    # For Russian
```

#### 2. UrbanSound8K

The UrbanSound8K dataset is a compilation of urban sounds from 10 different classes.

```
../_audio_data/UrbanSound8K/
```

### Requirements

**To ensure proper functionality, make sure to have "espeak" installed on your system.**

Python libraries:

- torch
- ffmpeg
- pandas
- phonemizer
- tqdm
