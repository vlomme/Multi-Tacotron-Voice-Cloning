# Multi-Tacotron Voice Cloning 
This repository is a phonemic multilingual (Russian-English) implementation based on [Real-Time-Voice-Cloning](https://github.com/CorentinJ/Real-Time-Voice-Cloning). it is a four-stage deep learning framework that allows to create a numerical representation of a voice from a few seconds of audio, and to use it to condition a text-to-speech model. If you only need the English version, please use the [original implementation](https://github.com/CorentinJ/Real-Time-Voice-Cloning).

Этот репозиторий является многоязычной(русско-английской) фонемной реализацией, основанной на [Real-Time-Voice-Cloning](https://github.com/CorentinJ/Real-Time-Voice-Cloning). Она состоит из четырёх нейронных сетей, которые позволяют создавать числовое представление голоса из нескольких секунд звука и использовать его для создания модели преобразования текста в речь

### [Example](https://soundcloud.com/fn5va3vghrkh/sets/multi-tacotron)

## Quick start
Use the [colab online demo](https://colab.research.google.com/github/vlomme/Multi-Tacotron-Voice-Cloning/blob/master/Multi_Tacotron_Voice_Cloning.ipynb)

### Requirements
You will need the following whether you plan to use the toolbox only or to retrain the models.

**≥Python 3.6**.

[PyTorch](https://pytorch.org/get-started/locally/) (>=1.0.1).

Run `pip install -r requirements.txt` to install the necessary packages.

A GPU is mandatory, but you don't necessarily need a high tier GPU if you only want to use the toolbox.

### Pretrained models
Download the latest [here](https://drive.google.com/uc?id=1aQBmpflbX_ePUdXTSNE4CfEL9hdG2-O8).

### Datasets
| Name | Language | Link | Comments | My link | Comments |
| --- | -- | ------ | ----- | ----- | ----- |
| Phoneme dictionary | En, Ru | [En](https://github.com/cmusphinx/cmudict),[Ru](https://github.com/nsu-ai/russian_g2p) | Phoneme dictionary | [link](https://drive.google.com/file/d/1tNElQSmpveVx0qyqKr1j5slfyM1HVifj/view?usp=sharing) | Совместил русский и английский фонемный словарь |
| LibriSpeech | En | [link](http://www.openslr.org/12/) | 300 speakers, 360h clean speech |  |  |
| VoxCeleb | En  | [link](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/index.html#portfolio) | 7000 speakers, many hours bad speech |  |  |
| M-AILABS | Ru | [link](https://www.caito.de/2019/01/the-m-ailabs-speech-dataset/) | 3 speakers, 46h clean speech|  |  |
| open_tts, open_stt | Ru | [open_tts](https://github.com/snakers4/open_tts), [open_stt](https://github.com/snakers4/open_stt/) | many speakers, many hours bad speech | [link](https://drive.google.com/open?id=1H5vJMuCtBP6RhQ7noQwNji_yY9dCt4nU) | Почистил 4 часа речи одного спикера. Поправил анотацию, разбил на отрезки до 7 секунд |
| Voxforge+audiobook | Ru | [link](http://voxforge.org/downloads/) | Many speaker, 25h various quality | [link](https://drive.google.com/open?id=1OjbQMW3wLgVUNUTZ2diJXubkqdlzgNNn) | Выбрал хорошие файлы. Разбил на отрезки. Добавил аудиокниг из интернета. Получилось 200 спикеров по паре минут на каждого |
| RUSLAN | Ru | [link](https://github.com/ruslan-corpus/ruslan-corpus.github.io) | One speaker, 40h good speech | [link](https://drive.google.com/open?id=1ghrL6al-FDbNkoZ_rLVzLjnGmBNWLbpG) | Перекодировал в 16кГц |
| Mozilla | Ru | [link](https://voice.mozilla.org/ru/datasets) | 50 speaker, 30h good speech | [link](https://drive.google.com/open?id=1Mf4EGQUhlz6nmeky8qQTYFedmW0DNzjV) | Перекодировал в 16кГц, Раскидал разных пользователей по папкам |
| Russian Single | Ru | [link](https://www.kaggle.com/bryanpark/russian-single-speaker-speech-dataset) | One speaker, 9h good speech | [link](https://drive.google.com/open?id=1ycdLrB_k2GRVePGiuNSbq30TB1_JlpNg) | Перекодировал в 16кГц |


### Toolbox
You can then try the toolbox:

`python demo_toolbox.py -d <datasets_root>`  
or  
`python demo_toolbox.py`  


## Wiki
[Pretrained models](https://github.com/vlomme/Multi-Tacotron-Voice-Cloning/wiki/Pretrained-models)

[Тренировка (и для других языков)](https://github.com/vlomme/Multi-Tacotron-Voice-Cloning/wiki/%D0%A2%D1%80%D0%B5%D0%BD%D0%B8%D1%80%D0%BE%D0%B2%D0%BA%D0%B0-(%D0%B8-%D0%B4%D0%BB%D1%8F-%D0%B4%D1%80%D1%83%D0%B3%D0%B8%D1%85-%D1%8F%D0%B7%D1%8B%D0%BA%D0%BE%D0%B2))

[Training (and for other languages)](https://github.com/vlomme/Multi-Tacotron-Voice-Cloning/wiki/Training-(and-for-other-languages))

## Contribution
for any questions, please [email me](niw9102@gmail.com)

### Papers implemented  
| URL | Designation | Title | Implementation source |
| --- | ----------- | ----- | --------------------- |
|[**1806.04558**](https://arxiv.org/pdf/1806.04558.pdf) | **SV2TTS** | **Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech Synthesis** | [CorentinJ](https://github.com/CorentinJ/Real-Time-Voice-Cloning) |
|[1802.08435](https://arxiv.org/pdf/1802.08435.pdf) | WaveRNN (vocoder) | Efficient Neural Audio Synthesis | [fatchord/WaveRNN](https://github.com/fatchord/WaveRNN) |
|[1712.05884](https://arxiv.org/pdf/1712.05884.pdf) | Tacotron 2 (synthesizer) | Natural TTS Synthesis by Conditioning Wavenet on Mel Spectrogram Predictions | [Rayhane-mamah/Tacotron-2](https://github.com/Rayhane-mamah/Tacotron-2)
|[1710.10467](https://arxiv.org/pdf/1710.10467.pdf) | GE2E (encoder)| Generalized End-To-End Loss for Speaker Verification |  [CorentinJ](https://github.com/CorentinJ/Real-Time-Voice-Cloning) |
