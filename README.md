# Multi-Tacotron Voice Cloning 
This repository is a phonemic multilingual (Russian-English) implementation based on [CorentinJ](https://github.com/CorentinJ/Real-Time-Voice-Cloning). it is a four-stage deep learning framework that allows to create a numerical representation of a voice from a few seconds of audio, and to use it to condition a text-to-speech model.

Этот репозиторий является многоязычной(русско-английской) фонемной реализацией, основанной на [Real-Time-Voice-Cloning](https://github.com/CorentinJ/Real-Time-Voice-Cloning). Она состоит из четырёх нейронных сетей, которые позволяют создавать числовое представление голоса из нескольких секунд звука и использовать его для создания модели преобразования текста в речь


## Quick start
Use the [colab online demo](https://colab.research.google.com/github/vlomme/Multi-Tacotron-Voice-Cloning/blob/master/Multi-Tacotron-Voice-Cloning.ipynb)

### Requirements
You will need the following whether you plan to use the toolbox only or to retrain the models.

**≥Python 3.6**.
[PyTorch](https://pytorch.org/get-started/locally/) (>=1.0.1).
Run `pip install -r requirements.txt` to install the necessary packages.

A GPU is mandatory, but you don't necessarily need a high tier GPU if you only want to use the toolbox.

### Pretrained models
Download the latest [here](https://drive.google.com/uc?id=1aQBmpflbX_ePUdXTSNE4CfEL9hdG2-O8).

### Datasets
| URL | Designation | Title | Implementation source |
| --- | ----------- | ----- | --------------------- |
| |  | |  |
| |  | |  |
| |  | |  |
| |  | |  |

### Toolbox
You can then try the toolbox:

`python demo_toolbox.py -d <datasets_root>`  
or  
`python demo_toolbox.py`  


## Wiki
(coming soon!)


## Contribution
for any questions, please [email me](niw9102@gmail.com)

### Papers implemented  
| URL | Designation | Title | Implementation source |
| --- | ----------- | ----- | --------------------- |
|[**1806.04558**](https://arxiv.org/pdf/1806.04558.pdf) | **SV2TTS** | **Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech Synthesis** | [CorentinJ](https://github.com/CorentinJ/Real-Time-Voice-Cloning) |
|[1802.08435](https://arxiv.org/pdf/1802.08435.pdf) | WaveRNN (vocoder) | Efficient Neural Audio Synthesis | [fatchord/WaveRNN](https://github.com/fatchord/WaveRNN) |
|[1712.05884](https://arxiv.org/pdf/1712.05884.pdf) | Tacotron 2 (synthesizer) | Natural TTS Synthesis by Conditioning Wavenet on Mel Spectrogram Predictions | [Rayhane-mamah/Tacotron-2](https://github.com/Rayhane-mamah/Tacotron-2)
|[1710.10467](https://arxiv.org/pdf/1710.10467.pdf) | GE2E (encoder)| Generalized End-To-End Loss for Speaker Verification |  [CorentinJ](https://github.com/CorentinJ/Real-Time-Voice-Cloning) |
