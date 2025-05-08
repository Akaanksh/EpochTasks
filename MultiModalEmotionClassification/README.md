# Emotion Recognition Report - Akaanksh
## Overview
This project explores multiple approaches to speech emotion recognition using the RAVDESS dataset, starting with unimodal methods and advancing to multimodal fusion.
The validation accuracies reported are after fine tuning the hyperparameters for a decent amount of time. I realised that most of the models were being overfit due to lesser data. Therefore added L2 and dropout and fine tuned this to a slight extent too.
## 1. CNN on Spectrograms
The initial method used 2D Convolutional Neural Networks (CNNs) on spectrogram images generated from the RAVDESS dataset audio files. Spectrograms visually represent audio frequency content over time, making them a good fit for CNNs.
Architecture: A simple 3-layer CNN with max-pooling and dropout.


Performance:
Validation Accuracy: 75%

Observations: Model showed decent performance due to the structured spatial information in spectrograms.
## 2. ResNet on Spectrograms
Tried a pre-trained ResNet but realised more data is needed. Abandoned temporarily to try out other methods.


Performance:
Accuracy: 70%



## 3. RNN on Transcripts
The next method explored Recurrent Neural Networks (RNNs) on textual transcripts of the audio files.
Architecture: LSTM-based RNN.


Dataset Suitability: This approach was quickly abandoned because the RAVDESS dataset contains only acted phrases with very limited vocabulary and sentence variety. This leads to low variability and poor performance in a text-only model.


Performance:
Validation Accuracy: 32%


Observations: Text data was not sufficient to represent emotional variance effectively.


## 4. CNN on MFCCs
To continue exploring multimodal strategies, a CNN was implemented on Mel-Frequency Cepstral Coefficients (MFCCs), which capture the timbral texture of audio.
Architecture: A 2-layer CNN with dropout.


Performance:
Accuracy: 62%


Observations: Complementary to the spectrogram-based model, though slightly lower standalone accuracy.


## 5. Multimodal NN of Spectrogram CNN and MFCC CNN
To leverage the strengths of both MFCC and spectrogram models. Both are being used in this method. Still fine-tuning the hyperparameters due to which accuracy is not as high as it can be.
Method: Separate branches of both CNNs that then lead into a final dense NN.


Performance:
Accuracy: 68%


## 6. Ensemble of Spectrogram CNN and MFCC CNN
To leverage the strengths of both MFCC and spectrogram models, I am currently working on an ensemble system
Method: Average of logits from the two models before softmax.


Performance:
Accuracy: - 


Observations: Expected to improve robustness and capture more features, but training is ongoing.


## Future Work
Tune the ensemble weights instead of a simple average.
Try attention-based fusion or multimodal transformers.


## File Structure
- MEC-ConvertDataToSpectrTranscripts.ipynb: Converts the audio signals to Spectrograms and Transcripts.
- MEC-ExtractMFCCs.ipynb: Converts the audio to MFCCs.
- MEC-CNNSpectrogramClassification.ipynb: Applies CNNs on the Spectrograms. Also includes the ResNet implementation.
- MEC-RNNTranscriptClassification.ipynb: Tries using an RNN on the transcripts.
- MEC-CNN_MFCC_Classification.ipynb: Applies CNNs on the MFCCs
- MEC-MultiModal.ipynb: Uses both Spectrograms and MFCCs
- MEC-EnsembleMethod.ipynb: Uses both Spectrograms and MFCCs and averages the outputs to make the final decision
- requirements.txt
- README.md




## Notes
All methods use the same 80-20 train-validation split for consistency.



(Haven’t yet added the confusion matrices because ran out of colab free GPU power. Will add immediately once that is regenerated)
