## Multimodal Sentiment Analysis/Emotion Detection  
  
_Novel Approach with few implementations_  
  
- Different Ways to Extract Audio Features  
- Multimodal fusion (Text + Audio)  
- Use Mel frequency cepstral coefficients (MFCC) to train a recurrent neural network (LSTM)/Decision Tree and classify emotions/check accuracy  

#### Dataset:  
- Interactive Emotional Dyadic Motion Capture (IEMOCAP) -- [Link](http://sail.usc.edu/iemocap/)  
- Only audio and text data into consideration  
  
#### Model  
- 2 layers of LSTM. The batch size and epoch were varied/reduced as per requirement.  
- Decision Tree CLassifier  

### Requirements  
- numpy  
- sklearn  
- librosa (for pitch estimation, optional)  
- keras  
- tensorflow 1.3.0  
- python_speech_features  
- wave  
- cPickle  
- os  
- (more in code)  
  
### Genereic implementations
- models.py - keras implementation of LSTM,sklearn models
- calculate_features.py - script for features estimation
- extract_features.py, cf.py, to_speech.py, pitch.py

 Process:
  - Data reading 
  - Data preprocessing: normalization, padding sequences, transfromation labels 
  - The main part includes model building and 5-fold cross-validation

Librosa/python speech used to extract the feature out of a given voice.

**Files to look for:**
- [audio1](https://github.com/ManojVaidya/Multimodal-Sentiment-Analysis/tree/master/src/audio1.ipynb) : MFCC extraction using python speech_features
- [audio2](https://github.com/ManojVaidya/Multimodal-Sentiment-Analysis/tree/master/src/audio2.ipynb) : Audio visualization. (Performed for single file)
			[Reference](https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html)
- [multimodal1](https://github.com/ManojVaidya/Multimodal-Sentiment-Analysis/tree/master/src/multimodal1.ipynb): Fusion technique and decision tree
- [multimodal2](https://github.com/ManojVaidya/Multimodal-Sentiment-Analysis/tree/master/src/multimodal2.ipynb): MFCC+LSTM and scores

_Note:Most of it is an novel approach.Necessary stpes like changing input location(after download), parameter tuning, error handling(if any) etc., may be required._
