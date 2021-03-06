# MandarinTones
An automatic speech recognition system for translating audio speech in Mandarin to its corresponding sequence of tones for each syllable in the utterance. Each syllable in Mandarin Chinese is one of 5 possible tones (4 main tones + 1 light tone). 

Includes implementation of CTC loss, and a custom character-to-tone tagger.

### Example
Here is an example from the test data set. Click [here](/SSB11350366.wav) to play the test audio file.
<p>Sentence read: 目前已经收到不少手机厂商和软件厂商的合作邀请
<br>True tone sequence:  4231144331312343152413
<br>Predicted tone sequence: 4233144331312343152413

<p align="center">
  <img src="tone_probabilities.png?raw=true"/>
</p>

The graph above plots probability of a particular tone (vertical axis) vs time frame in the audio clip, as predicted by our model (horizontal axis). Tone 5 is the 'light tone' or 'no tone' in Chinese, and 0 indicates 'blank' (you can think of it as a pause between two syllables). We see that the model makes one error, the fourth syllable is predicted as a 3rd tone instead of a 1st tone. Indeed, 2nd red peak in in the graph is relatively low indicating that the model isn't very confident about it being a 3rd tone.

There are different ways to get from the above probability plot to a prediction of the tone sequence. For example, for prediction pred2 we predict the highest probability tone for each frame, then collapse consecutive repeated tones not separated by blanks (then finally remove blanks). More accurate results can often be obtained by performing a beam search, this gives prediction pred1.

### Training data
We trained on the [AISHELL-3 dataset](https://www.openslr.org/93/). 
