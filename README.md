# MandarinTones
An automatic speech recognition system for converting audio speech in Mandarin to the corresponding syllabus tone sequence. Each syllable in Mandarin Chinese is one of 5 possible tones (4 main tones + 1 light tone). 

### Example
Here is an example from the test data set. Click [here](/SSB11350366.wav) to play the test audio file.
<p>Sentence read: 目前已经收到不少手机厂商和软件厂商的合作邀请
<br>True tone sequence:  4231144331312343152413
<br>Predicted tone sequence: 4233144331312343152413

<p align="center">
  <img src="tone_probabilities.png?raw=true"/>
</p>

The graph above plots probability of a particular tone vs time frame in audio clip, as predicted by our model. Tone 5 is the 'light tone' or 'no tone' in Chinese, and 0 indicates 'blank' (you can think of it as a pause between two syllables).

There are different ways to get from the above probability plot to a prediction of the tone sequence. For example, for pred2 we predict the highest probability tone for each frame, then collapse repeated tones not separated by blanks (then finally remove blanks). More accurate results can often be obtained by performing a beam search (pred1).
