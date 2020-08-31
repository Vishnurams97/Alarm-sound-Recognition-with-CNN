# Alarm-sound-Recognition-with-Convolutional Neural Network
### The project is the prototype of the proposal in this link - https://link.springer.com/chapter/10.1007/978-981-15-0199-9_31
* The project's target is to recognize the ICU beep sounds and classify them as Normal and Alarm sounds in real time.
This project uses two strategies namely Supervised Learning and Peak detection.


1)The <code>Datasets</code> folder contains,

	Notebook file
	* defining Model structure, Training and Testing the model.
	* Testing the model with realtime audio input.

	All audio samples in wav format.
   	containing,
	* Beep and Alarm audio samples denoted by 'd' as sample name.
	* Normal audio samples denoted by 'n' as sample name.

	Keras models saved after training as .h5 files.

2)The Raspberry Pi folder contains the python script that uses the pretrained cnn_model(cnn_model.h5) which is run by raspberry pi 3B+ model for predicting the Alarm sounds in real time noisy environments. The predictions can be influenced by the Audio input Quality. I used a USB Microphone with Raspberry Pi model 3B+.

The Project depends on Python libraries namely <code>Keras(with tensorflow backend), Librosa, PyAudio, PyAudio, Sklearn, Numpy ,etc.</code>

For testing - https://www.youtube.com/watch?v=Gd3nCvwoPQ4
