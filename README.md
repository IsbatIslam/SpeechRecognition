# Speech Recognition Using Mel-Frequency Cepstrum Coefficients and a Convolutional Neural Network

## Introduction
In my previous project, I delved into the complexities of sound recognition from musical instruments. I used techniques like Fast Fourier Transforms (FFTs) to dissect the sound into its fundamental frequency and overtones. Despite this, I realized that recognizing a sound purely by its frequency isn't enough; factors like timbre and pitch play a crucial role in making each sound unique.

This realization inspired my next personal project – to explore the intricacies of human voice. I wanted to see if I could develop a program capable of identifying the specific digit I'm saying, whether it's 'one' or 'five'. To tackle this challenge, I went beyond FFTs, diving into spectrograms, Mel-spectrograms, and Mel-frequency cepstrum coefficients (MFCC). These techniques allowed me to analyze speech patterns in depth.

I also ventured into the realm of machine learning, employing a Convolutional Neural Network (CNN) to recognize subtle differences in speech, such as distinguishing between 'two' and 'four'. This personal project was a journey into the heart of audio processing, pushing my skills to new heights and opening up exciting possibilities in the world of voice recognition


## Procedure
The objective of my project was to develop a program capable of predicting spoken digits between 1-5. I conducted a detailed analysis of Mel-frequency cepstral coefficients (MFCCs) and spectrograms corresponding to these digits to gain insights into their unique audio signals. Subsequently, I performed an experiment to assess the accuracy of my prediction model.

**Part 1 -- Visualizing Noise**

In the initial phase, I explained the workings of Fourier Transforms and Mel-spectrograms, focusing on how these techniques are utilized to extract MFCCs. This segment primarily involved theoretical explanations and rigorous code validation, which was pivotal for the subsequent stages.

**Part 2 -- Creating Dataset**

The second phase involved generating a dataset by extracting audio features exclusively from my voice recordings. Despite the time constraints, I opted to solely use my voice as the data source, acknowledging the inherent bias. I processed each sample to extract MFCCs, storing the results in a structured JSON file. Additionally, I presented MFCCs and Mel-spectrograms of select digits, showcasing what the computer should recognize during model training.

**Part 3 -- Predicting Digits**

In the final phase, I employed a Convolutional Neural Network (CNN) and the prepared JSON file to construct a deep learning model. This model was then utilized to predict audio samples. I meticulously evaluated the model's accuracy and conducted a comprehensive analysis of the results, aiming to provide detailed explanations and discussions surrounding the obtained outcomes.

## Part 1: Visualizing Noise and Theory
The feature we want to extract is Mel-frequency coefficients (MFCC). But to understand what MFCCs represent, we need to understand the underlying concepts:


### Fourier Transforms:
<img src="https://github.com/IsbatIslam/SpeechRecognition/assets/106777293/f58b11a6-8fd1-45df-baff-ba296d7563f8">

The formula above is the Complex Fourier Transform equation where:

ĝ(f) = the complex Fourier point,

g(t) = the waveform (sum of multiple pure tone signals)

e^(-i2πft) = Euler's formula of a unit circle around a complex plane, composed of the imaginary part and the real part

When a computer applies a Fast Fourier Transform on a sample, it ends up with a pair of parameters, the magnitude and the phase that a complex number conveniently represents. Thus, what happens in a complex Fourier Transform is that real numbers, from the waveform signal, are converted into complex numbers that can represent the signal's position (magnitude and phase) in the complex plane.



I will try to explain the equation piece by piece:

**e^(-i2πft)** is the part of the formula which generates a unit circle around the complex plane. As seen in NYC Physics, unit circles are directly linked with sin/cos waves, and it's no different here. e^(-i2πft) generates a ***pure tone***, and the " f, "which is frequency, determines at which speed we can complete one cycle (a period) around the unit circle.

As seen in the equation, we then multiply the unit circle with the signal **g(t)**, which wraps around the waveform around the unit circle, which looks something like this:


<img src="https://github.com/IsbatIslam/SpeechRecognition/assets/106777293/780a6722-517e-48b9-98d9-30ce01adb31b">

As we see in the above figure, the signal above is multiplied by the unit circle giving us a flower-like figure in the imaginary plane. The computer generates such unit circles at many frequencies producing many pure tones that wrap around the signal in the same manner. When a pure tone is present in the waveform, the unit circle x the waveform =
<img src="https://github.com/IsbatIslam/SpeechRecognition/assets/106777293/bcd5a3e2-a5c1-41f8-b87b-dcf00734394e">


In the above figure, the e^(-i2πft) and the g(t) wave signal match, giving us a "simple" image signal. Whenever the image is "stable" like the image here, it means that the complex number found, ĝ(f) representing a specific phase and magnitude (the red dot), is present in the signal. The green spot is simply the red spot after applying the integral. The integral sums up all of the continuous points of the petal-like shape and averages them, giving the red spot, which is like the "center of mass." The green spot is the red spot times the number of times the number of time the frequency appears throughout the time domain. The more distance between the green dot and the origin, the more of that specific frequency is in the waveform.


<img src="https://github.com/IsbatIslam/SpeechRecognition/assets/106777293/92f75aa2-3871-4d50-8c31-a044e2b73364">

Graphics from Elan Ness-Cohn, site: https://sites.northwestern.edu/elannesscohn/2019/07/30/developing-an-intuition-for-fourier-transforms/


###Mel-Spectrogram

For my project, I had to push the Frequency Domain further and convert the waveform signal into a Time-Frequency domain, a spectrogram. A spectrogram is simply a visual representation of a signal that had undergone a FFT, but instead of all being put into one magnitude of a Frequency Domain, it is spread out throughout its time domain. Thus, if a frequency were present at the beginning but not at the end, we would see it and its intensity throughout time without it being bunched up like a Frequency Domain.


A Mel-Spectrogram is a perceptually accurate (to a human) version of a spectrogram. Indeed it shows a perceptually-relevant amplitude, pitch and time-frequency representation of what a human ear would perceive.


A keyword here is logarithm. We convert our signal into a log-spectrogram because the human ear perceives sound logarithmically, which is via decibels (dB).

Humans also do not perceive frequency linearly, but logarithmically. Indeed, the difference between 200Hz - 300Hz seems less audibly noticeable than a difference between 1500Hz - 1700Hz, despite being the same difference. Thus, to get a perceptually accurate pitch (frequency) representation, we apply a Mel-Filter bank on our log-amplitude spectrogram to compensate for that.

<img src="https://github.com/IsbatIslam/SpeechRecognition/assets/106777293/eccc91bb-0b88-462e-82a4-b37635288425">

###Mel-Frequency Cepstrum Coefficient

**Mel-frequency Cepstrum Coefficient**, the MFCC, is the feature that I will extract from all of my audio samples.

I presented the Fourier Transforms and the Mel-spectrogram first because, to obtain the MFCC, we must apply those two first to a waveform signal. But, what is an MFCC?

Mel-frequency refers to the Mel-spectrogram, a perceptually relevant way of representing sound, and a coefficient is simply a number. But what is cepstrum?

First, let's see how speech works.
Speech first comes through your vocals folds and appears as glottal pulses. Then the glottal pulses go through your vocal tract, which acts as a filter. The vocal tract adds information about the timbre, the phonetics, **the formants** before it finally comes out of your mouth as a word(s).

If we were to trace the speech signal, we would have the first graph of the following image:


<img src="https://github.com/IsbatIslam/SpeechRecognition/assets/106777293/c5d05a4c-10ef-4b75-b3a7-da949e705ecc">
Link for image: https://www.dsprelated.com/freebooks/sasp/Spectral_Envelope_Examples.html

The first image is a simple time domain. The second graph is the frequency domain of the initial signal.  We can divide the frequenciy domain into its glottal pulse components, and its vocal tract components, called the **spectral detail** and the **spectral envelope**, respectively. In the second graph, if we were to trace a line relating all the peaks, we'd get the spectral envelope, the highest frequencies emitted which contains the information about pitch and phonetics. If we were to subtract the spectral envelope from the frequency domain, we'd only be left with dark zig-zags of the signal at very high frequency, which is the spectral detail.

For speech recognition, we are interested in the **spectral envelope** as it contains the most information about how the words are said, the timbre and, most importantly, the **formants**, which are peaks containing high amounts of energy, often indicating a vowel. And to do so, we resort to the **cepstrum** part of MFCC.

Let's take X(t) = G(t) * V(t) where X(t) = speech signal, G(t) = glottal pulses and V(t) = vocal tract

if we were to apply a log onto them, the equation would become:
log(X) = log(G) + log(V)

From this equation, we can separate both signals and end up solely with the spectral envelope, and the cepstrum helps us do so. But how?

<img src="https://github.com/IsbatIslam/SpeechRecognition/assets/106777293/27599c55-297b-4909-8c03-290a9608a8c1">

Link for image: https://www.researchgate.net/figure/Short-time-cepstrum-with-signal-to-noise-ratio-SNR-0-dB-and-reverberation-time-T-60_fig6_224009858

A cepstrum is a spectrum of a spectrum. Scientists decided to do a wordplay with the spectrum and switched the four first letters around. In short, we must apply an inverse Fourier transform on a Mel-spectrogram, which will give us a quefrency vs amplitude graph (shown above), and the peaks of those graphs are called rhamonics. Those rhamonics are related to harmonics (inverse first few letters), as they provide information about the pitch. Those quefrencies(freque + ncy --> quefre  +ncy) are then filtered via liftering (inverse the first few letters of filtering), which removes the glottal pulses from the the waveform signal, leaving us with spectral envelope. This filtering can be done with a discrete cosine transform since it is simpler than a Fourier transform. Indeed, it returns real values and not complex values, as we want a coefficient. Thus, the coefficients in MFCCs are unitless discrete values that the computer interprets as short term power spectrums. The more intense each signal of the spectral enveloppe is, the higher the value for the MFCC will be.

###Visualizing signals and code validation

<img src="https://github.com/IsbatIslam/SpeechRecognition/assets/106777293/ed98d9d1-df16-444a-8b8a-da82422fd00d">
The test audio "A4 61 key Piano.wav" is the test file to validate the code we used for the first term project. The fundamental frequency is supposed to be 440hz, and looking at the Mel-spectrogram here, the peak frequency appears to be that also. Another thing to consider is that the Mel-spectrogram uses the logarithmic frequency scale. But before 1000hz, the sound perception is generally linear, as obtained in our spectrogram. However, we can see the frequencies curve logarithmically as we climb higher frequencies (the scale at the left doubles between each increment, however the distance between the colored bars do not double as well) suggesting that the Mel-spectrogram is accurate. Not to mention, the decibel scale seems fairly accurate as well as the only intensities shown are by the 440hz note and its harmonics and everything is in dB's. The rest is black, as expected since there are no other noise in the pitch perfect A4 piano sample.
<img src="https://github.com/IsbatIslam/SpeechRecognition/assets/106777293/9701ea6d-0a84-417e-ae8c-8ac41cbb91ef">

The above is the MFCC of that signal. We see a strong signal around 1.7 seconds which is what we expect, as it fades away through time. The remaining coefficients are almost all 0, which is expected as the audio is silent. Therefore, the MFCC extractor works well and can be used to verify data later on. However, another thing to notice is the blue columns on the bottom. That column is simply a constant value added that does not contribute to the overall shape of the spectrum, it can be ignored.


I will now discuss the parameters as these parameters will reappear quite frequently throughout the project.


**sr** = SampleRate, which is how many samples are recorded per second of audio. For data preprocessing, the sr will be set at 22050Hz, which is right above the human hearing range (0-20000Hz). We call 22.05KHz the Nyquist frequency, which is the frequency exactly half of the sample rate of the audio (recorded at 44.1Khz) so that we can fully reconstruct the audio.

**n_fft** = number of samples per frame. This is the number of rows you want.

**hop_length** = number of frames between each n_fft's. They superpose with other frames and do a process called: "windowing." Windowing is when the amplitudes of discontinuities which causes artifacts are removed via the intersecting hop_length frames, causing a dampening of the middle portion where there is the "cut". A hop_length of 512 is generally recommended for speech processing, whereas a hop_length of 2048 is more suited for music.

**n_mfcc** = how many coefficients you want per column. 12 to 13 values are generally enough as they carry most of the information.

##Part 2 -- Creating Dataset
### Collecting Data
Firstly I have to collect data. Since I wanted to identify any digit between 1 to 5, I needed someone to say those numbers many times. However, I had to resort to using my voice as the sole source due to exams and time constraints. I used iPhone's Voice Memo app to record myself saying one 20 times, then say two 20 times, then three, four, and finally five. In total, I had 100 samples of my voice. However, I needed more for a proper machine learning project, so I doubled that to 200 by copy-pasting all the audio files. There will be a bias towards my voice, but my goal here is to get predictions.

I had to record samples of 1 second exactly, so I got samples of 2 seconds and trimmed each sample to 1 second, with only my voice waveforms in the audio. Then I named them like "one.001.wav" to keep track of them as I added them into their respective folder for loop-sorting code that I will implement.

The JSON data set looks like this:
<img src="https://github.com/IsbatIslam/SpeechRecognition/assets/106777293/17f72cd6-573b-4833-af9d-9284d9c76770">

When I do the MFCC of the audio file one.001.wav on my VsCode program, the extract_MFCCs(), for the first columns of MFCCs, I get:
'''
-4.62450500e+02 -4.62083496e+02 -4.80334564e+02 -4.82499298e+02
-4.99524048e+02 -5.31922607e+02 -5.40833130e+02 -5.40863892e+02
-4.89946442e+02 -4.18862579e+02 -3.74721405e+02 -3.69519867e+02
-3.74226685e+02 -3.32831421e+02 -2.91220520e+02 -2.79762726e+02
-2.62248962e+02 -2.37423996e+02 -2.26962082e+02 -2.25026672e+02
-2.26425659e+02 -2.42446487e+02 -2.67482697e+02 -2.96944244e+02
-3.37571198e+02 -4.02511047e+02 -4.20845917e+02 -4.27300079e+02
-4.60325653e+02 -4.78326599e+02 -4.92361847e+02 -4.93767853e+02
-4.84722900e+02 -5.01926514e+02 -5.14944275e+02 -5.24156250e+02
-5.26284851e+02 -5.22723022e+02 -5.25027283e+02 -5.26569519e+02
-5.25164124e+02 -5.28662720e+02 -5.30513489e+02 -5.30043701e+02
'''
The first value of -4.62450500e+02 is also what I obtain in my JSON file as in the JSON file. The first value is "-462.45050048828125," indicating that this code also works. I did more tests to double check, and they all correspond to the same thing, so it's safe to assume all 200 samples are good, as in part 1, the class AudioAnalysis has been validated.

If you do len(data["MFCCs"]), you will get 200, which are the 200 samples. If you do len(data["MFCCs"][0], you will get 13, which are the 13 columns of MFCC coefficients, as stated earlier. However, what explains the hundred thousands of data is:

SR = 22 050

SR/hop_lenght = 22 050/512 = 43.1 --> 44 columns in a row

44 * 13 = 572 vectors in a MFCC matrix

572 * 200 = 114 400 samples in audio_data.json


###Comparing Audio Samples
Here are one example per digit that I've said as I compare between them. I did not present more because they all generally follow the same pattern (four001.wav looks very similar to four013.wav for instance), so I don't want to extend the length of the presentation needlessely.

This is signal one:
<img src="https://github.com/IsbatIslam/SpeechRecognition/assets/106777293/18eb1a15-0016-4c2c-9fa6-9587bda5fdc3">

This is signal two:
<img src="https://github.com/IsbatIslam/SpeechRecognition/assets/106777293/8563bfdf-8f0e-4a70-94b9-ce98a1b819e6">

Disregarding the black and blue colour palette difference, notice that the signal when I say "one" seems to increase as time goes by before sharply declining, whereas when I say "two," the signal starts high. This is a difference that the computer must be able to pick up on to differentiate between both.

This is signal three:
<img src="https://github.com/IsbatIslam/SpeechRecognition/assets/106777293/c2e2b867-395a-4274-833e-e973342c90dc">

This is the signal five:
<img src="https://github.com/IsbatIslam/SpeechRecognition/assets/106777293/0a65ef01-390c-4eed-b7be-feb7aabbcbf8">

In this case, the difference between three and five seems to really show on the cloudy surface on top of three, whereas for five, the signal simply seems to start more intense than when we say three in the beginning.

This is signal 4:
<img src="https://github.com/IsbatIslam/SpeechRecognition/assets/106777293/df8cdd5e-d166-486e-9a1a-b9a828733e65">


Here are the MFCC's of the corresponding signals:

<img src="https://github.com/IsbatIslam/SpeechRecognition/assets/106777293/1f240ffb-3726-4797-add5-942e7ee6df0e">

<img src="https://github.com/IsbatIslam/SpeechRecognition/assets/106777293/bb7bd5b8-a84a-40b1-8855-da288f2264d0">

<img src="https://github.com/IsbatIslam/SpeechRecognition/assets/106777293/42c1d910-6407-4d42-b1fa-d3760c4bb9dc">

<img src="https://github.com/IsbatIslam/SpeechRecognition/assets/106777293/29ae1c2f-6508-4fe4-b4d1-bc248c901dcf">

<img src="https://github.com/IsbatIslam/SpeechRecognition/assets/106777293/2be41a75-d55f-4976-9988-dcfad1ef66c1">

By looking at MFCCs, the differences that we've seen on the Mel-spectrograms are hardly discernable in the MFCCs, especially for human eyes. However, each rectangular square represents an MFCC coefficient which the computer will see as a vector in 3-dimensions (#rows,#columns, depth), where rows would be 22050 Hz/ 512 ms = 44 rows, 13 columns through a monodimensional signal (44,13,1). These vectors will let the computer extract the prevalent features of each digit, including its timbre and formants.

##Part 3 -- Prediciting the digits
###Convolutional Neural Network

A Convolutional Neural Network, or CNN, is a deep learning algorithm that works a lot like how neurons work. In fact, it's inspired the Visual Cortex in the human brain. It tries to emulate the human visual system as it extracts different features. The CNN deep learning architecture is especially good for image recognition as it can really scan the image's features super efficiently. But if CNN is good for image classification, why use it for audio recognition? Well, we're using MFCCs Mel-spectrograms that can be considered as images. Thus we have an image classification problem in which CNNs excel.

<img src="https://github.com/IsbatIslam/SpeechRecognition/assets/106777293/e44c979b-a1c3-4694-b78d-ab5b51d57aad">

**Image** from Phani Ratan, site: https://www.analyticsvidhya.com/blog/2020/10/what-is-the-convolutional-neural-network-architecture/

There are first, **the convolution layers**. In the convolution layers, there are **kernels** that are applied to images via a dot product. Kernels are just grids with weights. After the kernels have been applied, it outputs a layer of the same dimensions as the initial image.

Here's an illustration:

<img src="https://github.com/IsbatIslam/SpeechRecognition/assets/106777293/7a16eb20-ce0e-4ea3-9045-68b7ee56254e">

We see in the image above that the convolution filter, which is the kernel, is being applied to the source pixel, which is one pixel of a whole picture. Then, on the output layer (destination pixel), the pixel will be in the same space but with the values of the dot product of all pixels in a 3x3 dimension around the source pixel. Thus, all the pixels around the middle get one value representative of all 3x3 values around the source pixel. However, in the corners and last horizontal and vertical pixels, since there are no more pixels beyond, we apply a **Zero Padding**, which replaces those empty spaces with zeroes so that the dot product can happen and give a value for the extremities. Thus, each convolution layer extracts the features, numbers, and shapes from raw numbers and vectors in the source image.


<img src="https://github.com/IsbatIslam/SpeechRecognition/assets/106777293/de55f870-c0b1-4970-88f1-583072daa565">
Image from Yugesh Verma, link: https://analyticsindiamag.com/guide-to-different-padding-methods-for-cnn-models/

Then there are the **Pooling layers**. Pooling simply downsamples the output of the convolution layers. In our project, we'll be using max pooling. They use "strides" that move from one row/column to the other. Depending on the stride value, it moves one row at a time, or two, ten, etc. We do MaxPooling to get the sharpest and most prevalent features out of each audio sample.

<img src="https://github.com/IsbatIslam/SpeechRecognition/assets/106777293/d774c6ef-09e7-4651-8d94-a3d9870a3941">

site: https://computersciencewiki.org/index.php/File:MaxpoolSample2.png

Thus, a CNN model uses these these two steps multiple times to extract the input image's features and classify them. Thus, for when it'll predict, if certain features matches the classified feature, the model would be able to predict what the signal it corresponds to.


To prepare the CNN model, run the code "create_cnn.py" and "predict_digit.py to start predicting"


First of all, I found this whole CNN building and training code thanks to "Valerio Velardo - The Sound of AI" from YouTube.
link: https://www.youtube.com/watch?v=INawFGUy-nU&list=PL-wATfeyAMNpCRQkKgtOZU_ykXc63oyzp&index=3


Now that we have the model, it's time to do predicitions.


As we see, I tried to put two files, five and one, and it worked, the computer could predict that I just said "five", and "one". This means that the objective of our project has been attained, however I want to push the testings further in the analysis section below. If you want to try this, add the audio files that I've put on the ZIP folders, and add them to google drive and then copy the the link from the "files" folder on the left side, in the taskbar.

## Analysis
I want to analyze my model and its actual accuracy rate using my own experiments. To do my own experiments, I will use the same dataset used to train the model and see its percentage of success. Then I will do this again, but with entirely new samples that I will record and try again to find how many it can get right.

I will do it with 50 samples, half of the original samples and a quarter of the total number of total samples. I think this is a good sample size to judge its accuracy. Then I will try the same tests, but on 20 new voice samples, other than from the dataset.

We cannot run the code above due to the files all being in my PC. I will send everything in a ZIP file so it would be possible to run this code by copy-pasting it. In the meantime, here's the output of my first test:

{'label': ['one', 'two', 'three', 'four', 'five'],

'predicted_digit': ['one', 'one', 'one', 'one', 'one', 'one', 'one', 'one', 'one', 'one', 'two', 'two', 'two', 'three', 'three', 'two', 'three', 'two', 'two', 'two', 'three', 'three', 'three',
'three', 'three', 'three', 'three', 'three', 'three', 'three', 'four', 'four', 'four', 'four', 'four', 'four', 'four', 'four', 'four', 'four', 'five', 'five', 'five', 'five', 'five', 'five', 'five', 'five', 'five', 'five']}

There was 10 tests, randomly picked for each digit, from the same dataset that trained my model.
Here's the success rate:

one: 10/10

two: 7/10, mixed up all three times with three

three: 10/10

four: 10/10

five: 10/10

I redid the test a second time, with 50 more completely random samples.

{'label': ['one', 'two', 'three', 'four', 'five'],

'predicted_digit': ['one', 'one', 'one', 'one', 'one', 'one', 'one', 'one', 'one', 'one', 'two', 'two', 'three', 'two', 'two', 'two', 'two', 'two', 'two', 'two', 'three', 'three', 'three', 'three', 'three', 'three', 'three', 'three', 'three', 'three', 'four', 'four', 'four', 'four', 'four', 'four', 'four', 'four', 'four', 'four', 'five', 'five', 'five', 'five', 'five', 'five', 'five', 'five', 'five', 'five']}

one: 10/10

two:9/10, mixed up with three again

three:10/10

four:10/10

five:10/10

Here are the tests with 20 samples out of the dataset, straight from my voice:



{'label': ['one', 'two', 'three', 'four', 'five'],

 'predicted_digit': ['four', 'one', 'one', 'four', 'two', 'three', 'two', 'two', 'three', 'three', 'five', 'three', 'four', 'four', 'four', 'four', 'five', 'five', 'five', 'five']}

one: 2/4, messed up two times with 4

two:3/4 messed up once with three

three:3/4 messed up once with five

four:4/4

five:4/4

## Conclusion

Throughout this project, I have attained my objective of creating a program that can predict what digit you're (me in this case) saying. I started by understanding the concept of MFCC, deepening my knowledge of Fourier transform, and learning how CNN works. However, as good as they seem, my results present a few significant issues.


## References

I would like to thank "Valerio Velardo -- The Sound of AI" from Youtube. His videos helped immensely and accelerated the research as, thanks to him, I knew where to look and what to do. The videos that came in especially handy are:


- Audio Signal Processing for Machine Learning Playlist
https://www.youtube.com/watch?v=iCwMQJnKk2c&t=1s


- Implementing a Speech Recognition System in TensorFlow 2
https://www.youtube.com/watch?v=INawFGUy-nU&t=2s


- Making Predictions with the Speech Recognition System
https://www.youtube.com/watch?v=cgkUcd-BFwA


- Preparing the Speech Dataset
https://www.youtube.com/watch?v=VPJ2jazh_KI

Other sources:

- A Comprehensive Guide to Convolutional Neural Networks — the ELI5 way by Sumit Saha
https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53

- Developing An Intuition for Fourier Transforms
https://sites.northwestern.edu/elannesscohn/2019/07/30/developing-an-intuition-for-fourier-transforms/


- a LOT of library and module documentations

















