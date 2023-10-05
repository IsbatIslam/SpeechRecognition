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
![image.png](data:image/png;base64,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)


