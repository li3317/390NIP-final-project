# 390NIP-final-project
CS390 final project

## Speech Style Transfer
___________
#### PHONETIC POSTERIORGRAMS FOR MANY-TO-ONE VOICE CONVERSIONWITHOUT PARALLEL DATA TRAINING
1. Tried to Implement  Paper
    https://www.researchgate.net/publication/307434911_Phonetic_posteriorgrams_for_many-to-one_voice_conversion_without_parallel_data_training
2. The model is seperated into two seperated trainable network
3. First network is trained to output phoneme and duration graph (PPG) for certain frequency
4. First network convert audio file into spectrum graph to process the input with the timit dataset
5. Second network insert its input audio file to first trained network to get PPG
6. Second network get relationship between PPG and MCEPs
7. The whole model insert PGG to Second Network to get MCEP correspond to it perform conversion.
8. However, this implementation stopped at research stage because structure of model is too complicated and TIMIT Dataset cost hug money to using it
-----------
#### Voice style transfer with random CNN
1. Found a project with simpler concept (similar to image style transfer) with no dataset requirement
https://github.com/mazzzystar/randomCNN-voice-transfer
2. Decided to do tranlation from pytorch to tensorflow
2. Finished Creating Pipeline, Loading, Preprocessing, Saving the Processed Data
3. Tried to find pretrained model for audio as a replacement for vgg in image style transfer
4. Failed to get speech related pretrained model like wavenet, ctc
5. Tried to use VGG instead by converting the spectrum graph (frequency vs amplitude) image insead of raw data
6. Failed due to input shape limitation of vgg
7. Successfully got content, style, total loss value in tensorflow 1, but failed to get gradient
8. Decided to do translation with tensorflow 2 instead.