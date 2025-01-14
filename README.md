# Classifying-epilepsy-from-spike-count-and-amplitude-from-EEG-data
Machine Learning Project: Sequence Classification with CNN and RNN

Overview

This project implements sequence classification using both Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs). The goal is to classify labeled subword sequences provided in the file lamstar_input_labeled.txt into binary classes. The project uses TensorFlow and Keras for model implementation and includes data preprocessing, model training, and performance evaluation.

Project Structure

Data Preprocessing: Converts input data from the provided file into numerical sequences and encodes labels for training.

CNN Model: A 1D convolutional neural network designed for feature extraction and classification.

RNN Model: A recurrent neural network with LSTM layers designed for handling sequential data.

Evaluation: Performance metrics such as confusion matrix, classification report, and accuracy are generated for both models.

Requirements

The project requires the following libraries:

Python 3.7+

TensorFlow/Keras

scikit-learn

NumPy

Matplotlib

Seaborn

Install the required libraries using pip:

pip install tensorflow scikit-learn numpy matplotlib seaborn

Data Format

The input file lamstar_input_labeled.txt should be a tab-separated file with the following format:

segment	subwords	label
1	0.1;0.2;0.3	ClassA
2	0.4;0.5;0.6	ClassB
...

segment: Identifier for the sequence (not used in training).

subwords: Semi-colon-separated numeric values representing the sequence.

label: Class label for the sequence.

CNN Model

The CNN model architecture includes:

Conv1D layers for feature extraction.

MaxPooling1D layers for downsampling.

Dropout for regularization.

Fully connected Dense layers with ReLU and sigmoid activations for binary classification.

Training Parameters

Epochs: 30

Batch Size: 16

Loss Function: Binary Cross-Entropy

Optimizer: Adam

Evaluation

The CNN model outputs:

Confusion matrix

Classification report (precision, recall, F1-score)

Accuracy score

RNN Model

The RNN model architecture includes:

Two LSTM layers (64 and 32 units).

Dropout layers for regularization.

Dense layers with ReLU and sigmoid activations for binary classification.

Training Parameters

Epochs: 500

Batch Size: 16

Loss Function: Binary Cross-Entropy

Optimizer: Adam

Evaluation

The RNN model outputs:

Accuracy score

Running the Project

Place the input file lamstar_input_labeled.txt in the project directory.

Run the Python scripts for the CNN and RNN models to preprocess the data, train the models, and evaluate their performance.

Example for running the RNN model:

python rnn_model.py

Example for running the CNN model:

python cnn_model.py

Results

Both models evaluate the performance on a test set and output:

Confusion matrix visualization

Classification metrics

Acknowledgments

This project leverages TensorFlow/Keras for deep learning and scikit-learn for preprocessing and evaluation. Special thanks to open-source contributors for these tools.

License

This project is licensed under the GNU License. See the LICENSE file for details.

