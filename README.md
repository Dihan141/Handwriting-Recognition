# Handwriting-Recognition
Machine learning project for recognizing handwritten texts

Instructions:
Must need python 3.10 version.
I repeat!!! Must need python 3.10

Prerequisites:
    You must have CUDA version 11.2 and cudNNN library version 8.1

Dataset:
    Move IAM_Words into project directory

Setup Instructions:
1. Setup virtual environment:
    python -m venv [name]
2. Activate virtual environment:
    [name]\Scripts\activate
3. Install packages from requirements
    pip install -r requirements.txt
4. Run train.py if not trained

Documentation:
1. Overview
The Handwriting Recognition model is built using TensorFlow and Keras. It consists of several components for data processing, model training, and inference.

2. Code Structure

    train.py: Contains the script for training the handwriting recognition model.

    configs.py: Defines the configuration settings for the model, such as paths, dimensions, and training parameters.

    inferenceModel.py: Implements a class for using the trained model for inference on new images.

    model.py: Defines the architecture of the handwriting recognition model using residual blocks and bidirectional LSTM layers.

    SpellChecking.py: Implements a spell checker using the spellchecker library, providing suggestions for word corrections.
    
    Canvas.py: Defines the GUI using the Tkinter library. Users can draw on a canvas, submit drawings for recognition, and receive predictions with suggested corrections.


3. Training
    3.1. Dataset Preparation
        The IAM Words dataset is used for training. The dataset is preprocessed, and  image paths, labels, and vocabulary are extracted from the dataset.

    3.2. Model Configuration
        Model configuration is handled by the ModelConfigs class in configs.py. It includes settings for paths, image dimensions, batch size, learning rate, and training epochs.

    3.3. Data Processing
        The DataProvider class is used for loading and processing the dataset. Data augmentation techniques, such as random brightness, rotation, and erosion/dilation, are applied to the training data.

    3.4. Model Architecture
        The model architecture is defined in model.py. It utilizes residual blocks and bidirectional LSTM layers.

    3.5. Model Training
        The model is compiled using the CTC loss function and trained using the specified callbacks such as early stopping and model checkpointing. Training progress is logged, and the resulting model is saved im Models directory

    3.6. Callbacks
        Several callbacks are used during training, such as early stopping, model checkpointing, and logging for TensorBoard.

    3.7. Results
        Training and validation datasets are saved as CSV files, and model configurations are stored for reference.


4. Inference
    4.1. Loading Trained Model
        The trained model is loaded using the ImageToWordModel class in inferenceModel.py. The class uses an ONNX inference model and includes a method for making predictions on new images.

    4.2. Recognition Process
        The drawn image is converted to an RGB PIL Image for further processing. Preprocessing is applied to adjust the image size. The preprocessed image is passed through the trained model for text prediction. The predicted text is displayed in the GUI. Suggestions for corrections are shown based on the predicted text.

    4.3. Inference on Validation Data
        The trained model is applied to the validation dataset. Predictions are made, and Character Error Rate(CER) is calculated for each prediction.

    4.4. Visualization
        Images and their predictions are displayed, and average CER is computed.