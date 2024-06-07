import numpy as np

import tensorflow as tf
from tensorflow import keras

from recording_helper import record_audio, terminate
from tf_helper import preprocess_audiobuffer

commands = ['down', 'go', 'left', 'no', 'right', 'stop', 'up', 'yes']

import tensorflow as tf

loaded_model = keras.models.load_model('saved_model04.keras')
loaded_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

def predict_mic():
    audio = record_audio()
    spec = preprocess_audiobuffer(audio)
    prediction = loaded_model(spec)
    label_pred = np.argmax(prediction, axis=1)
    command = commands[label_pred[0]]
    print("Izgovoreno:", command)
    return command

if __name__ == "__main__":
    while True:
        command = predict_mic()
        if command == "stop":
            terminate()
            break