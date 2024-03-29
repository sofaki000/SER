from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, GRU
from keras.utils.vis_utils import plot_model

from keras_models.attention_model import Attention


#123,64,32
# 256,512,512-> better train, worse test
def get_gru_model(num_features,output_classes):
    model = Sequential([
        GRU(126, return_sequences=False, input_shape=(num_features, 1)),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(output_classes, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def get_gru_model_with_more_layers(num_features,output_classes):
    model = Sequential([
        GRU(126, return_sequences=False, input_shape=(num_features, 1)),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(output_classes, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def get_gru_model_with_attention_and_more_layers(num_features,output_classes):
    model = Sequential([
        GRU(126, return_sequences=True, input_shape=(num_features, 1)),
        Attention(units=128),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(output_classes, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
#123,32, 64,32
# 256,512,512-> better train, worse test
def get_gru_model_with_attention(num_features,output_classes):
    model = Sequential([
        GRU(126, return_sequences=True, input_shape=(num_features, 1)),
        Attention(units=32),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(output_classes, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    plot_model(model, 'dense_model.png', show_shapes=True)
    return model

