from keras import Model, Sequential
from keras.layers import Reshape, Dropout, GRU
from keras.layers import LSTM
from keras.layers import Input, Dense
from keras_models.attention_model import Attention


def get_lstm_model_with_attention(time_steps, input_dim,  output_classes):
    model_input = Input(shape=(time_steps, input_dim))
    x = LSTM(64, return_sequences=True, name="lstm")(model_input)
    x = Attention(units=32)(x)
    x = Reshape(target_shape=(1,32))(x)
    x = LSTM(64, return_sequences=True, name="lstm_2")(x)
    x = Attention(units=32)(x)
    x = Dense(output_classes)(x)
    model = Model(model_input, x)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy')
    return model


def get_lstm_model(time_steps, input_dim,output_classes):
    model_input = Input(shape=(time_steps, input_dim))
    x = LSTM(64, return_sequences=True, name="lstm")(model_input)
    x = LSTM(64, return_sequences=True, name="lstm_2")(x)
    x = Dense(output_classes)(x)
    model = Model(model_input, x)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy')
    return model


# 256, 128,128,64
# 256  256 512 512
def get_lstm_model_with_dropout_and_attention(num_features, output_classes):
    model = Sequential([
        LSTM(256, return_sequences=True, input_shape=(num_features,1)),
        Attention(units=256),
        Dropout(0.2),
        Dense(512, activation='relu'),
        Dropout(0.2),
        Dense(512, activation='relu'),
        Dropout(0.2),
        Dense(output_classes, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def get_lstm_model_with_dropout_and_more_attention_layers(num_features, output_classes):
    model = Sequential([
        LSTM(256, return_sequences=True, input_shape=(num_features,1)),
        Attention(units=256),
        Dropout(0.2),
        Dense(512, activation='relu'),
        Dropout(0.2),
        Dense(512, activation='relu'),
        Dropout(0.2),
        Dense(output_classes, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

#256,128,64
# 256 512 512
def get_lstm_model_with_dropout(num_features,output_classes):
    model = Sequential([
        LSTM(256, return_sequences=False, input_shape=(num_features,1)),
        Dropout(0.2),
        Dense(512, activation='relu'),
        Dropout(0.2),
        Dense(512, activation='relu'),
        Dropout(0.2),
        Dense(output_classes, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

