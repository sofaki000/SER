import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

Ravdess = "C:\\Users\\Lenovo\\Desktop\\ser\\SER\\data\\sav\\audio_speech_actors_01-24\\"
Crema = "C:\\Users\\Lenovo\\Desktop\\ser\\SER\\data\\AudioWAV\\"
#Tess = "C:\\Users\\Lenovo\\Desktop\\ser\\SER\\data\\TESS Toronto emotional speech set data\\TESS Toronto emotional speech set data\\"
Tess= "C:\\Users\\Lenovo\\Desktop\\τεχνολογία ήχου και εικόνας\\FINAL\\SER\\data\\"
Savee = "C:\\Users\\Lenovo\\Desktop\\ser\\SER\\data\\AudioData\\"

def load_ravdess_dataset(load_all_data, number_of_samples_to_load_per_ds):
    # data preparation - RAVDESS
    ravdess_directory_list = os.listdir(Ravdess)

    file_emotion = []
    file_path = []

    counter1 = 0

    for dir in ravdess_directory_list:
        # as their are 20 different actors in our previous directory we need to extract files for each actor.
        actor = os.listdir(Ravdess + dir)
        for file in actor:
            part = file.split('.')[0]
            part = part.split('-')
            # third part in each file represents the emotion associated to that file.
            file_emotion.append(int(part[2]))
            file_path.append(Ravdess + dir + '/' + file)

            counter1 += 1

            if load_all_data is False and counter1 == number_of_samples_to_load_per_ds:
                break
        if load_all_data is False and counter1 == number_of_samples_to_load_per_ds:
            break
    # dataframe for emotion of files
    emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])

    # dataframe for path of files.
    path_df = pd.DataFrame(file_path, columns=['Path'])
    Ravdess_df = pd.concat([emotion_df, path_df], axis=1)

    # changing integers to actual emotions.
    Ravdess_df.Emotions.replace(
        {1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad', 5: 'angry', 6: 'fear', 7: 'disgust', 8: 'surprise'},
        inplace=True)
    Ravdess_df.head()
    return Ravdess_df

def load_crema_dataset(load_all_data,number_of_samples_to_load_per_ds):
    crema_directory_list = os.listdir(Crema)

    file_emotion = []
    file_path = []

    counter2 = 0

    for file in crema_directory_list:
        # storing file paths
        file_path.append(Crema + file)
        # storing file emotions
        part = file.split('_')
        if part[2] == 'SAD':
            file_emotion.append('sad')
        elif part[2] == 'ANG':
            file_emotion.append('angry')
        elif part[2] == 'DIS':
            file_emotion.append('disgust')
        elif part[2] == 'FEA':
            file_emotion.append('fear')
        elif part[2] == 'HAP':
            file_emotion.append('happy')
        elif part[2] == 'NEU':
            file_emotion.append('neutral')
        else:
            file_emotion.append('Unknown')

        if load_all_data is False and counter2 == number_of_samples_to_load_per_ds:
            break
        counter2 += 1

    # dataframe for emotion of files
    emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])


    # dataframe for path of files.
    path_df = pd.DataFrame(file_path, columns=['Path'])
    Crema_df = pd.concat([emotion_df, path_df], axis=1)

    return Crema_df

def load_tess_dataset(load_all_data, number_of_samples_to_load_per_ds):
    tess_directory_list = os.listdir(Tess)

    file_emotion = []
    file_path = []

    counter3 = 0
    for dir in tess_directory_list:
        directories = os.listdir(Tess + dir)
        for file in directories:
            part = file.split('.')[0]
            part = part.split('_')[2]
            if part == 'ps':
                file_emotion.append('surprise')
            else:
                file_emotion.append(part)
            file_path.append(Tess + dir + '\\' + file)
            counter3 += 1
            if load_all_data is False and counter3 == number_of_samples_to_load_per_ds:
                break
        if load_all_data is False and counter3 == number_of_samples_to_load_per_ds:
            break
    # dataframe for emotion of files
    emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])

    # dataframe for path of files.
    path_df = pd.DataFrame(file_path, columns=['Path'])

    Tess_df = pd.concat([emotion_df, path_df], axis=1)
    return Tess_df


def get_savee_dataset(load_all_data, number_of_samples_to_load_per_ds):
    file_emotion = []
    file_path = []

    counter4 = 0
    for dirname, _, filenames in os.walk(Savee):
        for filename in filenames:
            file_name = os.path.join(dirname, filename)

            if file_name.endswith(".wav") is False:
                continue

            file_path.append(file_name)

            label = filename[::-1].split('_')[0][::-1]

            if label[:1] == 'a':
                file_emotion.append('angry')
            elif label[:1] == 'd':
                file_emotion.append('disgust')
            elif label[:1] == 'f':
                file_emotion.append('fear')
            elif label[:1] == 'h':
                file_emotion.append('happy')
            elif label[:1] == 'n':
                file_emotion.append('neutral')
            elif label[:1] == 's':
                if label[:2] == 'sa':
                    file_emotion.append('sad')
                else:
                    file_emotion.append('surprise')
            counter4 += 1
            if load_all_data is False and counter4 == number_of_samples_to_load_per_ds:
                break
        if load_all_data is False and counter4 == number_of_samples_to_load_per_ds:
            break

    emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])
    path_df = pd.DataFrame(file_path, columns=['Path'])

    Savee_df = pd.concat([emotion_df, path_df], axis=1)
    return Savee_df

def get_dataframe_with_all_datasets(number_of_samples_to_load=20):
    load_all_data = False
    if number_of_samples_to_load==-1:
        load_all_data = True
        number_of_samples_to_load_per_ds = 0
    else:
        number_of_samples_to_load_per_ds= int(number_of_samples_to_load/2)

    #Ravdess_df = load_ravdess_dataset(load_all_data, number_of_samples_to_load_per_ds)
   # Crema_df = load_crema_dataset(load_all_data, number_of_samples_to_load_per_ds)
    Tess_df = load_tess_dataset(load_all_data, number_of_samples_to_load_per_ds)
    #Savee_df = get_savee_dataset(load_all_data, number_of_samples_to_load_per_ds)


    # creating Dataframe using all the 4 dataframes we created so far.
    # data_path = pd.concat([Ravdess_df, Crema_df, Tess_df, Savee_df], axis = 0)
    # data_path = pd.concat([Crema_df, Tess_df], axis=0)
    data_path = pd.concat([Tess_df], axis=0)
    data_path.to_csv("data_path.csv",index=False)
    data_path.head()
    return data_path

def get_dataframe_with_one_dataset(number_of_samples_to_load=20):
    load_all_data = False
    if number_of_samples_to_load==-1:
        load_all_data = True
        number_of_samples_to_load_per_ds = 0
    else:
        number_of_samples_to_load_per_ds= int(number_of_samples_to_load/2)

    # Ravdess_df = load_ravdess_dataset(load_all_data, number_of_samples_to_load_per_ds)
    # Crema_df = load_crema_dataset(load_all_data, number_of_samples_to_load_per_ds)
    Tess_df = load_tess_dataset(load_all_data, number_of_samples_to_load_per_ds)
    # Savee_df = get_savee_dataset(load_all_data, number_of_samples_to_load_per_ds)


    # creating Dataframe using all the 4 dataframes we created so far.
    # data_path = pd.concat([Ravdess_df, Crema_df, Tess_df, Savee_df], axis = 0)
    data_path = pd.concat([Tess_df], axis=0)
    data_path.to_csv("data_path.csv",index=False)
    data_path.head()
    return data_path

#data_path = pd.concat([Tess_df], axis = 0)
# Tess_df.to_csv("Tess_df.csv",index=False)
# Tess_df.head()
# data = pd.read_csv("Tess_df.csv")
# em = []
# for i in data.Emotions:
#     em.append(i)
# print(em)

def plot_emotion_dist(data_path):
    #emotions distribution in dataset
    plt.title('Count of Emotions', size=16)
    sns.histplot(data_path.Emotions)
    plt.ylabel('Count', size=12)
    plt.xlabel('Emotions', size=12)
    sns.despine(top=True, right=True, left=False, bottom=False)
    plt.show()