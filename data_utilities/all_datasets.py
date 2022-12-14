import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

def get_dataframe_with_all_datasets():
    #paths for my data
    Ravdess = "C:\\Users\\user\\PycharmProjects\\SER_4_datasets\\audio_speech_actors_01-24\\"
    Crema = "C:\\Users\\user\\PycharmProjects\\SER_4_datasets\\AudioWAV\\"
    Tess = "C:\\Users\\user\\PycharmProjects\\SER_4_datasets\\TESS_Toronto_emotional_speech_set_data\\"
    Savee = "C:\\Users\\user\\PycharmProjects\\SER_4_datasets\\AudioData\\"

    #data preparation - RAVDESS
    ravdess_directory_list = os.listdir(Ravdess)

    file_emotion = []
    file_path = []
    for dir in ravdess_directory_list:
        # as their are 20 different actors in our previous directory we need to extract files for each actor.
        actor = os.listdir(Ravdess + dir)
        for file in actor:
            part = file.split('.')[0]
            part = part.split('-')
            # third part in each file represents the emotion associated to that file.
            file_emotion.append(int(part[2]))
            file_path.append(Ravdess + dir + '/' + file)
    # dataframe for emotion of files
    emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])

    # dataframe for path of files.
    path_df = pd.DataFrame(file_path, columns=['Path'])
    Ravdess_df = pd.concat([emotion_df, path_df], axis=1)

    # changing integers to actual emotions.
    Ravdess_df.Emotions.replace({1:'neutral', 2:'calm', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust', 8:'surprise'}, inplace=True)
    Ravdess_df.head()

    #data preparation - CREMA
    crema_directory_list = os.listdir(Crema)

    file_emotion = []
    file_path = []

    for file in crema_directory_list:
        # storing file paths
        file_path.append(Crema + file)
        # storing file emotions
        part=file.split('_')
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

        # dataframe for emotion of files
        emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])

        # dataframe for path of files.
        path_df = pd.DataFrame(file_path, columns=['Path'])
        Crema_df = pd.concat([emotion_df, path_df], axis=1)
        Crema_df.head()

    #data preparation - TESS
    tess_directory_list = os.listdir(Tess)

    file_emotion = []
    file_path = []

    for dir in tess_directory_list:
        directories = os.listdir(Tess + dir)
        for file in directories:
            part = file.split('.')[0]
            part = part.split('_')[2]
            if part=='ps':
                file_emotion.append('surprise')
            else:
                file_emotion.append(part)
            file_path.append(Tess + dir + '\\' + file)

    # dataframe for emotion of files
    emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])

    # dataframe for path of files.
    path_df = pd.DataFrame(file_path, columns=['Path'])

    Tess_df = pd.concat([emotion_df, path_df], axis=1)
    Tess_df.head()

    #data preparation - SAVEE
    savee_directory_list = os.listdir(Savee)

    file_emotion = []
    file_path = []

    for file in savee_directory_list:
        file_path.append(Savee + file)
        part = file.split('_')[0]
        ele = part[:-6]
        if ele=='a':
            file_emotion.append('angry')
        elif ele=='d':
            file_emotion.append('disgust')
        elif ele=='f':
            file_emotion.append('fear')
        elif ele=='h':
            file_emotion.append('happy')
        elif ele == 'n':
            file_emotion.append('neutral')
        elif ele == 'sa':
            file_emotion.append('sad')
        else:
            file_emotion.append('surprise')

        # dataframe for emotion of files
        emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])

        # dataframe for path of files.
        path_df = pd.DataFrame(file_path, columns=['Path'])
        Savee_df = pd.concat([emotion_df, path_df], axis=1)
        Savee_df.head()

    # creating Dataframe using all the 4 dataframes we created so far.
    data_path = pd.concat([Ravdess_df, Crema_df, Tess_df, Savee_df], axis = 0)
    data_path.to_csv("data_path.csv",index=False)
    data_path.head()

#data_path = pd.concat([Tess_df], axis = 0)
# Tess_df.to_csv("Tess_df.csv",index=False)
# Tess_df.head()
# data = pd.read_csv("Tess_df.csv")
# em = []
# for i in data.Emotions:
#     em.append(i)
# print(em)

def plot_emotion_dist():
    #emotions distribution in dataset
    plt.title('Count of Emotions', size=16)
    sns.histplot(data_path.Emotions)
    plt.ylabel('Count', size=12)
    plt.xlabel('Emotions', size=12)
    sns.despine(top=True, right=True, left=False, bottom=False)
    plt.show()