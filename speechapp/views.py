from django.shortcuts import render
from django.http import HttpResponse
# Create your views here.

#import pydub
import librosa
import numpy as np
#from pydub import AudioSegment
import os
import wave
import json
#import matplotlib.pyplot as plt


def WAV(wav_path):
    # pydub.AudioSegment.converter = "/home/speech/ffmpeg/bin/./ffmpeg.exe"
    # MP3_File = AudioSegment.from_mp3(file=mp3_path)
    # MP3_File.export(wav_path, format="wav")
    y, sr = librosa.load(wav_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=1)
    a = mfccs[:, : 4]
    return a


def Read_WAV(wav_path):
    """
    这是读取wav文件的函数，音频数据是单通道的。返回json
    :param wav_path: WAV文件的地址
    """
    wav_file = wave.open(wav_path, 'r')
    numchannel = wav_file.getnchannels()  # 声道数
    samplewidth = wav_file.getsampwidth()  # 量化位数
    framerate = wav_file.getframerate()  # 采样频率
    numframes = wav_file.getnframes()  # 采样点数
    print("channel", numchannel)
    print("sample_width", samplewidth)
    print("framerate", framerate)
    print("numframes", numframes)
    Wav_Data = wav_file.readframes(numframes)
    Wav_Data = np.fromstring(Wav_Data, dtype=np.int16)
    Wav_Data = Wav_Data * 1.0 / (max(abs(Wav_Data)))  # 对数据进行归一化
    #  生成音频数据,ndarray不能进行json化，必须转化为list，生成JSON
    dict = {"channel": numchannel,  # 声道数
            "samplewidth": samplewidth,  # 量化位数
            "framerate": framerate,  # 采样频率
            "numframes": numframes,  # 采样点数
            "WaveData": list((Wav_Data))}
    return json.dumps(dict)


# MP3文件和WAV文件的地址
# path = 'D://python//speech1/speech_recognition//ill'
path = '/home//speech//speech_recognition//ill'
paths = os.listdir(path)
wav_paths = []
# 获取WAV文件的相对地址
for wav_path in paths:
    wav_paths.append(path + "//" + wav_path)
print('wav_paths', wav_paths)
# # 得到MP3文件对应的WAV文件的相对地址
# wav_paths = []
# for mp3_path in mp3_paths:
#     wav_path = path2 + "//" + mp3_path.split('//')[-1].split('.')[0] + '.wav'
#     wav_paths.append(wav_path)


z = []
x = []
for wav_path in wav_paths:
    a = WAV(wav_path)
    x.extend(a)
    z.append('宝宝生病了')
print(x)
print(z)
# wav文件返回json，并且绘制频谱图
# for wav_path in wav_paths:
#     wav_json = Read_WAV(wav_path)
#     print('wav_json', wav_json)
#     wav = json.loads(wav_json)
#     wav_data = wav['WaveData']
#     framerate = int(wav['framerate'])
#     DrawSpectrum(wav_data, framerate)

# MP3文件和WAV文件的地址
# path = 'D://python//speech1/speech_recognition//sleepy'
path = '/home//speech//speech_recognition//sleepy'
paths = os.listdir(path)
wav_paths = []
# 获取WAV文件的相对地址
for wav_path in paths:
    wav_paths.append(path + "//" + wav_path)
print('wav_paths', wav_paths)
# # 得到MP3文件对应的WAV文件的相对地址
# wav_paths = []
# for mp3_path in mp3_paths:
#     wav_path = path2 + "//" + mp3_path.split('//')[-1].split('.')[0] + '.wav'
#     wav_paths.append(wav_path)
for wav_path in wav_paths:
    a = WAV(wav_path)
    x.extend(a)
    z.append('宝宝困了')
print(x)
print(z)
# wav文件返回json，并且绘制频谱图
# for wav_path in wav_paths:
#     wav_json = Read_WAV(wav_path)
#     print('wav_json', wav_json)
#     wav = json.loads(wav_json)
#     wav_data = wav['WaveData']
#     framerate = int(wav['framerate'])
#     DrawSpectrum(wav_data, framerate)


# # MP3文件和WAV文件的地址
# path = 'D://python//speech1/speech_recognition//hungry'
path = '/home//speech//speech_recognition//hungry'
paths = os.listdir(path)
wav_paths = []
# 获取WAV文件的相对地址
for wav_path in paths:
    wav_paths.append(path + "//" + wav_path)
print('wav_paths', wav_paths)
# # 得到MP3文件对应的WAV文件的相对地址
# wav_paths = []
# for mp3_path in mp3_paths:
#     wav_path = path2 + "//" + mp3_path.split('//')[-1].split('.')[0] + '.wav'
#     wav_paths.append(wav_path)
# print('wav_paths', wav_paths)
for wav_path in wav_paths:
    a = WAV(wav_path)
    x.extend(a)
    z.append('宝宝饿了')
print(x)
print(z)
# # wav文件返回json，并且绘制频谱图
# # for wav_path in wav_paths:
# #     wav_json = Read_WAV(wav_path)
# #     print('wav_json', wav_json)
# #     wav = json.loads(wav_json)
# #     wav_data = wav['WaveData']
# #     framerate = int(wav['framerate'])
# #     DrawSpectrum(wav_data, framerate)
#
# 训练测试集分离
from sklearn.model_selection import train_test_split
x_train, x_test, z_train, z_test = train_test_split(x, z, test_size=0.2, random_state=666)

# 通过管道标准化高斯核函数的svm分类器
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline


# 选用高斯核函数
def RBFkernelSVC(gamma):
    return Pipeline([
        ('std_scaler', StandardScaler()),
        ('SVC', SVC(kernel='rbf', gamma=gamma))
    ])


# gamma越大，高斯函数越窄，过拟合
# gamma越小，高斯函数越宽，欠拟合
svc = RBFkernelSVC(gamma=1)
svc.fit(x_train, z_train)
# print(svc.score(x_test, z_test))


def index(request):
    # MP3文件和WAV文件的地址
    try:
        if request.method == 'POST':
            file = request.FILES.get('myfile', None)
            # with open('D://python//speech1/speech_recognition//test//%s' % file.name, 'wb+') as f:
            with open('/home//speech//speech_recognition//test//%s' % file.name, 'wb+') as f:
                for chunk in file.chunks():
                    f.write(chunk)
            return render(request, 'speechapp/speech.html')
    except Exception as e:
        # with open('D://python//speech1//error','w') as error:
        with open('/home/speech/error','w') as error:
            error.write(str(e))
        return render(request, 'speechapp/speech.html')
    try:
        if request.method == 'GET':
            # path_test = 'D://python//speech1/speech_recognition//test'
            path_test = '/home//speech//speech_recognition//test'
            paths_test = os.listdir(path_test)
            wav_paths_test = []
            for wav_path_test in paths_test:
                wav_paths_test.append(path_test + "//" + wav_path_test)
            print('wav_paths_test', wav_paths_test)
            # # 得到MP3文件对应的WAV文件的相对地址
            # wav_paths_test = []
            # for mp3_path_test in mp3_paths_test:
            #     wav_path_test = path2_test + "//" + mp3_path_test.split('//')[-1].split('.')[0] + '.wav'
            #     wav_paths_test.append(wav_path_test)
            # print('wav_paths_test', wav_paths_test)
            x_test = []
            # 将MP3文件转化成WAV文件
            for wav_path_test in wav_paths_test:
                a = WAV(wav_path_test)
                x_test.extend(a)
            print(x_test)
            z_predict = svc.predict(x_test)
            return render(request, 'speechapp/speech.html', {'z_predict': z_predict})
    except:
        return render(request, 'speechapp/speech.html')