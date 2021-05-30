#!/usr/bin/env python
# coding: utf-8

# In[2]:


from IPython.display import display
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import os
import shutil
import posixpath

import wfdb
import pywt


# ## Удаление шумов

# Для удаления шумов используется вейлвет преобразование. В качестве вейвлета используется вейвлет Добеши 6, который разлагает необработанный сигнал на 6 уровней. Восстановление сигнала происходит обратным вейвлет преобразованием, в котором участвуют вейвлет-коэффициенты с 3го по 6ой уровень. 

# In[46]:


from matplotlib import pyplot as plt

#loop for each signal from MIT-BIH dataset

list_of_trouble_signals = []

###Init_path
YOUR_PATH_SOURCE_DATA = 'physionet.org/files/mitdb/1.0.0/'
YOUR_PATH_PREP_DATA = 'physionet.org/prep_files/'
PATH_WITH_LIST_OF_SIGNALS = 'physionet.org/files/mitdb/1.0.0/RECORDS'

with open(PATH_WITH_LIST_OF_SIGNALS, 'r') as fp:
    for line in fp:
        line = line.rstrip('\n')
        record = wfdb.rdrecord(YOUR_PATH_SOURCE_DATA + line, physical=False)
        annotation = wfdb.rdann(YOUR_PATH_SOURCE_DATA + line, 'atr')
        ### Some bug???
        if min(annotation.subtype) < 0 or max(annotation.subtype) > 127:
            list_of_trouble_signals.append(line)
            continue
        
        # get an array and perform a db6 wavelet transformation   
        X = record.d_signal[:, 0]
        coeffs = pywt.wavedec(X, 'db6', level=6)
        
        # ingor first two levels:
        coeffs[-1] = np.zeros_like(coeffs[-1])
        coeffs[-2] = np.zeros_like(coeffs[-2])
        
        #Inverse Discrete Wavelet Transform
        new_d_signal = pywt.waverec(coeffs, 'db6')
        
        #Check denoise signal for allowed range(Some Bug???)
        if max(new_d_signal) > 2047 or min(new_d_signal) < -2048:
            list_of_trouble_signals.append(line)
            continue
        
        #Create a new denoise signal
        record.d_signal[:, 0] = new_d_signal
        record.wrsamp(write_dir=YOUR_PATH_PREP_DATA)
        annotation.wrann(write_dir=YOUR_PATH_PREP_DATA)


# ### Неожиданности

# В результате удаления шумов на некоторых сигналах произошло непредсказуемое поведение. Чтобы не обрывать весь цикл, в коде комментариями "Some Bug" отмечены участки, которые игнорируют эти проблемы. Список таких сигналов записан в list_of_trouble_signals. Проанализируем ошибки в этих сигналах.

# In[47]:


list_of_trouble_signals


# In[48]:


#Some Analysis(101.dat)


record = wfdb.rdrecord(YOUR_PATH_SOURCE_DATA + list_of_trouble_signals[0], physical=False)
annotation = wfdb.rdann(YOUR_PATH_SOURCE_DATA + list_of_trouble_signals[0], 'atr')
X = record.d_signal[:, 0]
coeffs = pywt.wavedec(X, 'db6', level=6)

coeffs[-1] = np.zeros_like(coeffs[-1])
coeffs[-2] = np.zeros_like(coeffs[-2])

new_d_signal = pywt.waverec(coeffs, 'db6')

record.d_signal[:, 0] = new_d_signal
record.wrsamp(write_dir=YOUR_PATH_PREP_DATA)
annotation.wrann(write_dir=YOUR_PATH_PREP_DATA)


# Ошибка поля subtype в классе Annotation. В классе Annotation есть 2 словаря: классы и метки. В документации к библиотеке wfdb метод subtype не понятно за что отвечает. Поэтому те элементы subtype, которые не удовлетворяют допустимым границам, просто занулим

# In[49]:


#Some Analysis(116.dat)

record = wfdb.rdrecord(YOUR_PATH_SOURCE_DATA + list_of_trouble_signals[2], physical=False)
annotation = wfdb.rdann(YOUR_PATH_SOURCE_DATA + list_of_trouble_signals[2], 'atr')
X = record.d_signal[:, 0]
coeffs = pywt.wavedec(X, 'db6', level=6)

coeffs[-1] = np.zeros_like(coeffs[-1])
coeffs[-2] = np.zeros_like(coeffs[-2])

new_d_signal = pywt.waverec(coeffs, 'db6')

record.d_signal[:, 0] = new_d_signal
record.wrsamp(write_dir=YOUR_PATH_PREP_DATA)
annotation.wrann(write_dir=YOUR_PATH_PREP_DATA)


# Вторая проблема связана с тем, что после прямого и обратного вейвлет преобразования, амплитуда сигнала вышла за недопустимые пределы. Поступим также, как и с subtype: Если амплитуда выходит из области допустимых значений, приведем ее граничной

# ### Код с учетом исправлений

# In[50]:


#Repare signals

for line in list_of_trouble_signals:
    record = wfdb.rdrecord(YOUR_PATH_SOURCE_DATA + line, physical=False)
    annotation = wfdb.rdann(YOUR_PATH_SOURCE_DATA + line, 'atr')
    X = record.d_signal[:, 0]
    coeffs = pywt.wavedec(X, 'db6', level=6)

    coeffs[-1] = np.zeros_like(coeffs[-1])
    coeffs[-2] = np.zeros_like(coeffs[-2])

    new_d_signal = pywt.waverec(coeffs, 'db6')
    new_d_signal[new_d_signal > 2047] = 2047
    new_d_signal[new_d_signal < -2048] = -2048

    record.d_signal[:, 0] = new_d_signal
    record.wrsamp(write_dir=YOUR_PATH_PREP_DATA)
    annotation.subtype[annotation.subtype < 0] = 0
    annotation.subtype[annotation.subtype > 127] = 0
    annotation.wrann(write_dir=YOUR_PATH_PREP_DATA)


# ## Нарезка Холтера_v1

# Нарезку холтера будем производить при помощи аннотированных R-пиков каждого сигнала: Отступ на 149 влево и на 150 вправо от пика.

# In[53]:


YOUR_PATH_PREP_SLICE_DATA = 'physionet.org/sliced_files/'

with open(PATH_WITH_LIST_OF_SIGNALS, 'r') as fp:
    for line in fp:
        line = line.rstrip('\n')
        record = wfdb.rdrecord(YOUR_PATH_PREP_DATA + line, physical=False, channels=[0])
        annotation = wfdb.rdann(YOUR_PATH_PREP_DATA + line, 'atr')
        for counter, test_ind in enumerate(annotation.sample):
            
            #edge case handling 
            if (test_ind - 150) < 0 or (test_ind+149) > record.sig_len:
                continue
                
            slice_record = wfdb.rdrecord(YOUR_PATH_PREP_DATA + line, sampfrom=test_ind - 150, 
                            sampto=test_ind+149, channels=[0], physical=False)           
            slice_annotation = wfdb.rdann(YOUR_PATH_PREP_DATA + line, 'atr', 
                                          sampfrom=test_ind - 150, sampto=test_ind + 149, 
                                          shift_samps=True)
            slice_record.record_name = line + '_' + str(counter)
            slice_annotation.record_name = line + '_' + str(counter)
            slice_record.wrsamp(write_dir=YOUR_PATH_PREP_SLICE_DATA)
            slice_annotation.wrann(write_dir=YOUR_PATH_PREP_SLICE_DATA)


# ## Нарезка Холтера_v2

# Предыдущая нарезка сохраняет каждый разрез как отдельный сигнал со своей заголовочной(.hea) и аннотированной(.atr) частью. Это неудобно при составлении общего датасета: необходимо перебрать каждый файл из папки, вытянуть из него необходимую информацию(частоту сигнала и метку класса) и собрать все в 1 массив. Сделаем это сразу, в процессе нарезки:

# ### Некоторые проблемы
# В датасете MIT иногда аннотируются не только R-пики.
# Например, могут аннотироваться шумы или изменения ритма. Процесс сегментирования холтера основан на том, что мы берем массив аннотаций и для каждой делаем отступ на 150 влево и вправо, формируя одно сердцебиение. Из-за того, что аннотируются не только R-пики, иногда происходит так, что сегмент из 300 точек включает в себя больше одной аннотации. В таком случае не понятно, как классифицировать сигнал(т.к. есть 2 анотации => 2 метки класса). Каждая такая ситуация индивидуальна(может быть и 3 аннотации и различные непредвиденные обстоятельства), поэтому такие случаи тяжело автоматизировать. Это также ведет к тому, что сегменты будут пересекаться(не могу пока что сказать, плохо это или нет). 
# 
# Поэтому этот код сегменты с более одной аннотацией игнорирует(соответствующий участок помечен комментарием). Далее посмотрим, сколько процентов данных относительно статьи мы из-за этого потеряли и не повлияло ли это на разнообразие классов(сверяться также будем с данными статьи).

# ### Пример такой аннотации

# In[9]:


YOUR_PATH_PREP_DATA = 'physionet.org/prep_files/'
PATH_WITH_LIST_OF_SIGNALS = 'physionet.org/files/mitdb/1.0.0/RECORDS'

flag = False

with open(PATH_WITH_LIST_OF_SIGNALS, 'r') as fp:
    for line in fp:
        line = line.rstrip('\n')
        record = wfdb.rdrecord(YOUR_PATH_PREP_DATA + line, physical=False, channels=[0])
        annotation = wfdb.rdann(YOUR_PATH_PREP_DATA + line, 'atr')
        for test_ind in annotation.sample:
            
            #edge case handling 
            if (test_ind - 150) < 0 or (test_ind+149) > record.sig_len:
                continue
                
            slice_record = wfdb.rdrecord(YOUR_PATH_PREP_DATA + line, sampfrom=test_ind - 150, 
                            sampto=test_ind+150, channels=[0], physical=False)           
            slice_annotation = wfdb.rdann(YOUR_PATH_PREP_DATA + line, 'atr', 
                                          sampfrom=test_ind - 150, sampto=test_ind + 150, 
                                          shift_samps=True)
            
            signal = slice_record.d_signal
            signal = signal.reshape(300)
            mark = slice_annotation.symbol
            
            
            if len(mark) > 1:
                wfdb.plot_wfdb(record=slice_record, annotation=slice_annotation, 
                               title='Abnormal signal', time_units='seconds')
                flag = True
                break
                
        if flag == True:
            break


# In[ ]:


YOUR_PATH_PREP_DATA = 'physionet.org/prep_files/'
PATH_WITH_LIST_OF_SIGNALS = 'physionet.org/files/mitdb/1.0.0/RECORDS'

X = np.array([])
y = np.array([])
data = np.array([])

with open(PATH_WITH_LIST_OF_SIGNALS, 'r') as fp:
    for line in fp:
        line = line.rstrip('\n')
        record = wfdb.rdrecord(YOUR_PATH_PREP_DATA + line, physical=False, channels=[0])
        annotation = wfdb.rdann(YOUR_PATH_PREP_DATA + line, 'atr')
        for test_ind in annotation.sample:
            
            #edge case handling 
            if (test_ind - 150) < 0 or (test_ind+149) > record.sig_len:
                continue
                
            slice_record = wfdb.rdrecord(YOUR_PATH_PREP_DATA + line, sampfrom=test_ind - 150, 
                            sampto=test_ind+150, channels=[0], physical=False)           
            slice_annotation = wfdb.rdann(YOUR_PATH_PREP_DATA + line, 'atr', 
                                          sampfrom=test_ind - 150, sampto=test_ind + 150, 
                                          shift_samps=True)
            
            signal = slice_record.d_signal
            signal = signal.reshape(300)
            mark = slice_annotation.symbol
            
            ##Ignore more than 1 label
            if len(mark) > 1:
                continue 
            
            if X.size == 0:
                X = np.append(X, signal)
                y = np.append(y, mark)
            else:
                X = np.concatenate([X, signal])
                y = np.concatenate([y, mark])

##Repare dimensions
X = X.reshape(y.shape[0], 300)
y = y.reshape(y.shape[0], 1)


# In[ ]:


##Save dataset
data = np.hstack((X, y))
np.save('physionet.org/dataset', data)


# ### Анализ полученного датасета

# In[3]:


data = np.load('physionet.org/dataset.npy')

X = data[:, :-1]
y = data[:, -1]


# In[4]:


print("The resulting number of ECG signals -- {}".format(X.shape[0]))
print("The number of ECG signals in the article -- 107679")
print("Lost data(as a percentage) -- {} %".format(100 - (X.shape[0] / 107679) * 100))


# In[5]:


types_of_classes = []
num = []
for label in y:
    if not label in types_of_classes:
        types_of_classes.append(label)
        num.append(1)
    else:
        i = types_of_classes.index(label)
        num[i] += 1


# In[7]:


print(types_of_classes)
print('Number of classes -- {}'.format(len(types_of_classes)))
print(num)


# После нарезки имеем 22 класса. В статье их 5([N, SVEB, VEB, F, Q]) , поэтому какие-то из них необходимо отнести в отдельный класс Unknown, какие-то отождествить. Эта задача уже выполнялась, например, тут https://github.com/hsd1503/PhysioNet . Поэтому воспользуемся таблицей в readme из гитхаба. 

# In[11]:


N_beat = ['N', 'L', 'R', 'B']
SVEB_beat = ['A', 'a', 'J', 'S', 'e', 'j', 'n']
VEB_beat = ['V', 'r', 'E']
F_beat = ['F']

for i, label in enumerate(y):
    if label in N_beat:
        y[i] = 'N'
    elif label in SVEB_beat:
        y[i] = 'S'
    elif label in VEB_beat:
        y[i] = 'V'
    elif label in F_beat:
        y[i] = 'F'
    else:
        y[i] = 'Q'


# Сравним теперь наполнение классов с данными со статьи.

# In[12]:


article_classes = ['N', 'S', 'V', 'F', 'Q']
art_num = [90265, 2503, 7106, 789, 7016]

types_of_classes = []
num = []
for label in y:
    if not label in types_of_classes:
        types_of_classes.append(label)
        num.append(1)
    else:
        i = types_of_classes.index(label)
        num[i] += 1
        
print('Summary from the article: Heartbeat type {} \n Number {}'.format(article_classes, art_num))
print('Our Summary: Heartbeat type {} \n Number {}'.format(types_of_classes, num))


# Видно, что сэмплов типа Unknown сильно больше(хотя данных у меня поменьше, чем в статье). Почему так я пока не знаю, но не доверять гитхабу я смысла не вижу. 

# In[13]:


## Change labels and save dataset

data[:, -1] = y
np.save('physionet.org/dataset', data)

