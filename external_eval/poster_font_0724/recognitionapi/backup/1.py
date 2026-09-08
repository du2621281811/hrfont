import os
import codecs


with codecs.open("./fontlist_5330-with_path.txt",mode='r',encoding='utf-8'):
    data1 = f.readlines()
f.close()

data1 = [info.strip() for info in data1]



with codecs.open("./常用字_107.txt",mode='r',encoding='utf-8'):
    data1 = f.readlines()
f.close()

data2 = [info.strip() for info in data2]


filename = "/root/新加卷3/songpeng/FontRecognition/fonts_14345"


for temp in data1:
    if temp.split(",,,")[1] in data2:
        continue
    else:
        if int(temp.split(",,,")[0]) < 2000000 and "":


