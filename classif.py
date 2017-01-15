# coding: utf-8

import collections
import re
import zipfile

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.cross_validation import cross_val_score

###################################################################
#      Чтение и запись.
###################################################################

CLASS2NUM = {}
NUM2CLASS = []

def raw_text_iter(zfname, fname, title_column):
    zf = zipfile.ZipFile(zfname)
    for line in zf.open(fname):
        line = line.split("\t")
        yield line[title_column].decode("utf-8"), line[title_column+1].decode("utf-8")
        
def read_classes(zfname, fname, class_column=0):
    zf = zipfile.ZipFile(zfname)
    result = []
    for line in zf.open(fname):
        class_name = line.split("\t")[class_column].strip()
        try:
            class_id = CLASS2NUM[class_name]
        except KeyError:
            class_id = len(CLASS2NUM)
            CLASS2NUM[class_name] = class_id
            NUM2CLASS.append(class_name)
        result.append(class_id)        
    return np.asarray(result)
    
def write_classes(Ys, fname="prediction.txt"):
    with open(fname, "wb") as f:
        f.write("\n".join(map(NUM2CLASS.__getitem__, Ys)))
        f.write("\n")        
    
###################################################################
#      Предобработка текста.
###################################################################

def data_iter(rt_iter, pipeline, marks=("title", "body", "both")):
    for title, body in rt_iter:
        for proc in pipeline:
            title, body = proc(title), proc(body)
        result = []    
        for m in marks:
            if m == "title":
                result.extend(["title_" + w for w in title])
            elif m == "body":
                result.extend(["body_" + w for w in body])
            else:
                result.extend(title)
                result.extend(body)
        yield " ".join(result)

def tok_proc(s): 
    return filter(None, re.sub(ur"[^A-zА-я]", u" ", s).lower().split())
        

if __name__ == "__main__":
    def read_data(pipeline, marks, zfile="news_data.zip"):
        train_data = list(data_iter(raw_text_iter(zfile, "news/news_train.txt", 1), pipeline, marks))
        train_Ys = read_classes(zfile, "news/news_train.txt", class_column=0)
        test_data = list(data_iter(raw_text_iter(zfile, "news/news_test.txt", 0), pipeline, marks))
        return dict(train_data=train_data, train_Ys=train_Ys, test_data=test_data)
        
    data = read_data([tok_proc], marks=("both",))
    
    # Подбираем параметры регрессии.
    for alpha in [0.1, 0.5, 1.0, 5, 10]:
        classif = make_pipeline(TfidfVectorizer(sublinear_tf=True, max_df=0.5), RidgeClassifier(alpha=alpha))
        print "alpha:", alpha, "score:", cross_val_score(classif, data["train_data"], data["train_Ys"], cv=5)
  
     
    
    # Классификатор с лучшими параметрами.
    classif = make_pipeline(TfidfVectorizer(sublinear_tf=True, max_df=0.5), RidgeClassifier(alpha=0.5))
    
    # Обучаем на полной обучайющей выборке.
    classif.fit(data["train_data"], data["train_Ys"])
    
    # Предсказываем тестовые значения и записываем предсказания.
    write_classes(classif.predict(data["test_data"]), "prediction.txt")
