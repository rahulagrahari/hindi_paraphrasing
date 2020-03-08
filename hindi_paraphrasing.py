# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import warnings

warnings.filterwarnings("ignore")
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from hindi_tokenizer import HindiTokenizer as hindi
from hindi_tokenizer.wordsDict import words_dict


def generate_stem_words(word):
    suffixes = {
        1: ["ो", "े", "ू", "ु", "ी", "ि", "ा"],
        2: ["तृ", "ान", "ैत", "ने", "ाऊ", "ाव", "कर", "ाओ", "िए", "ाई", "ाए", "नी", "ना", "ते", "ीं", "ती",
            "ता", "ाँ", "ां", "ों", "ें", "ीय", "ति", "या", "पन", "पा", "ित", "ीन", "लु", "यत", "वट", "लू"],
        3: ["ेरा", "त्व", "नीय", "ौनी", "ौवल", "ौती", "ौता", "ापा", "वास", "हास", "काल", "पान", "न्त", "ौना", "सार",
            "पोश", "नाक",
            "ियल", "ैया", "ौटी", "ावा", "ाहट", "िया", "हार", "ाकर", "ाइए", "ाईं", "ाया", "ेगी", "वान", "बीन",
            "ेगा", "ोगी", "ोगे", "ाने", "ाना", "ाते", "ाती", "ाता", "तीं", "ाओं", "ाएं", "ुओं", "ुएं", "ुआं", "कला",
            "िमा", "कार",
            "गार", "दान", "खोर"],
        4: ["ावास", "कलाप", "हारा", "तव्य", "वैया", "वाला", "ाएगी", "ाएगा", "ाओगी", "ाओगे",
            "एंगी", "ेंगी", "एंगे", "ेंगे", "ूंगी", "ूंगा", "ातीं", "नाओं", "नाएं", "ताओं", "ताएं", "ियाँ", "ियों",
            "ियां",
            "त्वा", "तव्य", "कल्प", "िष्ठ", "जादा", "क्कड़"],
        5: ["ाएंगी", "ाएंगे", "ाऊंगी", "ाऊंगा", "ाइयाँ", "ाइयों", "ाइयां", "अक्कड़", "तव्य:", "निष्ठ"],
    }

    special_suffixes = ["र्", "ज्य", "त्य"]
    dict_special_suffixes = {"र्": "ृ",
                             "ज्य": "ज्",
                             "त्य": "त्"}
    ans = word
    bl = False

    if word in words_dict.keys():
        return words_dict[word]

    for L in 5, 4, 3, 2, 1:
        if len(word) > L + 1:
            for suf in suffixes[L]:
                if word.endswith(suf):
                    ans = word[:-L]
                    bl = True
        if bl == True:
            break

    if bl == True:
        for suf in suffixes[1]:
            if ans.endswith(suf):
                # use case - गानेवाला
                ans = generate_stem_words(ans)

    for suf in special_suffixes:
        if ans.endswith(suf):
            l = len(suf)
            ans = ans[:-l]
            ans += dict_special_suffixes[suf]

    return ans


# read the dataset
data = pd.read_excel("./TestHindi.xls", header=None)

# -------Data-Preprocessing-------- #
# tokenizing the sentences
tokenized0 = []
tokenized1 = []
paraphrased = []
for index, i in data.iterrows():
    tokenized0.append(word_tokenize(i[0]))
    tokenized1.append(word_tokenize(i[1]))
    paraphrased.append(i[2])

tokenizedDF = pd.DataFrame(
    {0: tokenized0,
     1: tokenized1,
     2: paraphrased
     })

# removing the stopwords
t = hindi.Tokenizer()
stop = t.get_stop_words()

tokenizedDF[0] = tokenizedDF[0].apply(lambda x: [item for item in x if item not in stop])
tokenizedDF[1] = tokenizedDF[1].apply(lambda x: [item for item in x if item not in stop])

# removing stem words

stem = ['कों', 'ौ', 'ै', 'ा', 'ी', 'ू', 'ो', 'े', '्', 'ि', 'ु', 'ं', 'ॅ', 'कर', 'ाओ', 'िए', 'ाई', 'ाए', 'ने', 'नी',
        'ना', 'ते', 'ीं', 'ती', 'ता', 'ाॅ', 'ां', 'ों', 'ें', 'ाकर', 'ाइए', 'ाईं', 'ाया', 'ेगी', 'ेगा', 'ोगी', 'ोगे',
        'ाने', 'ाना', 'ाते', 'ाती', 'ाता', 'तीं', 'ाओं', 'ाएं', 'ुऔं', 'ुएं', 'ुआं']
tokenizedDF[0] = tokenizedDF[0].apply(lambda x: [generate_stem_words(word) for word in x])
tokenizedDF[1] = tokenizedDF[1].apply(lambda x: [generate_stem_words(word) for word in x])

# finding synonames and ngrams

from nltk.corpus import wordnet as wn
from pyiwn.pyiwn import pyiwn

iwn = pyiwn.IndoWordNet('hindi')

sims = []

Synonyms1 = []
Synonyms2 = []
j = 0
for index, i in tokenizedDF.iterrows():
    # terms1=tokenizedDF[0].iloc[index]
    # terms1 = tokenizedDF[0].iloc[index]
    terms1 = i[0]
    terms2 = i[1]
    syn1 = []
    syn2 = []
    # print(terms1)
    for word in zip(terms1, terms2):
        try:
            synonyms1 = iwn.synsets(word[0])[0]
            synonyms2 = iwn.synsets(word[1])[0]
        except IndexError:
            continue
        syn1.append(synonyms1)
        syn2.append(synonyms2)
    # except:  # if wordnet is not able to find a synset for word1
        #     sims.append([0 for i in range(0, len(terms1))])
        #     continue
    Synonyms1.append(syn1)
    # # print(Synonyms1)
    # terms2 = tokenizedDF[1].iloc[index]
    # print(terms1)
    # for word2 in terms2:
    #     try:
    #         syn2.append(iwn.synsets(word2)[0])
    #     except:  # if wordnet is not able to find a synset for word1
    #         sims.append([0 for i in range(0, len(terms2))])
    #         continue
    Synonyms2.append(syn2)

newinput_list1 = []
newinput_list2 = []
from nltk import ngrams

for index, i in tokenizedDF.iterrows():
    terms1 = tokenizedDF[0].iloc[index]
    terms2 = tokenizedDF[1].iloc[index]
    newinput_list1.append(list(zip(terms1, terms1[1:])))
    newinput_list2.append(list(zip(terms2, terms2[1:])))
newinput_list1
newinput_list2

# getting the cosine and tf-idf score

import re, math
from collections import Counter

WORD = re.compile(r'\w+')


def get_cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in vec1.keys()])
    sum2 = sum([vec2[x] ** 2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


def text_to_vector(text):
    output = []
    vect1 = {}
    for k in text:
        if k not in output:
            output.append(k)
    for i in output:
        count = 0
        for j in text:
            if i == j:
                count = count + 1
        if i in vect1:
            vect1[i].append(count)
        else:
            vect1[i] = count
        # count=str(count)
        # vect1.append(i+':'+count)
        # print(Counter({vect1}))
    return vect1


#      words = WORD.findall(text)
#      print(words)
#      return Counter(words)

cosine = []
for index, i in tokenizedDF.iterrows():
    terms1 = []
    terms2 = []
    terms1 = tokenizedDF[0].iloc[index]
    terms2 = tokenizedDF[1].iloc[index]
    # text1 = ['जानकारी', 'मुताबिक','जानकारी', 'जंगलों', 'पन्द्रह्', 'फरवरी', 'फायर', 'सीजन', 'शुरू','जानकारी', 'जंगलों', 'पन्द्रह्']
    # text2 = ['आमतौर', 'जंगलों', "'फायर", 'सीजन', "'", 'पन्द्रह', 'फरवरी', 'शुरू']
    # text1 = 'जानकारी के मुताबिक जंगलों में पन्द्रह् फरवरी से फायर सीजन शुरू होता है। '
    # text2 = 'आमतौर पर यहां के जंगलों में  सीजन पन्द्रह  फरवरी से शुरू होता है'
    vector1 = text_to_vector(terms1)
    # print(vector1)
    vector2 = text_to_vector(terms2)
    # print(vector2)
    cosine.append(get_cosine(vector1, vector2))

print('Cosine:', cosine)

import re, math
from collections import Counter

WORD = re.compile(r'\w+')


def text_to_vector(text):
    words = WORD.findall(text)
    return Counter(words)


import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

sim = []
for index, i in tokenizedDF.iterrows():
    text1 = data[0].iloc[index]
    text2 = data[1].iloc[index]
    #     text1 = 'जानकारी मुताबिक जंगलों में पन्द्रह् फरवरी से फ'
    #     text2 = 'आमतौर पर यहां के जंगलों में  सीजन पन्द्रह  फरवरी से शुरू होता है'
    # text1 = ['जानकारी', 'मुताबिक','जानकारी', 'जंगलों', 'पन्द्रह्', 'फरवरी', 'फायर', 'सीजन', 'शुरू','जानकारी', 'जंगलों', 'पन्द्रह्']
    # text2 = ['आमतौर', 'जंगलों', "'फायर", 'सीजन', "'", 'पन्द्रह', 'फरवरी', 'शुरू']
    corpus = [text1, text2]
    vectorizer = TfidfVectorizer(min_df=1)
    vec_1 = vectorizer.fit_transform(corpus).toarray()[0]
    vec_2 = vectorizer.fit_transform(corpus).toarray()[1]
    sim.append(np.dot(vec_1, vec_2.T) / (np.linalg.norm(vec_1) * np.linalg.norm(vec_2)))
print(sim)

# pos tagging

from nltk.tag import tnt
from nltk.corpus import indian

nltk.download('indian')
train_data = indian.tagged_sents('hindi.pos')
# print(train_data)
tagged_words_1 = []
tagged_words_2 = []

tnt_pos_tagger = tnt.TnT()
for index, i in tokenizedDF.iterrows():
    tnt_pos_tagger.train(train_data)
    text1 = tokenizedDF[0].iloc[index]
    text2 = tokenizedDF[1].iloc[index]
    # text=['जानकारी', 'मुताबिक', 'जंगलों', 'पन्द्रह्', 'फरवरी', 'फायर', 'सीजन', 'शुरू']
    tagged_words_1.append(tnt_pos_tagger.tag(text1))
    tagged_words_2.append(tnt_pos_tagger.tag(text2))
print(tagged_words_1)

FinalDataFrame = pd.DataFrame(
    {'Column1': tokenized0,
     'Column2': tokenized1,
     'IsParaphrased': paraphrased,
     'TaggedWordsCol1': tagged_words_1,
     'TaggedWordsCol2': tagged_words_2,
     'TF-IDF Score': sim,
     'Cosine Similarity': cosine,
     'N-gramCol1': newinput_list1,
     'N-gramCol2': newinput_list2,
     'Synonyms_Col1': Synonyms1,
     'Synonyms_Col2': Synonyms2
     })

from pandas import ExcelWriter
from pandas import ExcelFile

writer = ExcelWriter('FinalDataFrame.xlsx')
FinalDataFrame.to_excel(writer, 'Sheet1', index=False)
writer.save()

FinalDataFrame = pd.read_excel("FinaldataFrame.xlsx")
