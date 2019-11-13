import pandas as pd
from nltk.tokenize import word_tokenize
from hindi_tokenizer import HindiTokenizer as hindi
from hindi_tokenizer import wordsDict as word_dict
from pyiwn.pyiwn import pyiwn
import math
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tag import tnt
from nltk.corpus import indian
import nltk


class PreProcessing:

    def __init__(self, path):
        self.data = pd.read_excel(path, header=None)
        self.newinput_list1 = []
        self.newinput_list2 = []

    def tokenisation(self):
        # tokenizing the sentences
        self.tokenized0 = []
        self.tokenized1 = []
        self.paraphrased = []
        for index, i in self.data.iterrows():
            self.tokenized0.append(word_tokenize(i[0]))
            self.tokenized1.append(word_tokenize(i[1]))
            self.paraphrased.append(i[2])

        tokenizedDF = pd.DataFrame(
            {0: self.tokenized0,
             1: self.tokenized1,
             2: self.paraphrased
             })

        # removing the stopwords
        t = hindi.Tokenizer()
        stop = t.get_stop_words()

        tokenizedDF[0] = tokenizedDF[0].apply(lambda x: [item for item in x if item not in stop])
        tokenizedDF[1] = tokenizedDF[1].apply(lambda x: [item for item in x if item not in stop])

        # removing stem words

        stem = word_dict.stem_word
        tokenizedDF[0] = tokenizedDF[0].apply(lambda x: [item for item in x if item not in stem])

        # finding synonames and ngrams

        iwn = pyiwn.IndoWordNet('hindi')

        sims = []

        self.Synonyms1 = []
        self.Synonyms2 = []
        j = 0
        for index, i in tokenizedDF.iterrows():
            # terms1=tokenizedDF[0].iloc[index]
            terms1 = tokenizedDF[0].iloc[index]
            syn1 = []
            syn2 = []
            # print(terms1)
            for word1 in terms1:
                try:
                    syn1.append(iwn.synsets(word1)[0])
                except:  # if wordnet is not able to find a synset for word1
                    sims.append([0 for i in range(0, len(terms1))])
                    continue
            self.Synonyms1.append(syn1)
            # print(Synonyms1)
            terms2 = tokenizedDF[1].iloc[index]
            # print(terms1)
            for word2 in terms2:
                try:
                    syn2.append(iwn.synsets(word2)[0])
                except:  # if wordnet is not able to find a synset for word1
                    sims.append([0 for i in range(0, len(terms2))])
                    continue
            self.Synonyms2.append(syn2)

        newinput_list1 = []
        newinput_list2 = []
        from nltk import ngrams
        for index, i in tokenizedDF.iterrows():
            terms1 = tokenizedDF[0].iloc[index]
            terms2 = tokenizedDF[1].iloc[index]
            newinput_list1.append(list(zip(terms1, terms1[1:])))
            newinput_list2.append(list(zip(terms2, terms2[1:])))
        self.newinput_list1 = newinput_list1
        self.newinput_list2 = newinput_list2
        self.tokenizedDF = tokenizedDF

    def compute_cosine_similarity(self):
        self.cosine = []
        for index, i in self.tokenizedDF.iterrows():
            terms1 = []
            terms2 = []
            terms1 = self.tokenizedDF[0].iloc[index]
            terms2 = self.tokenizedDF[1].iloc[index]
            # text1 = ['जानकारी', 'मुताबिक','जानकारी', 'जंगलों', 'पन्द्रह्', 'फरवरी', 'फायर', 'सीजन', 'शुरू','जानकारी', 'जंगलों', 'पन्द्रह्']
            # text2 = ['आमतौर', 'जंगलों', "'फायर", 'सीजन', "'", 'पन्द्रह', 'फरवरी', 'शुरू']
            # text1 = 'जानकारी के मुताबिक जंगलों में पन्द्रह् फरवरी से फायर सीजन शुरू होता है। '
            # text2 = 'आमतौर पर यहां के जंगलों में  सीजन पन्द्रह  फरवरी से शुरू होता है'
            vector1 = text_to_vector(terms1)
            # print(vector1)
            vector2 = text_to_vector(terms2)
            # print(vector2)
            self.cosine.append(get_cosine(vector1, vector2))

    def compute_tfidf_similarity(self):
        self.sim = []
        for index, i in self.tokenizedDF.iterrows():
            text1 = self.data[0].iloc[index]
            text2 = self.data[1].iloc[index]
            #     text1 = 'जानकारी मुताबिक जंगलों में पन्द्रह् फरवरी से फ'
            #     text2 = 'आमतौर पर यहां के जंगलों में  सीजन पन्द्रह  फरवरी से शुरू होता है'
            # text1 = ['जानकारी', 'मुताबिक','जानकारी', 'जंगलों', 'पन्द्रह्', 'फरवरी', 'फायर', 'सीजन', 'शुरू','जानकारी', 'जंगलों', 'पन्द्रह्']
            # text2 = ['आमतौर', 'जंगलों', "'फायर", 'सीजन', "'", 'पन्द्रह', 'फरवरी', 'शुरू']
            corpus = [text1, text2]
            vectorizer = TfidfVectorizer(min_df=1)
            vec_1 = vectorizer.fit_transform(corpus).toarray()[0]
            vec_2 = vectorizer.fit_transform(corpus).toarray()[1]
            self.sim.append(np.dot(vec_1, vec_2.T) / (np.linalg.norm(vec_1) * np.linalg.norm(vec_2)))

    def pos_tagging(self):
        nltk.download('indian')
        train_data = indian.tagged_sents('hindi.pos')
        # print(train_data)
        self.tagged_words_1 = []
        self.tagged_words_2 = []

        tnt_pos_tagger = tnt.TnT()
        for index, i in self.tokenizedDF.iterrows():
            tnt_pos_tagger.train(train_data)
            text1 = self.tokenizedDF[0].iloc[index]
            text2 = self.tokenizedDF[1].iloc[index]
            # text=['जानकारी', 'मुताबिक', 'जंगलों', 'पन्द्रह्', 'फरवरी', 'फायर', 'सीजन', 'शुरू']
            self.tagged_words_1.append(tnt_pos_tagger.tag(text1))
            self.tagged_words_2.append(tnt_pos_tagger.tag(text2))

    def save_final_data(self):
        FinalDataFrame = pd.DataFrame(
            {'Column1': self.tokenized0,
             'Column2': self.tokenized1,
             'IsParaphrased': self.paraphrased,
             'TaggedWordsCol1': self.tagged_words_1,
             'TaggedWordsCol2': self.tagged_words_2,
             'TF-IDF Score': self.sim,
             'Cosine Similarity': self.cosine,
             'N-gramCol1': self.newinput_list1,
             'N-gramCol2': self.newinput_list2,
             'Synonyms_Col1': self.Synonyms1,
             'Synonyms_Col2': self.Synonyms2
             })

        from pandas import ExcelWriter
        from pandas import ExcelFile
        writer = ExcelWriter('FinalDataFrame.xlsx')
        FinalDataFrame.to_excel(writer, 'Sheet1', index=False)
        writer.save()
        pd.read_excel("FinaldataFrame.xlsx")
        return FinalDataFrame



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
    output=[]
    vect1={}
    for k in text:
        if k not in output:
            output.append(k)
    for i in output:
        count = 0
        for j in text:
            if i == j:
                count=count+1
        if i in vect1:
            vect1[i].append(count)
        else:
            vect1[i]= count
        #count=str(count)
        #vect1.append(i+':'+count)
        #print(Counter({vect1}))
    return vect1


def pre_process(path):
    p = PreProcessing(path)
    p.tokenisation()
    p.compute_cosine_similarity()
    p.compute_tfidf_similarity()
    p.pos_tagging()
    p.save_final_data()
