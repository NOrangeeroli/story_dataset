import json
import re
import os
import jieba
import jieba.posseg as pseg
import nltk
import random
import numpy as np
from nltk.tokenize import WordPunctTokenizer
import operator
from collections import Counter

#--------------------------------------------------------------------rakeen
def is_number(s):
    try:
        float(s) if '.' in s else int(s)
        return True
    except ValueError:
        return False

def load_stop_words(stop_word_file):
    """
    Utility function to load stop words from a file and return as a list of words
    @param stop_word_file Path and file name of a file containing stop words.
    @return list A list of stop words.
    """
    stop_words = []
    for line in open(stop_word_file):
        if line.strip()[0:1] != "#":
            for word in line.split():  # in case more than one per line
                stop_words.append(word)
    return stop_words

def separate_words(text, min_word_return_size):
    """
    Utility function to return a list of all words that are have a length greater than a specified number of characters.
    @param text The text that must be split in to words.
    @param min_word_return_size The minimum no of characters a word must have to be included.
    """
    splitter = re.compile('[^a-zA-Z0-9_\\+\\-/]')
    words = []
    for single_word in splitter.split(text):
        current_word = single_word.strip().lower()
        #leave numbers in phrase, but don't count as words, since they tend to invalidate scores of their phrases
        if len(current_word) > min_word_return_size and current_word != '' and not is_number(current_word):
            words.append(current_word)
    return words

def split_sentences(text):
    """
    Utility function to return a list of sentences.
    @param text The text that must be split in to sentences.
    """
    sentence_delimiters = re.compile(u'[.!?,;:\t\\\\"\\(\\)\\\'\u2019\u2013]|\\s\\-\\s')
    sentences = sentence_delimiters.split(text)
    return sentences

def build_stop_word_regex(stop_word_file_path):
    stop_word_list = load_stop_words(stop_word_file_path)
    stop_word_regex_list = []
    for word in stop_word_list:
        word_regex = r'\b' + word + r'(?![\w-])'  # added look ahead for hyphen
        stop_word_regex_list.append(word_regex)
    stop_word_pattern = re.compile('|'.join(stop_word_regex_list), re.IGNORECASE)
    return stop_word_pattern

def generate_candidate_keywords(sentence_list, stopword_pattern):
    phrase_list = []
    for s in sentence_list:
        tmp = re.sub(stopword_pattern, '|', s.strip())
        phrases = tmp.split("|")
        for phrase in phrases:
            phrase = phrase.strip().lower()
            if phrase != "":
                phrase_list.append(phrase)
    return phrase_list

def calculate_word_scores(phraseList):
    word_frequency = {}
    word_degree = {}
    for phrase in phraseList:
        word_list = separate_words(phrase, 0)
        word_list_length = len(word_list)
        word_list_degree = word_list_length - 1
        #if word_list_degree > 3: word_list_degree = 3 #exp.
        for word in word_list:
            word_frequency.setdefault(word, 0)
            word_frequency[word] += 1
            word_degree.setdefault(word, 0)
            word_degree[word] += word_list_degree  #orig.
            #word_degree[word] += 1/(word_list_length*1.0) #exp.
    for item in word_frequency:
        word_degree[item] = word_degree[item] + word_frequency[item]

    # Calculate Word scores = deg(w)/frew(w)
    word_score = {}
    for item in word_frequency:
        word_score.setdefault(item, 0)
        word_score[item] = word_degree[item] / (word_frequency[item] * 1.0)  #orig.
    #word_score[item] = word_frequency[item]/(word_degree[item] * 1.0) #exp.
    return word_score

def generate_candidate_keyword_scores(phrase_list, word_score):
    keyword_candidates = {}
    for phrase in phrase_list:
        keyword_candidates.setdefault(phrase, 0)
        word_list = separate_words(phrase, 0)
        candidate_score = 0
        for word in word_list:
            candidate_score += word_score[word]
        if len(word_list)>8:
            continue
        keyword_candidates[phrase] = candidate_score
    return keyword_candidates

class Rake(object):
    def __init__(self, stop_words_path):
        self.stop_words_path = stop_words_path
        self.__stop_words_pattern = build_stop_word_regex(stop_words_path)

    def run(self, text):
        sentence_list = split_sentences(text)

        phrase_list = generate_candidate_keywords(sentence_list, self.__stop_words_pattern)

        word_scores = calculate_word_scores(phrase_list)

        keyword_candidates = generate_candidate_keyword_scores(phrase_list, word_scores)

        sorted_keywords = sorted(keyword_candidates.items(), key=operator.itemgetter(1), reverse=True)
        return sorted_keywords

def rakeen(text):
    # Split text into sentences
    sentenceList = split_sentences(text)
    #stoppath = "FoxStoplist.txt" #Fox stoplist contains "numbers", so it will not find "natural numbers" like in Table 1.1
    stoppath = "SmartStoplist.txt"  #SMART stoplist misses some of the lower-scoring keywords in Figure 1.5, which means that the top 1/3 cuts off one of the 4.0 score words in Table 1.1
    stopwordpattern = build_stop_word_regex(stoppath)

    # generate candidate keywords
    phraseList = generate_candidate_keywords(sentenceList, stopwordpattern)

    # calculate individual word scores
    wordscores = calculate_word_scores(phraseList)

    rake = Rake("SmartStoplist.txt")
    keywords = rake.run(text)
    return keywords
#-----------------------------------------------------------------rakezh
# Data structure for holding data
class Word():
    def __init__(self, char, freq = 0, deg = 0):
        self.freq = freq
        self.deg = deg
        self.char = char
 
    def returnScore(self):
        return self.deg/self.freq
 
    def updateOccur(self, phraseLength):
        self.freq += 1
        self.deg += phraseLength
 
    def getChar(self):
        return self.char
 
    def updateFreq(self):
        self.freq += 1
 
    def getFreq(self):
        return self.freq
    
    def getDeg(self):
        return self.deg
# Check if contains num
def notNumStr(instr):
    for item in instr:
        if '\u0041' <= item <= '\u005a' or ('\u0061' <= item <='\u007a') or item.isdigit():
            return False
    return True
 
# Read Target Case if Json
def readSingleTestCases(testFile):
    with open(testFile) as json_data:
        try:
            testData = json.load(json_data)
        except:
            # This try block deals with incorrect json format that has ' instead of "
            data = json_data.read().replace("'",'"')
            try:
                testData = json.loads(data)
                # This try block deals with empty transcript file
            except:
                return ""
    returnString = ""
    for item in testData:
        try:
            returnString += item['text']
        except:
            returnString += item['statement']
    return returnString
 
def rakezh(rawText):
    # Construct Stopword Lib
    import os
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    swLibList = [line.rstrip('\n') for line in open(curr_dir+'/'+r"sp.txt",'r',encoding='utf-8')]
    # Construct Phrase Deliminator Lib
    conjLibList = [line.rstrip('\n') for line in open(curr_dir+'/'+r"spw.txt",'r',encoding='utf-8')]
    # Cut Text
    rawtextList = pseg.cut(rawText)
    
    # Construct List of Phrases and Preliminary textList
    textList = []
    listofSingleWord = dict()
    lastWord = ''
    #poSPrty = ['m','x','uj','ul','mq','u','v','f']
    poSPrty = ['m','x','uj','ul','mq','u','f']
    meaningfulCount = 0
    checklist = []
    for eachWord, flag in rawtextList:
        checklist.append([eachWord,flag])
        if eachWord in conjLibList or not notNumStr(eachWord) or eachWord in swLibList or flag in poSPrty or eachWord == '\n':
            if lastWord != '|':
                textList.append("|")
                lastWord = "|"
        elif eachWord not in swLibList and eachWord != '\n':
            textList.append(eachWord)
            meaningfulCount += 1
            if eachWord not in listofSingleWord:
                listofSingleWord[eachWord] = Word(eachWord)
            lastWord = ''
    
    # Construct List of list that has phrases as wrds
    newList = []
    tempList = []
    for everyWord in textList:
        if everyWord != '|':
            tempList.append(everyWord)
        else:
            newList.append(tempList)
            tempList = []
    r'''
    print('newlist',newList)
    print(rawText)
    gg=[]
    for i,j in rawtextList:
        gg.append(i)
    print(gg)
    '''
    tempStr = ''
    for everyWord in textList:
        if everyWord != '|':
            tempStr += everyWord + '|'
        else:
            if tempStr[:-1] not in listofSingleWord:
                listofSingleWord[tempStr[:-1]] = Word(tempStr[:-1])
                tempStr = ''
 
    # Update the entire List
    for everyPhrase in newList:
        res = ''
        for everyWord in everyPhrase:
            listofSingleWord[everyWord].updateOccur(len(everyPhrase))
            res += everyWord + '|'
        phraseKey = res[:-1]
        if phraseKey not in listofSingleWord:
            listofSingleWord[phraseKey] = Word(phraseKey)
        elif len(everyPhrase)>1:
            listofSingleWord[phraseKey].updateFreq()
    # Get score for entire Set
    outputList = dict()
    #print('newList',newList)
    for everyPhrase in newList:
        if len(everyPhrase) > 8:
            continue
        score = 0
        phraseString = ''
        outStr = ''
        for everyWord in everyPhrase:
            score += listofSingleWord[everyWord].returnScore()
            phraseString += everyWord + '|'
            outStr += everyWord
        phraseKey = phraseString[:-1]
        freq = listofSingleWord[phraseKey].getFreq()
        if meaningfulCount==0:
            continue
        #if freq / meaningfulCount < 0.01 and freq < 2 :
        #if freq < 2 :
        #    continue
        outputList[outStr] = score
 
    sorted_list = sorted(outputList.items(), key = operator.itemgetter(1), reverse = True)
    #return sorted_list[:10]
    return sorted_list

punczh=['。','，','！','？','…','：']
puncen=['.',',','!','?',':']

def ispunczh(word):
    for i in punczh:
        if word==i:
            return True
    return False

def ispuncen(word):
    for i in puncen:
        if word==i:
            return True
    return False

def getfirstzh(text):
    r'''k=len(text)-1
    for i in range(len(text)):
        if i<10:
            continue
        if ispunczh(text[i]):
            k=i
            break
    return text[:k+1]
    '''
    text=jieba.lcut(text)
    temp=''
    sto=''
    for i in range(len(text)):
        temp+=text[i]
        if i<6:
            continue
        if ispunczh(text[i]):
            while (i+1<len(text) and (not iszh(text[i+1]))):
                i+=1
            for j in text[i+1:]:
                sto+=j
            break
    return temp,sto,i+1,len(text)-i-1

def getfirsten(text):
    text=jieba.lcut(text)
    temp=''
    sto=''
    for i in range(len(text)):
        temp+=text[i]
        if i<9:
            continue
        if ispuncen(text[i]):
            while (i+1<len(text) and (not isen(text[i+1]))):
                i+=1
            for j in text[i+1:]:
                sto+=j
            break
    return temp,sto,i+1,len(text)-i-1

def isen(word):
    return ('a'<=word<='z') or ('A'<=word<='Z')

def iszh(word):
    return ('\u4e00'<= word <= '\u9fa5')

def strB2Q(ustring):
    """半角转全角"""
    rstring = ""
    for uchar in ustring.replace("...", "…"):
        inside_code=ord(uchar)
        if uchar in punctuation:
            if inside_code == 32:
                inside_code = 12288
            elif inside_code >= 32 and inside_code <= 126:
                inside_code += 65248
        rstring += chr(inside_code)
    return rstring

def repetition_distinct(name, cands):
    result = {}
    tgt_dir = "./lex_rept/%s"%(name.split("/")[0])
    if ("/" in name) and (not os.path.exists(tgt_dir)):
        os.mkdir(tgt_dir)
    fout = open("./lex_rept/%s.txt"%name, "w")
    for i in range(1, 5):
        num, all_ngram, all_ngram_num = 0, {}, 0.
        for k, cand in enumerate(cands):
            ngs = ["_".join(c) for c in ngrams(cand, i)]
            all_ngram_num += len(ngs)
            for s in ngs:
                if s in all_ngram:
                    all_ngram[s] += 1
                else:
                    all_ngram[s] = 1
            for s in set(ngs):
                if ngs.count(s) > 1:
                    if i == 4: fout.write("%d|||%s|||%s\n"%(k, s, " ".join(cand)))
                    num += 1
                    break
        result["repetition-%d"%i] = "%.4f"%(num / float(len(cands)))
        result["distinct-%d"%i] = "%.4f"%(len(all_ngram) / float(all_ngram_num))
    fout.close()
    return result

def judgezh(k):
    if k=="？" or k=="！" or k=="“" or k=="”" or k=="：" or k=="；" or k=="。":
        return 0
    return (k>='\u4e00' and k<='\u9fff')

def judgeen(k):
    if k=="?" or k=="!" or k=="\"" or k==":" or k==";" or k==".":
        return 0
    return ('a'<=k<='z') or ('A'<=k<='Z')

def addvocab(vocab,text):
    for i in text:
        vocab[i]=1
    return vocab

def swap(t1,t2):
    return t2,t1

def countlabelzh(data,suf='',debug=0):
    table={
        'cnt':0,
        'ppso':0,
        'pl':0,
        'fsl':0
    }
    table['cnt']=len(data)
    cnt=0
    for i in range(len(data)):
        story=data[i]
        inputs=story[0]
        targets=story[1]
        cnt+=1
        if cnt%100==0:
            print(cnt,len(data))
        if suf!='all':
            ff,sto,lff,lsto=getfirstzh(inputs)
            data[i][0]=sto
            table['fsl']+=lff
            if len(sto)==0:
                sto=ff
            temp=rakezh(sto)
            keywords=[]
            tot=0
            if suf=='test':
                print(cnt,temp)
            for j in temp:
                j=list(j)
                if j[1]>1:
                    flag=0
                    for k in keywords:
                        if re.search(j[0],k,re.M) is not None:
                            flag=1
                            break
                    if flag==1:
                        continue
                    for k in range(len(keywords)-1,-1,-1):
                        if re.search(keywords[k],j[0],re.M) is not None:
                            keywords=keywords[:k]+keywords[k+1:]
                    keywords.append(j[0])
                    tot+=1
                    if tot>7:
                        break
            if tot<2:
                print('tot',i,tot)
                if tot==1:
                    temp=temp[1:]
                random.shuffle(temp)
                while tot<2 and len(temp)>0:
                    if len(temp[0][0])>0:
                        keywords.append(temp[0][0])
                        tot+=1
                    temp=temp[1:]
                print('end',tot)
            temp=0
            for j in keywords:
                temp+=len(jieba.lcut(j))
            l=lff+lsto
            k=temp/l
            if k>0.25:
                while k>0.25:
                    keywords=keywords[:-1]
                    temp=0
                    for j in keywords:
                        temp+=len(jieba.lcut(j))
                    k=temp/l
            table['ppso']+=len(keywords)
            table['pl']+=temp
            random.shuffle(keywords)
            data[i].append(keywords)
            temp=targets+'#'+ff
            for j in keywords:
                temp+='#'+j
            data[i].append(temp)
            #print(keywords)
    table['ppso']/=table['cnt']
    table['pl']/=table['cnt']
    table['fsl']/=table['cnt']
    if suf!='all' and not debug:
        with open('/mnt/c/onlg/storal/storal_zh_label_%s2.json'%(suf),'w',encoding='utf-8') as fout:
            json.dump(data,fout,indent=4,ensure_ascii=False)
    return table

def countlabelen(data,suf='',debug=0):
    table={
        'cnt':0,
        'ppso':0,
        'pl':0,
        'fsl':0
    }
    table['cnt']=len(data)
    cnt=0
    for i in range(len(data)):
        story=data[i]
        inputs=story[0]
        targets=story[1]
        cnt+=1
        if cnt%100==0:
            print(cnt,len(data))
        if suf!='all':
            ff,sto,lff,lsto=getfirsten(inputs)
            data[i][0]=sto
            #print(ff,lff)
            if len(sto)<3:
                sto=ff
            temp=rakeen(sto)
            if suf=='test':
                print(cnt,temp)
            keywords=[]
            tot=0
            for j in temp:
                j=list(j)
                if j[1]>1:
                    flag=0
                    for k in keywords:
                        if re.search(j[0],k,re.M) is not None:
                            flag=1
                            break
                    if flag==1:
                        continue
                    for k in range(len(keywords)-1,-1,-1):
                        if re.search(keywords[k],j[0],re.M) is not None:
                            keywords=keywords[:k]+keywords[k+1:]
                    keywords.append(j[0])
                    tot+=1
                    if tot>7:
                        break
            if tot<5:
                print('tot',i,tot)
                for j in range(1,6):
                    if tot==j:
                        temp=temp[j:]
                        break
                random.shuffle(temp)
                while tot<5 and len(temp)>0:
                    if len(temp[0][0])>0:
                        keywords.append(temp[0][0])
                        tot+=1
                    temp=temp[1:]
                print('end',tot)
            temp=0
            for j in keywords:
                temp+=len(WordPunctTokenizer().tokenize(j))
            l=len(WordPunctTokenizer().tokenize(inputs))
            k=temp/l
            if k>0.25:
                while k>0.25:
                    keywords=keywords[:-1]
                    temp=0
                    for j in keywords:
                        temp+=len(WordPunctTokenizer().tokenize(j))
                    k=temp/l
            table['ppso']+=len(keywords)
            temp=0
            for j in keywords:
                temp+=len(WordPunctTokenizer().tokenize(j))
            table['pl']+=temp
            table['fsl']+=len(WordPunctTokenizer().tokenize(ff))
            random.shuffle(keywords)
            data[i].append(keywords)
            temp=targets+' # '+ff
            for j in keywords:
                temp+=' # '+j
            data[i].append(temp)
    table['ppso']/=table['cnt']
    table['pl']/=table['cnt']
    table['fsl']/=table['cnt']
    if suf!='all' and not debug:
        with open('/mnt/c/onlg/storal/storal_en_label_%s2.json'%(suf),'w',encoding='utf-8') as fout:
            json.dump(data,fout,indent=4,ensure_ascii=False)
    return table

def countzh(debug):
    detail={}
    label={'all':0,'test':0,'valid':0,'train':0}
    with open('/mnt/c/onlg/storal/storal_zh_label.json','r',encoding='utf-8') as fout:
        label['all']=json.load(fout)
    with open('/mnt/c/onlg/storal/storal_zh_label_test.json','r',encoding='utf-8') as fout:
        label['test']=json.load(fout)
    with open('/mnt/c/onlg/storal/storal_zh_label_valid.json','r',encoding='utf-8') as fout:
        label['valid']=json.load(fout)
    with open('/mnt/c/onlg/storal/storal_zh_label_train.json','r',encoding='utf-8') as fout:
        label['train']=json.load(fout)
    with open('/mnt/c/onlg/storal/storal_zh_unlabel.json','r',encoding='utf-8') as fout:
        unlabel=json.load(fout)
    for q in ['all','test','train','valid']:
        print('zh',q)
        detail[q]=countlabelzh(label[q],q,debug)
    if not debug:
        with open('/mnt/c/onlg/storal/storal_zh_label_keywords.json','w',encoding='utf-8') as fout:
            json.dump(detail,fout,indent=4,ensure_ascii=False)
def counten(debug):
    detail={}
    r'''label={'all':0,'test':0,'valid':0,'train':0}
    with open('/mnt/c/onlg/storal/storal_en_label.json','r',encoding='utf-8') as fout:
        label['all']=json.load(fout)
    with open('/mnt/c/onlg/storal/storal_en_label_test.json','r',encoding='utf-8') as fout:
        label['test']=json.load(fout)
    with open('/mnt/c/onlg/storal/storal_en_label_valid.json','r',encoding='utf-8') as fout:
        label['valid']=json.load(fout)
    with open('/mnt/c/onlg/storal/storal_en_label_train.json','r',encoding='utf-8') as fout:
        label['train']=json.load(fout)
    with open('/mnt/c/onlg/storal/storal_en_unlabel2.json','r',encoding='utf-8') as fout:
        unlabel=json.load(fout)
    for q in ['all','test','train','valid']:'''
    label={'gg':0}
    with open('/mnt/c/onlg/storal/storal_en_label_gg.json','r',encoding='utf-8') as fout:
        label['gg']=json.load(fout)
    for q in ['gg']:
        print('en',q)
        detail[q]=countlabelen(label[q],q,debug)
    r'''
    if not debug:
        with open('/mnt/c/onlg/storal/storal_en_label_keywords.json','w',encoding='utf-8') as fout:
            json.dump(detail,fout,indent=4,ensure_ascii=False)'''
if __name__=='__main__':
    debug=0
    print('debug=%d'%(debug))
    print(rakezh('有一回，鹿和马为了一块草地争吵得不可开交，各人都想将这块草地占为己有。最后鹿仗着自己那对厉害的角，终于战胜了马。这对马来说，简直是无法容忍的。怎样才能重新把鹿赶走呢？马考虑来考虑去；终于想到去求助于人。它找到了一个很强壮的男人。这个人来到草地，同鹿干了一仗，将鹿杀死了。从此，这块引起纠纷的草地，完全归马独自占有了。不过，那位帮助马取得胜利的男人，也已将马占为己有了。马得到了好处，不再贫困，却也失去了自由。'))

    #countzh(debug)
    #counten(debug)
