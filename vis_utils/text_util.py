#import string
#import pyltp
#import re
from zhon.hanzi import punctuation

from pyltp import SentenceSplitter
import numpy as np
from keras.utils import to_categorical
from keras.preprocessing import sequence
import jieba
import _pickle as pickle
import pdb

word_dict = pickle.load(open('./cache/w2v.dict.pkl', 'rb'))
idx_dict = pickle.load(open('./cache/idx_dict.pkl', 'rb'))
np_embedding_weights = np.load('./cache/word_embed.npy')

word_unknown = len(word_dict.keys()) + 1

#from spacy.lang.en import English
#
#STOP_WORDS = ['the', 'a', 'an']
#nlp = English()
#nlp.add_pipe(nlp.create_pipe('sentencizer'))
#
#def normalize(text):
#	text = text.lower().strip()
#	doc = nlp(text)
#	filtered_sentences = []
#	for sentence in doc.sents:
#		filtered_tokens = list()
#		for i, w in enumerate(sentence):
#			s = w.string.strip()
#			if len(s) == 0 or s in string.punctuation and i < len(doc) - 1:
#				continue
#			if s not in STOP_WORDS:
#				s = s.replace(',', '.')
#				filtered_tokens.append(s)
#		filtered_sentences.append(' '.join(filtered_tokens))
#	return filtered_sentences

sen_seq_length = 10
word_seq_length = 25

def senSplit(text):
    text = text.strip().split()
    sentences = []
    for t in text:
        #t = re.sub(u"([%s])+"%punctuation, r"\1", t)
        sentences.extend(list(SentenceSplitter.split(t)))
    sentences = [s for s in sentences if len(s) > 1]
    return sentences

def splitAll(text):
    text = text.strip().split()
    sentences = []
    for t in text:
        #t = re.sub(u"([%s])+"%punctuation, r"\1", t)
        sentences.extend([" ".join(jieba.cut(sen)) for sen in list(SentenceSplitter.split(t))])
    sentences = [s for s in sentences if len(s) > 1]
    return sentences

def idx2Word(encoded_text, idx_dict):
    s = [idx_dict[idx] for idx in encoded_text]
    return "".join(s)

def han_batch_generator(contents, labels, batch_size=128, keep=False, shuffle=True):
    sample_size = contents.shape[0]
    idx_array = np.arange(sample_size)
    while 1:
        if shuffle:
            np.random.shuffle(idx_array)
        batches = make_batches(sample_size, batch_size)
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            batch_ids = idx_array[batch_start:batch_end]
            batch_contents = contents[batch_ids]
            batch_contents = han_pro(batch_contents, keep=keep)
            batch_labels = to_categorical(labels[batch_ids])
            yield (batch_contents, batch_labels)
            

def encode_input(x):
    x = np.array(x)
    if not x.shape:
        x = np.expand_dims(x, 0)
    texts = x
    return han_pro(texts)

def han_pro(contents, sentence_num=sen_seq_length, word_num=word_seq_length, keep=False):
    contents_seq = np.zeros(shape=(len(contents), sentence_num, word_num))
    for index, content in enumerate(contents):
        sentences = senSplit(content)
        word_seq = get_sequence(sentences, maxlen=word_num)
        word_seq = word_seq[:sentence_num]
        contents_seq[index][:len(word_seq)] = word_seq
    return contents_seq
                         
def make_batches(size, batch_size):
    nb_batch = int(np.ceil(size/float(batch_size)))
    return [(i*batch_size, min(size, (i+1)*batch_size)) for i in range(0, nb_batch)]


def get_sequence(contents, maxlen=sen_seq_length, mode='post', keep=False):
    word_r = []
    contents = "\n".join(contents)
    contents = " ".join(list(jieba.cut(contents))).replace(" \n ", "\n")
    contents = [content.split(" ") for content in contents.split('\n')]
    for content in contents:
        if keep:
            word_c = np.array([word_dict[w] if w in word_dict else word_unknown for w in content])
        else:
            word_c = np.array([word_dict[w] for w in content if w in word_dict])
        word_r.append(word_c)
    word_seq = sequence.pad_sequences(word_r, maxlen=maxlen, padding=mode, truncating=mode, value=0)
    return word_seq
