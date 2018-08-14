# -*- coding: utf-8 -*-


from keras.preprocessing import sequence
from keras.utils import to_categorical
import jieba
import logging
import _pickle as pickle
from tqdm import tqdm
import numpy as np
from pyltp import SentenceSplitter

from sklearn.metrics import precision_score,recall_score,f1_score

sentence_length = 150
sen_seq_length = 10
word_seq_length = 20

batch_vol = 25600


jieba.setLogLevel(logging.INFO)
jieba.enable_parallel(4)

word_embed_dict = pickle.load(open('./cache/w2v.dict.pkl', 'rb'))
word_unknown = len(word_embed_dict.keys()) + 1

def score(pred, label, gate=0.5):
    if len(label.shape) == 1:
        p = (pred>gate).astype("int")
        p = np.squeeze(p)
        l = label
    else:
        p = np.argmax(pred, axis=1)
        l = np.argmax(label, axis=1)
    pre_score = precision_score(l, p)
    rec_score = recall_score(l, p)
    f_score = f1_score(l, p)
    return pre_score, rec_score, f_score



            
#def seqPreComputeAndSave(contents, labels, maxlen=sentence_length, mode='post', keep=False, isTrain=True):
#    word_r = []
#    if isTrain:
#        folder = 'train/'
#    else:
#        folder = 'test/'
#    contents = "\n".join(contents)
#    contents = " ".join(list(jieba.cut(contents))).replace(" \n ", "\n")
#    contents = [content.split(" ") for content in contents.split('\n')]
#    counter = 0
#    for idx, content in tqdm(enumerate(contents)):
#        if keep:
#            word_c = np.array([word_embed_dict[w] if w in word_embed_dict else word_unknown for w in content])
#        else:
#            word_c = np.array([word_embed_dict[w] for w in content if w in word_embed_dict])
#        word_r.append(word_c)
#        if (idx+1) % batch_vol == 0:
#            label_seq = labels[batch_vol*counter:batch_vol*(counter+1)]
#            word_seq = sequence.pad_sequences(word_r, maxlen=maxlen, padding=mode, truncating=mode, value=0)
#            np.save(seq_cache+folder+'X_%d.npy'%counter, word_seq)
#            np.save(seq_cache+folder+'y_%d.npy'%counter, label_seq)
#            word_r.clear()
#            counter+=1
#    word_seq = sequence.pad_sequences(word_r, maxlen=maxlen, padding=mode, truncating=mode, value=0)
#    label_seq = labels[counter*batch_vol:]
#    np.save(seq_cache+folder+'X_%d.npy'%counter, word_seq)
#    np.save(seq_cache+folder+'y_%d.npy'%counter, label_seq)
                    
    # ./cache/precompute_seq/train_seq or test_seq /seq_X_%d

    
#def hanPreComputeAndSave(contents, labels, sen_seq_length, word_seq_length, mode='post', keep=False, isTrain=True):
#    contents_seq = np.zero(shape=(batch_vol, sen_seq_length, word_seq_length))
#    if isTrain:
#        folder='train/'
#    else:
#        folder='test/'
#    counter = 0
#    for index, content in enumerate(contents):
#        sentences = SentenceSplitter.split(content)
#        Cs = "\n".join(sentences); Cs = " ".join(list(jieba.cut(Cs))).replace(" \n ", "\n"); Cs = [a.split(" ") for a in Cs.split('\n')]
#        w_r = []
#        for c in Cs:
#            if keep:
#                w_c = np.array([word_embed_dict[w] if w in word_embed_dict else word_unknown for w in c])
#            else:
#                w_c = np.array([word_embed_dict[w] for w in c if w in word_embed_dict])
#            w_r.append(w_c)
#        w_s = sequence.pad_sequences(w_r, maxlen=word_seq_length, padding=mode, truncating=mode, value=0)
#        regularized_index = index - counter*batch_vol
#        contents_seq[regularized_index][:len(w_s)] = w_s
#        if (index+1) % batch_vol == 0:
#            label_han = labels[batch_vol*counter:batch_vol*(counter+1)]
#            np.save(han_cache+folder+'X_%d.npy'%counter, contents_seq)
#            np.save(han_cache+folder+'y_%d.npy'%counter, label_han)
#            contents_seq = np.zero(shape=(batch_vol, sen_seq_length, word_seq_length))
#            counter+=1
#    label_han = labels[counter*batch_vol:]
#    np.save(han_cache+folder+'X_%d.npy'%counter, contents_seq)
#    np.save(han_cache+folder+'y_%d.npy'%counter, label_han)
        
            
        
                         
                         
def word_han_preprocess(contents, sentence_num=sen_seq_length, word_num=word_seq_length, keep=False):
    contents_seq = np.zeros(shape=(len(contents), sentence_num, word_num))
    for index, content in enumerate(contents):
        if index >= (len(contents)): break
        sentences = SentenceSplitter.split(content)
        word_seq = get_sequence(sentences, maxlen=word_num)
        word_seq = word_seq[:sentence_num]
        contents_seq[index][:len(word_seq)] = word_seq
    return contents_seq
                         


def get_sequence(contents, maxlen=sentence_length, mode='post', keep=False):
    word_r = []
    contents = "\n".join(contents)
    contents = " ".join(list(jieba.cut(contents))).replace(" \n ", "\n")
    contents = [content.split(" ") for content in contents.split('\n')]
    for content in tqdm(contents, disable=True):
        if keep:
            word_c = np.array([word_embed_dict[w] if w in word_embed_dict else word_unknown for w in content])
        else:
            word_c = np.array([word_embed_dict[w] for w in content if w in word_embed_dict])
        word_r.append(word_c)
    word_seq = sequence.pad_sequences(word_r, maxlen=maxlen, padding=mode, truncating=mode, value=0)
    return word_seq


def batch_generator(contents, labels, batch_size=128, shuffle=True, keep=False, preprocessfunc=None):
    assert preprocessfunc != None
    sample_size = contents.shape[0]
    idx_array = np.arange(sample_size)
    while 1:
        if shuffle:
            np.random.shuffle(idx_array)
        batches = make_batches(sample_size, batch_size)
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            batch_ids = idx_array[batch_start:batch_end]
            batch_contents = contents[batch_ids]
            batch_contents = preprocessfunc(batch_contents, keep=keep)
            batch_labels = to_categorical(labels[batch_ids])
            yield (batch_contents, batch_labels)
            

# func= 'seq', 'han'

def batch_generator2(batch_size=128, shuffle=True, func=None, isTrain=False):
    assert func != None
    if isTrain:
        location = 'train/'
        
    else:
        location = 'test/'
    cache_path = cachePrefix+func +'/'
    fileList = os.listdir()
    npy_NB = len(fileList)//2
    while 1:
        for i in range(npy_NB):
            now_X = np.load(cache_path+location+'X_%d.npy'%i)
            now_y = np.load(cache_path+location+'y_%d.npy'%i)
            SZ = now_y.shape[0]
            idx = np.arange(SZ)
            if shuffle:
                np.random.shuffle(idx)
            batches = make_batch(SZ, batch_size)
            for batch_index, (batch_start, batch_end) in enumerate(batches):
                batch_ids = idx[batch_start:batch_end]
                batch_X = now_X[batch_ids]
                batch_y = now_y[batch_ids]
                yield (batch_x, batch_y)
                
def make_batches(size, batch_size):
    nb_batch = int(np.ceil(size/float(batch_size)))
    return [(i*batch_size, min(size, (i+1)*batch_size)) for i in range(0, nb_batch)]

def word_text_preprocess(contents, word_maxlen=sentence_length, keep=False):
    word_seq = get_sequence(contents, maxlen=word_maxlen, keep=keep)
    return word_seq



def word_text_train_batch_generator(train_content, train_label, batch_size=128, keep=False):
    return batch_generator(contents=train_content, labels=train_label, batch_size=batch_size, keep=keep, preprocessfunc=word_text_preprocess)

def word_han_train_batch_generator(train_content, train_label, batch_size=128, keep=False):
    return batch_generator(contents=train_content, labels=train_label, batch_size=batch_size, keep=keep, preprocessfunc=word_han_preprocess)
