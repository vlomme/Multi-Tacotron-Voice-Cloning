import numpy as np
import tensorflow as tf
from tqdm import tqdm as tqdm
from distance import levenshtein
import os
import math
import re

class Hparams:
    batch_size = 512
    enc_maxlen = 20 # The maximum size of the graphemes word (Максимальный размер исходного слова)
    dec_maxlen = 20 # The maximum size of the phonemes word (Максимальный размер закодированного слова)
    num_epochs = 500 # Number of Epochs(Количество Эпох)
    hidden_units = 128 # Hidden layers(Скрытых слоёв)
    lr = 0.001 
    
    graphemes = ["<pad>", "<unk>", "</s>"] + list(".,?!abcdefghijklmnopqrstuvwxyzабвгдеёжзийклмнопрстуфхцчшщъыьэюя")
    phonemes = ["<pad>", "<unk>", "<s>", "</s>", "оу", "ай", "ей", "ой","Оу", "Ай", "Ей", "Ой"] + list(".,?!ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя")
    # dictionary(Словарь)
    dicdir = "g2p/en.dic"
    # Save Model( Сохранённая модель)
    logdir = "g2p/log/"
    
def load_dict(path_dict):
    phon = []
    words = {}
    with open(path_dict,"r", encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(' ',1)
            
            if (parts[0] in words):
                words[parts[0]]=words[parts[0]]+[parts[1].split(' ')]
            else:
                words[parts[0]]=[parts[1].split(' ')]
    return words 

def encode(inp, type, dict):
    '''type: "x" or "y"'''
    inp_str = inp.decode("utf-8")
    if type=="x": tokens = inp_str.split() + ["</s>"]
    else: tokens = ["<s>"] + inp_str.split() + ["</s>"]

    x = [dict.get(t, dict["<unk>"]) for t in tokens]

    return x
    
def generator_fn(words, prons):
    '''
    words: 1d byte array. e.g., [b"w o r d", ]
    prons: 1d byte array. e.g., [b'W ER1 D', ]
    
    yields
    xs: tuple of
        x: list of encoded x. encoder input
        x_seqlen: scalar.
        word: string
        
    ys: tuple of
        decoder_input: list of decoder inputs
        y: list of encoded y. label.
        y_seqlen: scalar.
        pron: string
    '''
    g2idx, idx2g, p2idx, idx2p = load_vocab()
    for word, pron in zip(words, prons):
        x = encode(word, "x", g2idx)
        y = encode(pron, "y", p2idx)
        decoder_input, y = y[:-1], y[1:]

        x_seqlen, y_seqlen = len(x), len(y)
        yield (x, x_seqlen, word), (decoder_input, y, y_seqlen, pron)

def input_fn(words, prons, batch_size, shuffle=False):
    '''Batchify data
    words: list of words. e.g., ["word", ]
    prons: list of prons. e.g., ['W ER1 D',]
    batch_size: scalar.
    shuffle: boolean
    '''
    shapes = ( ([None], (), ()),
               ([None], [None], (), ())  )
    types = (  (tf.int32, tf.int32, tf.string),
               (tf.int32, tf.int32, tf.int32, tf.string)  )
    paddings = (  (0, 0, ''),
                  (0, 0, 0, '')  )

    dataset = tf.data.Dataset.from_generator(
        generator_fn,
        output_shapes=shapes,
        output_types=types,
        args=(words, prons)) # <- converted to np string arrays
        
    if shuffle:
        dataset = dataset.shuffle(128*batch_size)    
    dataset = dataset.repeat() # iterate forever
    dataset = dataset.padded_batch(batch_size, shapes, paddings).prefetch(1)

    return dataset
def get_batch(words, prons, batch_size, shuffle=False):
    '''Gets training / evaluation mini-batches
    fpath1: source file path. string.
    fpath2: target file path. string.
    maxlen1: source sent maximum length. scalar.
    maxlen2: target sent maximum length. scalar.
    vocab_fpath: string. vocabulary file path.
    batch_size: scalar
    shuffle: boolean

    Returns
    batches
    num_batches: number of mini-batches
    num_samples
    '''
    batches = input_fn(words, prons, batch_size, shuffle=shuffle)
    num_batches = calc_num_batches(len(words), batch_size)
    #print(1)
    #print(num_batches)
    #print(2)
    return batches, num_batches, len(words)
def convert_idx_to_token_tensor(inputs, idx2token):
    '''Converts int32 tensor to string tensor.
    inputs: 1d int32 tensor. indices.
    idx2token: dictionary

    Returns
    1d string tensor.
    '''
    def my_func(inputs):
        return " ".join(idx2token[elem] for elem in inputs)

    return tf.py_func(my_func, [inputs], tf.string)
class Net:
    def __init__(self, hp):
        self.g2idx, self.idx2g, self.p2idx, self.idx2p = load_vocab()
        self.hp = hp
    
    def encode(self, xs):
        '''
        xs: tupple of 
            x: (N, T). int32
            seqlens: (N,). int32
            words: (N,). string
            
        returns
        last hidden: (N, hidden_units). float32
        words: (N,). string
        '''
        with tf.variable_scope("encode", reuse=tf.AUTO_REUSE):
            x, seqlens, words = xs
            x = tf.one_hot(x, len(self.g2idx))
            cell = tf.contrib.rnn.GRUCell(self.hp.hidden_units)
            _, last_hidden = tf.nn.dynamic_rnn(cell, x, seqlens, dtype=tf.float32)
            
        return last_hidden, words
        
    
    def decode(self, ys, h0=None):
        '''
        ys: tupple of 
            decoder_inputs: (N, T). int32
            y: (N, T). int32
            seqlens: (N,). int32
            prons: (N,). string.
        h0: initial hidden state. (N, hidden_units)
        
        returns
        logits: (N, T, len(p2idx)). float32. before softmax
        y_hat: (N, T). int32.
        y: (N, T). int32. label.
        prons: (N,). string. ground truth phonemes 
        last_hidden: (N, hidden_units). This is for autoregressive inference
        '''
        decoder_inputs, y, seqlens, prons = ys
       
        with tf.variable_scope("decode", reuse=tf.AUTO_REUSE):
            inputs = tf.one_hot(decoder_inputs, len(self.p2idx))
            
            cell = tf.contrib.rnn.GRUCell(self.hp.hidden_units)
            outputs, last_hidden = tf.nn.dynamic_rnn(cell, inputs, initial_state=h0, dtype=tf.float32)

            # projection
            logits = tf.layers.dense(outputs, len(self.p2idx))
            y_hat = tf.to_int32(tf.argmax(logits, axis=-1))
        
        return logits, y_hat, y, prons, last_hidden
            
    def train(self, xs, ys):
        # forward
        last_hidden, words = self.encode(xs)
        logits, y_hat, y, prons, last_hidden = self.decode(ys, h0=last_hidden)
        
        # train scheme
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
        nonpadding = tf.to_float(tf.not_equal(y, self.p2idx["<pad>"])) # 0: <pad>
        loss = tf.reduce_sum(ce*nonpadding) / (tf.reduce_sum(nonpadding)+1e-7)

        global_step = tf.train.get_or_create_global_step()
        train_op = tf.train.AdamOptimizer(hp.lr).minimize(loss, global_step=global_step)
        
        return loss, train_op, global_step

    
    def eval(self, xs, ys):
        '''Predicts autoregressively
        At inference input ys is ignored.
        Returns
        y_hat: (N, T2)
        '''
        decoder_inputs, y, seqlens, prons = ys

        decoder_inputs = tf.ones((tf.shape(xs[0])[0], 1), tf.int32) * self.p2idx["<s>"]

        ys = (decoder_inputs, y, seqlens, prons)

        last_hidden, words = self.encode(xs)
       
        h0 = last_hidden
        y_hats = []
        print("Inference graph is being built. Please be patient.")
        for t in tqdm(range(self.hp.dec_maxlen)):
            _, y_hat, _, _, h0 = self.decode(ys, h0)
            if tf.reduce_sum(y_hat, 1)==0: break
           
            ys = (y_hat, y, seqlens, prons)
            y_hats.append(tf.squeeze(y_hat))

        y_hats = tf.stack(y_hats, 1)
        
        # monitor a random sample
        n = tf.random_uniform((), 0, tf.shape(y_hats)[0]-1, tf.int32)
        word = words[n]
        pred = convert_idx_to_token_tensor(y_hats[n], self.idx2p)
        pron = prons[n]
        
        return y_hats, word, pred, pron
def calc_num_batches(total_num, batch_size):
    return total_num // batch_size + int(total_num % batch_size != 0)

# evaluation metric
def per(ref, hyp):
    '''Calc phoneme error rate'''
    num_phonemes, num_erros = 0, 0
    g2idx, idx2g, p2idx, idx2p = load_vocab()
    for r, h in zip(ref, hyp):
        r = r.split()
        h = " ".join(idx2p[idx] for idx in h)
        h = h.split("</s>")[0].strip().split()
        
        num_phonemes += len(r)
        num_erros += levenshtein(h, r)
#         print(h, r)
    per = round(num_erros / num_phonemes, 2)
    return per


def drop_lengthy_samples(words, prons, enc_maxlen, dec_maxlen):
    """We only include such samples less than maxlen."""
    _words, _prons = [], []
    for w, p in zip(words, prons):
        if len(w.split()) + 1 > enc_maxlen: continue
        if len(p.split()) + 1 > dec_maxlen: continue # 1: <EOS>
        _words.append(w)
        _prons.append(p)
    return _words, _prons
def prepare_data():
    words = [" ".join(list(word)) for word, prons in cmu.items()]
    prons = [" ".join(prons[0]) for word, prons in cmu.items()]
    indices = list(range(len(words)))
    from random import shuffle
    shuffle(indices)
    words = [words[idx] for idx in indices]
    prons = [prons[idx] for idx in indices]
    num_train, num_test = int(len(words)*.95), int(len(words)*.001)
    train_words, eval_words, test_words = words[:num_train], \
                                          words[num_train:-num_test],\
                                          words[-num_test:]
    train_prons, eval_prons, test_prons = prons[:num_train], \
                                          prons[num_train:-num_test],\
                                          prons[-num_test:]    
    return train_words, eval_words, test_words, train_prons, eval_prons, test_prons
def load_vocab():
    g2idx = {g: idx for idx, g in enumerate(hp.graphemes)}
    idx2g = {idx: g for idx, g in enumerate(hp.graphemes)}

    p2idx = {p: idx for idx, p in enumerate(hp.phonemes)}
    idx2p = {idx: p for idx, p in enumerate(hp.phonemes)}

    return g2idx, idx2g, p2idx, idx2p # note that g and p mean grapheme and phoneme, respectively.    

def g2p(text):
    global hp 
    hp = Hparams()
    re_for_russian_letters = re.compile(
        r'[^.,?!ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя ]',
        re.U
    )
    k = 0
    text_str=[]
    text_all=[]
    for tex in text:
        tex = re.sub(re_for_russian_letters, '', tex).replace(",", " ,").replace(".", " .").replace("!", " !").replace("?", " ?")
        tex = tex.lower().strip().split(' ')

        for te in tex:
            k = k+1
            text_str.append(""+' '.join(list(te)))
        text_str.append("<pad>")
        text_str.append("<pad>")
        k = k+2
    text_all = text_str
    out_phons = []
    ckpt = tf.train.latest_checkpoint(hp.logdir)
    tf.reset_default_graph()
    if (k>200):
        bs=hp.batch_size
    else:
        bs = k
    test_batches, num_test_batches, num_test_samples  = get_batch(text_all, text_all,
                                                                  bs,
                                                                  shuffle=False)
    iter = tf.data.Iterator.from_structure(test_batches.output_types, test_batches.output_shapes)
    test_init_op = iter.make_initializer(test_batches)
    net = Net(hp)        
    xs, ys = iter.get_next()
    y_hat, _, _, _ = net.eval(xs, ys)
    saver = tf.train.Saver()
    out_phon = ""
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    #config.gpu_options.per_process_gpu_memory_fraction = 0.5
    with tf.Session(config=config) as sess:
        
        saver.restore(sess, ckpt); 
        sess.run(test_init_op)
        _y_hats = []
        for i in range(num_test_batches):
            _y_hat = sess.run(y_hat)
            _y_hats.extend(_y_hat.tolist())
            if (i % 10==0):
                print(i,"/",num_test_batches)
        g2idx, idx2g, p2idx, idx2p = load_vocab()
        num_test_batches = len(_y_hats)
        for h in  _y_hats:
            num_test_batches = num_test_batches-1
            h = " ".join(idx2p[idx] for idx in h)
            if (num_test_batches % 10000==0):
                print(num_test_batches) 
            h = h.split("</s>")[0].strip().replace(" ", "")
            
            out_phon = out_phon +" " + h 
    out_phon = out_phon.replace(" ,", ",").replace(" .", ".").replace(" !", "!").replace(" ?", "?")
    #
    #print(out_phon)
    out_phons = out_phon.split(" E E ")
    out_phons[len(out_phons)-1]=out_phons[len(out_phons)-1].replace(" E E", "")
    #out_phons2=[]
    #for phon in out_phons:
    #    out_phons2.append(phon.replace(" E E", ""))
    return out_phons



    



def tn():
    global hp
    hp = Hparams()
    global cmu
    cmu = load_dict(hp.dicdir)
  
        
    train_words, eval_words, test_words, train_prons, eval_prons, test_prons = prepare_data()

    tf.reset_default_graph()
    # prepare batches
    train_batches, num_train_batches, num_train_samples = get_batch(train_words, train_prons,
                             hp.batch_size, shuffle=True)
    eval_batches, num_eval_batches, num_eval_samples = get_batch(eval_words, eval_prons,
                             hp.batch_size, shuffle=False)
    # create a iterator of the correct shape and type
    iter = tf.data.Iterator.from_structure(train_batches.output_types, train_batches.output_shapes)

    # create the initialisation operations
    train_init_op = iter.make_initializer(train_batches)
    eval_init_op = iter.make_initializer(eval_batches)
    # variable specs
    def print_variable_specs(fpath):
        def get_size(shp):
            size = 1
            for d in range(len(shp)):
                size *=shp[d]
            return size

        params, num_params = [], 0
        for v in tf.global_variables():
            params.append("{}==={}\n".format(v.name, v.shape))
            num_params += get_size(v.shape)
        print("num_params:", num_params)
    # Load model
    net = Net(hp)
    xs, ys = iter.get_next()
    loss, train_op, global_step = net.train(xs, ys)
    y_hat, word, pred, pron = net.eval(xs, ys)
    # Session
    saver = tf.train.Saver()
    with tf.Session() as sess:
        ckpt = tf.train.latest_checkpoint(hp.logdir)
        if ckpt is None:
            sess.run(tf.global_variables_initializer())
            print("Variables initialized")
        else:
            saver.restore(sess, ckpt)
        sess.run(train_init_op)
        total_steps = hp.num_epochs*num_train_batches
        _gs = sess.run(global_step)
        for _ in tqdm(range(_gs, total_steps+1)):
            # training
            _, _gs, _loss = sess.run([train_op, global_step,loss]) 

            epoch = math.ceil(_gs / num_train_batches)
                
            if _gs and _gs % num_train_batches == 0: # Be careful that you should evaluate at every epoch due to train_init_op
                print("epoch=", epoch, "is done!")
                sess.run(eval_init_op)
                _y_hats = []
                for _ in range(num_eval_batches):
                    _y_hat, _word, _pred, _pron = sess.run([y_hat, word, pred, pron])
                    _y_hats.extend(_y_hat.tolist())
                    
                # sample monitor
                print("wrd:", _word.decode("utf-8"))
                print("exp:", _pron.decode("utf-8"))
                print("got:", _pred.decode("utf-8"))
                    
                
                _per = per(eval_prons, _y_hats)
                print("per=%.2f"%_per)
                print()
                      
                sess.run(train_init_op)
                
                # save
                if not os.path.exists(hp.logdir): os.makedirs(hp.logdir)
                fname = os.path.join(hp.logdir, "my_model_loss_%.2f_per_%.2f" % (_loss, _per))
                saver.save(sess, fname, global_step=_gs)
       
        print("Training Done!")
                  
    print("Done!")
