
import os
os.environ["TF_CUDNN_RESET_RND_GEN_STATE"] = '1' # tf bug workaround
from sound_classifier import SoundClassifier
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Dropout, TimeDistributed, InputLayer, Bidirectional, Softmax, BatchNormalization, Lambda, LayerNormalization, Conv1D, MaxPooling1D
import tensorflow_addons as tfa # Must install tensorflow addons separately
from tensorflow_addons.layers import InstanceNormalization

from tensorflow.compat.v1.keras.layers import CuDNNGRU # can only run on GPU
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.nn import ctc_loss
from densenet import DenseNetBlock
import random
import numpy as np
import pandas as pd
import pickle
import gc
from pinyin_tagger import ChineseTokenizer
import os
#tf.enable_eager_execution() # Since we're using TF 1.x, in 2.x this is enabled by default
tf.compat.v1.enable_eager_execution()

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

class CTCClassifier:
    @staticmethod
    def lrelu(x):
        return tf.keras.activations.relu(x, alpha=0.1)

    def __init__(self, t_width=13, n_mfcc=100, stride=5):
        self.sc = SoundClassifier()
        self.t_width = t_width
        self.n_mfcc = n_mfcc
        self.stride = stride
        self.create_model()
        self.softmax = Softmax()
        self.aishell_filenames = ['C:/Users/Ahmad/Downloads/data_aishell3\\train\\train_0.p',
                                  'C:/Users/Ahmad/Downloads/data_aishell3\\train\\train_1.p',
                                  'C:/Users/Ahmad/Downloads/data_aishell3\\train\\train_2.p',
                                  'C:/Users/Ahmad/Downloads/data_aishell3\\train\\train_3.p',
                                  'C:/Users/Ahmad/Downloads/data_aishell3\\train\\train_4.p',
                                  'C:/Users/Ahmad/Downloads/data_aishell3\\train\\train_5.p',
                                  'C:/Users/Ahmad/Downloads/data_aishell3\\train\\train_6.p']
        self.aishell_test_filenames = ['C:/Users/Ahmad/Downloads/data_aishell3\\test\\test_0.p',
                                       'C:/Users/Ahmad/Downloads/data_aishell3\\test\\test_1.p',
                                       'C:/Users/Ahmad/Downloads/data_aishell3\\test\\test_2.p']
        custom_object = {
            'lrelu': CTCClassifier.lrelu,
        }
        keras.utils.get_custom_objects().update(custom_object)

    def create_ff_rnn(self):
        self.model = Sequential([
            InputLayer(input_shape=(None, self.t_width, self.n_mfcc, 1)),
            TimeDistributed(Lambda(lambda x: x[:, :, :20, :])),
            TimeDistributed(Flatten()),
            TimeDistributed(Dense(128, activation='relu')),
            BatchNormalization(),
            TimeDistributed(Dropout(0.05)),
            TimeDistributed(Dense(128, activation='relu')),
            BatchNormalization(),
            TimeDistributed(Dropout(0.05)),
            TimeDistributed(Dense(128, activation='relu')),
            BatchNormalization(),
            TimeDistributed(Dropout(0.05)),
            Bidirectional(keras.layers.GRU(128, return_sequences=True)),
            BatchNormalization(),
            TimeDistributed(Dense(128, activation='relu')),
            BatchNormalization(),
            TimeDistributed(Dropout(0.05)),
            TimeDistributed(Dense(6))
        ])

    def create_densenet2(self):
        self.model = Sequential([
            InputLayer(input_shape=(None, self.t_width, self.n_mfcc, 1)),
            TimeDistributed(Lambda(lambda x: x[:, :, 0:20, :])),
            #Normalization(axis=-2),
            DenseNetBlock(n_layers=3, n_filters=20),
            TimeDistributed(MaxPooling2D()),
            DenseNetBlock(n_layers=3, n_filters=40),
            DenseNetBlock(n_layers=3, n_filters=80),
            TimeDistributed(MaxPooling2D()),
            DenseNetBlock(n_layers=3, n_filters=80),
            TimeDistributed(MaxPooling2D()),
            TimeDistributed(Flatten()),
            Bidirectional(keras.layers.GRU(16, return_sequences=True)),
            #Normalization(),
            #BatchNormalization(center=False, scale=False),
            #LayerNormalization(axis=-2),
            #TimeDistributed(Dense(128, activation=CTCClassifier.lrelu)),
            #BatchNormalization(center=False, scale=False),
            #LayerNormalization(axis=-2),
            TimeDistributed(Dense(6))
        ])

        

    def create_17_6_21(self):
        # (time, mfcc, channels) # used n_mfcc=100, t_width=13
        self.model = Sequential([
            InputLayer(input_shape=(None, self.t_width, self.n_mfcc, 1)),
            #BatchNormalization(),
            TimeDistributed(Conv2D(filters=20, kernel_size=3, strides=(2, 2), padding='valid', activation=CTCClassifier.lrelu,
                                   kernel_regularizer=keras.regularizers.l2(l=0.001)), input_shape=[None, self.t_width, self.n_mfcc, 1]),
            #BatchNormalization(),
            InstanceNormalization(),
            TimeDistributed(Dropout(0.05)),
            
            TimeDistributed(Conv2D(filters=40, kernel_size=3, strides=(1, 2), padding='valid', activation=CTCClassifier.lrelu,
                                   kernel_regularizer=keras.regularizers.l2(l=0.001))),
            #BatchNormalization(),
            InstanceNormalization(),
            TimeDistributed(Dropout(0.05)),
            
            TimeDistributed(Conv2D(filters=80, kernel_size=3, strides=(1, 2), padding='same', activation=CTCClassifier.lrelu, kernel_regularizer=keras.regularizers.l2(l=0.01))),
            #BatchNormalization(),
            InstanceNormalization(),
            TimeDistributed(Dropout(0.05)),
            
            TimeDistributed(Conv2D(filters=80, kernel_size=3, strides=(1, 1), padding='same', activation=CTCClassifier.lrelu, kernel_regularizer=keras.regularizers.l2(l=0.01))),
            #BatchNormalization(),
            InstanceNormalization(),
            TimeDistributed(Dropout(0.05)),
            
            TimeDistributed(Conv2D(filters=20, kernel_size=3, strides=(1, 2), padding='valid', activation=CTCClassifier.lrelu, kernel_regularizer=keras.regularizers.l2(l=0.01))),
            #BatchNormalization(),
            InstanceNormalization(),
            TimeDistributed(Dropout(0.05)),
            
            TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="valid")),

            TimeDistributed(Flatten()),
            
            Bidirectional(keras.layers.LSTM(64, return_sequences=True)),
            #BatchNormalization(),
            InstanceNormalization(),
            # remove softmax activation since tf.nn.ctc_loss requires pre-softmax outputs (called logits).
            TimeDistributed(Dense(6, kernel_regularizer=keras.regularizers.l2(l=0.001)))
        ])



    def create_conv1d(self):
        self.model = Sequential([
            InputLayer(input_shape=(None, self.t_width, self.n_mfcc, 1)),
            TimeDistributed(Lambda(lambda x: x[:, :, 0:20, 0])),
            TimeDistributed(Conv1D(filters=40, kernel_size=3, padding='valid', activation=CTCClassifier.lrelu)),
            BatchNormalization(),
            TimeDistributed(Conv1D(filters=80, kernel_size=3, padding='valid', activation=CTCClassifier.lrelu)),
            BatchNormalization(),
            TimeDistributed(Conv1D(filters=160, kernel_size=4, padding='valid', activation=CTCClassifier.lrelu)),
            BatchNormalization(),
            TimeDistributed(MaxPooling1D()),
            TimeDistributed(Flatten()),
            Bidirectional(keras.layers.GRU(16, return_sequences=True)),
            #BatchNormalization(),
            #Normalization(),
            #BatchNormalization(center=False, scale=False),
            #LayerNormalization(axis=-2),
            #TimeDistributed(Dense(128, activation=CTCClassifier.lrelu)),
            #BatchNormalization(center=False, scale=False),
            #LayerNormalization(axis=-2),
            TimeDistributed(Dense(6))
        ])
        
    def create_densenet(self):
        self.model = Sequential([
            InputLayer(input_shape=(None, self.t_width, self.n_mfcc, 1)),
            TimeDistributed(Conv2D(filters=5, kernel_size=3, strides=(2, 2), padding='valid', activation=CTCClassifier.lrelu,
                                   input_shape=[None, self.t_width, self.n_mfcc, 1])),
            BatchNormalization(),
            TimeDistributed(Dropout(0.05)),
            DenseNetBlock(n_layers=5, n_filters=20),
            TimeDistributed(MaxPooling2D(pool_size=(1, 2))),
            #DenseNetBlock(n_layers=5, n_filters=40),
            #TimeDistributed(MaxPooling2D(pool_size=(1, 2))),
            DenseNetBlock(n_layers=5, n_filters=40),
            TimeDistributed(MaxPooling2D(pool_size=(1, 2))),
            TimeDistributed(Conv2D(filters=80, kernel_size=1, strides=(1, 2), padding='valid', activation='relu', kernel_regularizer=keras.regularizers.l2(l=0.01))),
            BatchNormalization(),
            TimeDistributed(Flatten()),
            TimeDistributed(Dense(128, activation='relu')),
            #Bidirectional(keras.layers.GRU(32, return_sequences=True)),
            BatchNormalization(),
            TimeDistributed(Dense(6))
        ])

    def create_aishell_10_6_21(self):
        # (time, mfcc, channels) # used n_mfcc=100, t_width=13
        self.model = Sequential([
            InputLayer(input_shape=(None, self.t_width, self.n_mfcc, 1)),
            #BatchNormalization(),
            TimeDistributed(Conv2D(filters=20, kernel_size=3, strides=(2, 2), padding='valid', activation=CTCClassifier.lrelu,
                                   kernel_regularizer=keras.regularizers.l2(l=0.001)), input_shape=[None, self.t_width, self.n_mfcc, 1]),
            BatchNormalization(),
            TimeDistributed(Dropout(0.05)),
            
            TimeDistributed(Conv2D(filters=40, kernel_size=3, strides=(1, 2), padding='valid', activation=CTCClassifier.lrelu,
                                   kernel_regularizer=keras.regularizers.l2(l=0.001))),
            BatchNormalization(),
            TimeDistributed(Dropout(0.05)),
            
            TimeDistributed(Conv2D(filters=80, kernel_size=3, strides=(1, 2), padding='same', activation=CTCClassifier.lrelu, kernel_regularizer=keras.regularizers.l2(l=0.01))),
            BatchNormalization(),
            TimeDistributed(Dropout(0.05)),
            
            TimeDistributed(Conv2D(filters=80, kernel_size=3, strides=(1, 1), padding='same', activation=CTCClassifier.lrelu, kernel_regularizer=keras.regularizers.l2(l=0.01))),
            BatchNormalization(),
            TimeDistributed(Dropout(0.05)),
            
            TimeDistributed(Conv2D(filters=20, kernel_size=3, strides=(1, 2), padding='valid', activation=CTCClassifier.lrelu, kernel_regularizer=keras.regularizers.l2(l=0.01))),
            BatchNormalization(),
            TimeDistributed(Dropout(0.05)),
            
            TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")),

            TimeDistributed(Flatten()),
            
            Bidirectional(keras.layers.LSTM(16, return_sequences=True)),
            BatchNormalization(),

            # remove softmax activation since tf.nn.ctc_loss requires pre-softmax outputs (called logits).
            TimeDistributed(Dense(6, kernel_regularizer=keras.regularizers.l2(l=0.001)))
        ])

    def create_model(self):
        # (time, mfcc, channels)
        self.model = Sequential([
            InputLayer(input_shape=(None, self.t_width, self.n_mfcc, 1)),
            #BatchNormalization(),
            TimeDistributed(Conv2D(filters=20, kernel_size=3, strides=(2, 2), padding='valid', activation=CTCClassifier.lrelu,
                                   kernel_regularizer=keras.regularizers.l2(l=0.001)), input_shape=[None, self.t_width, self.n_mfcc, 1]),
            BatchNormalization(),
            #LayerNormalization(axis=(-3,-2)),
            TimeDistributed(Dropout(0.05)),
            TimeDistributed(Conv2D(filters=40, kernel_size=3, strides=(1, 2), padding='valid', activation=CTCClassifier.lrelu,
                                   kernel_regularizer=keras.regularizers.l2(l=0.001))),
            BatchNormalization(),
            #LayerNormalization(axis=(-3,-2)),
            TimeDistributed(Dropout(0.05)),
            TimeDistributed(Conv2D(filters=80, kernel_size=3, strides=(1, 2), padding='same', activation=CTCClassifier.lrelu, kernel_regularizer=keras.regularizers.l2(l=0.01))),
            BatchNormalization(),
            #LayerNormalization(axis=(-3,-2)),
            TimeDistributed(Dropout(0.05)),
            TimeDistributed(Conv2D(filters=80, kernel_size=3, strides=(1, 1), padding='same', activation=CTCClassifier.lrelu, kernel_regularizer=keras.regularizers.l2(l=0.01))),
            BatchNormalization(),
            #LayerNormalization(axis=(-3,-2)),
            TimeDistributed(Dropout(0.05)),
            TimeDistributed(Conv2D(filters=20, kernel_size=3, strides=(1, 2), padding='valid', activation=CTCClassifier.lrelu, kernel_regularizer=keras.regularizers.l2(l=0.01))),
            BatchNormalization(),
            #LayerNormalization(axis=(-3,-2)),
            TimeDistributed(Dropout(0.05)),
            TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")),

            TimeDistributed(Flatten()),
            
            Bidirectional(keras.layers.LSTM(16, return_sequences=True)),
            BatchNormalization(),
            #LayerNormalization(axis=(-1)),
            # remove softmax activation since tf.nn.ctc_loss requires pre-softmax outputs (called logits).
            TimeDistributed(Dense(6, kernel_regularizer=keras.regularizers.l2(l=0.001)))
        ])

    def load_model(self, filename):
        self.model = keras.models.load_model(filename,
                                             custom_objects={'lrelu': CTCClassifier.lrelu, 'DenseNetBlock': DenseNetBlock})

    def save_model(self, filename):
        self.model.save(filename)


    def sliding_window(self, x, stride=None, random_offset=False, offset=0):
        if stride is None:
            stride = self.stride
        if random_offset:
            offset = np.random.randint(stride)
        return np.array([x[i+offset:i+offset+self.t_width,:] for i in range(0, len(x) - self.t_width + 1 - offset, stride)])

    def forvo_validation_score(self):
        val = self.create_training('C:/Users/Ahmad/Dropbox/Programming/ToneClassification/test_data',
                                   pickled_filename="C:/Users/Ahmad/Dropbox/Programming/ToneClassification/audio_test_data.p")
        fnames_val, xs_val, ys_val = val
        return self.validation_score(xs_val, ys_val)
    
    def accuracy_score(self, xs_val, ys_val, use_beam_search=True):
        n = len(xs_val)
        accuracy = 0.0
        for i in range(n):
            if use_beam_search:
                pred = self.beam_search_decode(xs_val[i])
            else:
                pred = self.predict_tones(xs_val[i])
            if ys_val[i] == pred:
                accuracy += 1.0
        return accuracy / n

    def avg_ctc_loss(self, xs_, ys_, batch_size=64):
        n = len(xs_)
        batches = tf.data.Dataset.range(n).shuffle(n).batch(batch_size).as_numpy_iterator()
        loss = 0.0
        for batch in batches:
            xs = [tf.convert_to_tensor(self.sliding_window(xs_[i])[np.newaxis,...,np.newaxis]) for i in batch]
            tones = [ys_[i] for i in batch]
            
            label_length = tf.constant([len(tone) for tone in tones])
            label_max_len = max([len(tone) for tone in tones])
            tones = [[int(c) for c in tone] + [0]*(label_max_len - len(tone)) for tone in tones]
            tones = tf.constant(tones) # shape [batch_size=1, max_label_seq_length]
            logit_length = tf.constant([x.shape[1] for x in xs])
            max_windows = max([x.shape[1] for x in xs])
            
            paddings = []
            logit_length = []
            for x in xs:
                num_windows = x.shape[1]
                logit_length.append(num_windows) 
                paddings.append([[0, max_windows - num_windows], [0, 0]])
            

            probs = []
            for i in range(len(xs)):
                x = xs[i]
                probs.append(tf.pad(self.model(x, training=False)[0], paddings[i]))
                #probs.append(tf.pad(self.model.predict(x)[0], paddings[i]))
                
            probs = tf.stack(probs)
            loss += tf.reduce_sum(ctc_loss(labels=tones, logits=probs, label_length=label_length,
                                           logit_length=logit_length, logits_time_major=False))
        return loss.numpy() / n

    def predict_from_audiofile(self, folder, filename):
        """ From filename to input to this sound classifier. """
        sc = SoundClassifier()
        mfcc = sc.mfcc(folder, filename, right_pad=256*(self.t_width-1)//2,
                                              left_pad=256*(self.t_width-1)//2, pad=None)
        mfcc = np.swapaxes(mfcc, 0, 1)
        return self.beam_search_decode(mfcc)
                
    def create_training(self, folder, pickled_filename=None):
        # Unpickled pickled_filename (which contains fnames, xs, ys).
        # Filters out those already loaded, and loads in any new ones. TODO.
        if pickled_filename:
            import pickle
            fnames_, xs_, ys_ = pickle.load(open(pickled_filename, "rb"))
        else:
            fnames_, xs_, ys_ = [], [], []
            
        # Each matrix in xs has dimension (time, mfcc) [this is compatible with keras RNN input convention]
        fnames, xs, ys = self.sc.get_training(folder=folder, filter_=fnames_, right_pad=256*(self.t_width-1)//2,
                                              left_pad=256*(self.t_width-1)//2, pad=None)

        xs = [np.swapaxes(x, 0, 1) for x in xs]

        # Remove cases where the tone has '33' since the actual pronunciation changes.
        f_res, x_res, y_res = fnames_ + fnames, xs_ + xs, ys_ + ys
        indices = [i for i in range(len(y_res)) if '33' not in y_res[i]]
        
        f_res = [f_res[i] for i in indices]
        x_res = [x_res[i] for i in indices]
        y_res = [y_res[i] for i in indices]
        return (f_res, x_res, y_res)

    def load_and_save_mozilla(self, tsv_file, save_filename):
        ct = ChineseTokenizer()
        mozilla = pd.read_csv(os.path.join('C:/Users/Ahmad/Downloads/zh-CN/cv-corpus-6.1-2020-12-11/zh-CN', tsv_file), sep='\t')
        mozilla['tones'] = mozilla['sentence'].apply(lambda s: ct.to_tones(s))
        filepath = 'C:/Users/Ahmad/Downloads/zh-CN/cv-corpus-6.1-2020-12-11/zh-CN/clips'
        filenames = mozilla['path'].tolist()
        tones = mozilla['tones'].tolist()
        #sentences = mozilla['sentence'].tolist()

        mfccs = self.load_mfccs(folderpath=filepath, filenames=filenames)
        
        import pickle
        pickle.dump((filenames, mfccs, tones),
                    open(os.path.join("C:/Users/Ahmad/Dropbox/Programming/ToneClassification", save_filename), "wb"))
        print("Done.")
        return filenames, mfccs, tones

    def load_labelled_from_folder(self, folderpath):
        """ Creates labelled data. Uses ChineseTokenizer to obtain the tone labels. 
            Assumes filenames start with the Chinese word then is followed by underscore.
        """
        filenames, xs = self.load_mfccs(folderpath, return_filenames=True, n_threads=10)
        ct = ChineseTokenizer()
        ys = [ct.to_tones(f.split('_')[0]) for f in filenames]
        for i in range(len(xs)-1,-1,-1):
            if xs[i].shape[0] <= self.t_width: # the audio file was empty
                del xs[i]
                del ys[i]
                del filenames[i]
                
        return filenames, xs, ys

    def load_mfccs(self, folderpath, filenames=None, return_filenames=False, n_threads=5):
        mfccs = []
        sc = SoundClassifier()
        if filenames is None:
            filenames = os.listdir(folderpath)
        
        def to_mfcc(fname_i):
            fname, i = fname_i
            if i % 100 == 0:
                print("loaded:", i)

            try:
                return sc.mfcc(folderpath, fname, right_pad=256*(self.t_width-1)//2,
                               left_pad=256*(self.t_width-1)//2, pad=None)
            except:
                print("Couldn't load:", fname)
                return None
            
        import multiprocessing.dummy
        with multiprocessing.dummy.Pool(n_threads) as p: 
            mfccs = p.map(to_mfcc, zip(filenames, list(range(len(filenames)))))

        # Retry loading any failed cases.
        for i in range(len(mfccs)):
            if mfccs[i] is None:
                #try:
                mfccs[i] = sc.mfcc(folderpath=folderpath, filename=filenames[i], right_pad=256*(self.t_width-1)//2,
                                       left_pad=256*(self.t_width-1)//2, pad=None)
                print("Reloaded:",filenames[i])
                #except:
                #    print("Retried and failed to load:", filenames[i])

        for i in range(len(mfccs)):
            if mfccs[i] is not None:
                mfccs[i] = np.swapaxes(mfccs[i], 0, 1)

        if return_filenames:
            return filenames, mfccs
        return mfccs
    

    def train_from_files(self, folderpath, filenames, tones, epochs=5, comp_val_sc=False, **kwargs):
        n = len(filenames)
        sc = SoundClassifier()
    
        for epoch in range(epochs):
            epoch_loss = 0.0
            loss = 0.0 # loss of last 500
            for i in range(len(filenames)):
                mfcc = sc.mfcc(folderpath, filenames[i], right_pad=256*(self.t_width-1)//2,
                                              left_pad=256*(self.t_width-1)//2, pad=None)
                mfcc = np.swapaxes(mfcc, 0, 1)
                x = tf.convert_to_tensor(self.sliding_window(mfcc)[np.newaxis,...,np.newaxis])
                sample_loss = self.training_step(x, [tones[i]], **kwargs)
                loss += sample_loss
                
                if i > 0 and i % 10 == 0:
                    print(f"inter-epoch {epoch} loss: {loss/10}, i={i}")
                    loss = 0.0

                if i % 30 == 0 and comp_val_sc:
                    print(f"validation score: {self.validation_score()}")

    def train(self, xs, ys, optimizer, epochs=5, print_interval=100, batch_size=4, validation_data=None, save_folder=None, **kwargs):
        #data = list(zip([self.sliding_window(x)[...,np.newaxis] for x in xs], ys)) # 1 channel to feed into conv layers.
        n = len(xs)
        #indices = list(range(n))
        #print("preprocessed input.")
        last_epoch_loss = np.inf
        best_val_wer = np.inf
        self.best_model = keras.models.clone_model(self.model)
        
        for epoch in range(epochs):
            #random.shuffle(indices)
            batches = tf.data.Dataset.range(n).shuffle(n).batch(batch_size).as_numpy_iterator()
            loss = 0.0
            j = 0
            epoch_loss = 0.0
            #for i in indices:
            for batch in batches:
                x_batch = [tf.convert_to_tensor(self.sliding_window(xs[i])[np.newaxis,...,np.newaxis]) for i in batch]
                y_batch = [ys[i] for i in batch]
                #x_batch = x_windows[i]
                #x_batch = [tf.convert_to_tensor(self.sliding_window(xs[i])[np.newaxis,...,np.newaxis])]
                #y_batch = [ys[i]]
                
                batch_loss = self.training_step(x_batch, y_batch, optimizer=optimizer, **kwargs) / batch_size
                epoch_loss += batch_loss
                loss += batch_loss
                j += 1
                if j % print_interval == 0:
                    print(f"batch loss: {loss/print_interval}, {j}, epoch loss: {epoch_loss/j}, epoch: {epoch}")
                    loss = 0.0
            print(f"epoch {epoch} loss: {epoch_loss/j}, lr: {optimizer.learning_rate.numpy()}")
            if epoch_loss > last_epoch_loss:
                optimizer.learning_rate = optimizer.learning_rate/2.0
            else:
                optimizer.learning_rate = optimizer.learning_rate*1.05
            last_epoch_loss = epoch_loss
            if validation_data is not None:
                xs_test, ys_test = validation_data
                wer = self.avg_wer_from_audios(xs_test, ys_test)
                if wer < best_val_wer:
                    best_val_wer = wer
                    self.best_model.set_weights(self.model.get_weights())
                    if save_folder:
                        self.save_model(save_folder)
                print(f"test set ctc: {self.avg_ctc_loss(xs_test, ys_test, batch_size=128):.5f}, acc: {self.accuracy_score(xs_test, ys_test):.5f}, wer: {wer:.5f}")

    def train_from_pickled(self, pickled_filenames, optimizer, epochs=5, print_interval=100, batch_size=32,
                           validation_data=None, save_folder=None, lr_scheduler=None, **kwargs):
        last_epoch_loss = np.inf
        
        best_val_ctc = np.inf
        self.best_model = keras.models.clone_model(self.model)
        
        for epoch in range(epochs):
            print(f"Epoch: {epoch} / {epochs}")
            if lr_scheduler is not None:
                optimizer.learning_rate = lr_scheduler(epoch)
                print(f"learning rate: {optimizer.learning_rate.numpy()}")
            loss = 0.0
            j = 0
            epoch_loss = 0.0
            for p_filename in pickled_filenames:
                filenames, xs, ys = pickle.load(open(p_filename, "rb"))
                ys = list(ys) # series to list
                n = len(xs)
                batches = tf.data.Dataset.range(n).shuffle(n).batch(batch_size).as_numpy_iterator()
                
                for batch in batches:
                    x_batch = [tf.convert_to_tensor(self.sliding_window(xs[i], random_offset=True)[np.newaxis,...,np.newaxis])
                               for i in batch]
                    y_batch = [ys[i] for i in batch]                
                    batch_loss = self.training_step(x_batch, y_batch, optimizer=optimizer, **kwargs) / batch_size
                    epoch_loss += batch_loss
                    loss += batch_loss
                    j += 1
                    if j % print_interval == 0:
                        print(f"batch loss: {loss/print_interval}, {j}, epoch loss: {epoch_loss/j}, epoch: {epoch}")
                        loss = 0.0
                xs_test, ys_test = validation_data
                wer = self.avg_wer_from_audios(xs_test, ys_test)
                print(f"epoch {epoch} loss: {epoch_loss/j}, lr: {optimizer.learning_rate.numpy()}, {p_filename}")
                #print(f"test set ctc: {self.avg_ctc_loss(xs_test, ys_test, batch_size=128):.5f}, acc: {self.accuracy_score(xs_test, ys_test):.5f}, wer: {wer:.5f}")
                
                if validation_data is not None:
                    xs_test, ys_test = validation_data
                    wer = self.avg_wer_from_audios(xs_test, ys_test)
                    val_ctc = self.avg_ctc_loss(xs_test, ys_test, batch_size=128)
                    print(f"test set ctc: {val_ctc:.5f}, acc: {self.accuracy_score(xs_test, ys_test):.5f}, wer: {wer:.5f}")

                    if val_ctc < best_val_ctc:
                        best_val_ctc = val_ctc
                        self.best_model.set_weights(self.model.get_weights())
                        if save_folder:
                            self.save_model(save_folder)
                            
                    training_ctc = self.avg_ctc_loss(xs[:300], ys[:300], batch_size=128)
                    training_wer = self.avg_wer_from_audios(xs[:300], ys[:300])
                    training_acc = self.accuracy_score(xs[:300], ys[:300])
                    print(f"train set ctc: {training_ctc:.5f}, acc: {training_acc:.5f}, wer: {training_wer:.5f}")
                    
                del filenames
                del xs
                del ys
                gc.collect()

            if lr_scheduler is not None:
                optimizer.learning_rate = lr_scheduler(epoch)
            elif epoch_loss > last_epoch_loss:
                optimizer.learning_rate = optimizer.learning_rate/2.0
            else:
                optimizer.learning_rate = optimizer.learning_rate*1.05
            last_epoch_loss = epoch_loss
                
            if validation_data is not None:
                xs_test, ys_test = validation_data
                wer = self.avg_wer_from_audios(xs_test, ys_test)
                print(f"test set ctc: {self.avg_ctc_loss(xs_test, ys_test, batch_size=128):.5f}, acc: {self.accuracy_score(xs_test, ys_test):.5f}, wer: {wer:.5f}")

    def predict(self, x, apply_softmax=True, training_mode=False):
        inputs = tf.convert_to_tensor(self.sliding_window(x)[np.newaxis,...,np.newaxis])
        if training_mode:
            outputs = self.model(inputs, training=True)[0]
        else:
            outputs = self.model.predict(inputs)[0]
            
        if apply_softmax:
            outputs = self.softmax(outputs).numpy()
        return outputs
        #return self.model(self.sliding_window(x)[np.newaxis,...,np.newaxis]).numpy()[0]

    def beam_search_decode(self, x):
        logits = self.predict(x, apply_softmax=False)
        logits = tf.expand_dims(logits, 1)
        
        # tf.nn.ctc_beam_search_decoder uses the convention that the LAST index is blank (we use first).
        logits = logits.numpy()
        blank_logits = np.copy(logits[:, :, 0])
        logits[:, :, 0] = logits[:, :, 5]
        logits[:, :, 5] = blank_logits
        
        sequence_length = [logits.shape[0]]
        sparse = tf.nn.ctc_beam_search_decoder(logits, sequence_length, beam_width=100)
        top_path = tf.sparse.to_dense(sparse[0][0])
        
        res = ''.join([str(c) for c in list(top_path.numpy()[0]) if c != 5]) # 5's represent blanks
        res = res.replace('0', '5')
        return res

    def predict_classes(self, x):
        #inputs = tf.convert_to_tensor(self.sliding_window(x)[np.newaxis,...,np.newaxis])
        #return np.argmax(self.model.predict(inputs)[0], axis=1)
        return np.argmax(self.predict(x), axis=1)

    def predict_tones(self, x):
        out = self.predict_classes(x)
        tones = ''
        for i in range(out.shape[0]):
            if out[i] != 0 and (i == 0 or out[i] != out[i-1]):
                tones += str(out[i])
        return tones

    def training_step(self, xs, tones, optimizer=tf.keras.optimizers.SGD(learning_rate=1e-5, nesterov=False), verbose=False):
        # xs is a list of np.arrays with shape (1, windows, time, mfcc, 1)

        label_length = tf.constant([len(tone) for tone in tones])
        label_max_len = max([len(tone) for tone in tones])
        tones = [[int(c) for c in tone] + [0]*(label_max_len - len(tone)) for tone in tones]
        tones = tf.constant(tones) # shape [batch_size=1, max_label_seq_length]
        logit_length = tf.constant([x.shape[1] for x in xs])
        max_windows = max([x.shape[1] for x in xs])
        
        paddings = []
        logit_length = []
        for x in xs:
            num_windows = x.shape[1]
            logit_length.append(num_windows) 
            paddings.append([[0, max_windows - num_windows], [0, 0]])
        
        with tf.GradientTape() as tape:
            probs = []
            #reg_loss = 0.0
            for i in range(len(xs)):
                x = xs[i]
                probs.append(tf.pad(self.model(x, training=True)[0], paddings[i]))
                #reg_loss += tf.reduce_sum(self.model.losses)
            
            probs = tf.stack(probs)
            loss = tf.reduce_sum(ctc_loss(labels=tones, logits=probs, label_length=label_length, logit_length=logit_length, logits_time_major=False))
            #loss += reg_loss

        if verbose:
            print(f"loss: {loss}")
            #print(f"reg_loss: {reg_loss}")

        loss_np = loss.numpy()
        if not np.isnan(loss_np):
            #if loss_np < 100: # large loss can cause exploding gradients, skip these examples.
            if True:
                gvs = tape.gradient(loss, self.model.trainable_variables)
                #capped_gvs = [tf.clip_by_value(grad, -1., 1.) for grad in gvs]
                capped_gvs, _ = tf.clip_by_global_norm(gvs, 90.0)
                try:
                    for grad in capped_gvs:
                        tf.debugging.check_numerics(grad, message=f'Gradient check failed, possible NaNs: grad')
                    optimizer.apply_gradients(zip(gvs, self.model.trainable_variables))
                except Exception as e:
                    pass
                    #print("found nans", e)
                    
                #print("capped gradients:", capped_gvs)
        else:
            return 100.0 * len(xs)
            #print("loss returned nan")
            #print("tones:", tones)
            #print(f"probs = {self.model.predict(xs)}")
        
        return loss_np
        

    def ctc_score(self, probs, tones):
        """ Returns the CTC score, where `tones` is the target tone string, and probs[i,j] is the prob that time i is
            a j-th tone. The CTC score is the probability that an agent following probs would output
            a string representing tones.
            Params:
            probs: probs[i,j] is the probability that time i was a j-th tone.
        """
        target_string = '0' + '0'.join(tones) + '0'
        target = [int(c) for c in target_string]
        
        # alpha[t, s] is the total probability of making string equivalent to target[0:s+1] at (end of) time t and
        # ending with target[s].
        # Not that 'a--bb' makes '-a-b' but not 'a--b-' does not (must end correctly).
        # 0 <= t < probs.shape[0] =: t_max, 0 <= s < len(target) =: s_max
        t_max = probs.shape[0]
        s_max = len(target)
        
        alpha = np.zeros((t_max, s_max))
        mask = np.zeros((t_max, s_max), dtype='int') # 1 if computed that entry, 0 otherwise.

        # Initial conditions
        alpha[0, 0] = probs[0, target[0]]
        alpha[0, 1] = probs[0, target[1]]
        # alpha[0, i] = 0 for i > 1

        # alpha[t, s] = 0 for

        for i in range(1, t_max):
            alpha[i, 0] = alpha[i-1, 0] * probs[i, 0]
            for j in range(1, s_max):
                if target[j] == 0 or j < 2 or target[j] == target[j-2]:
                    alpha[i, j] = (alpha[i-1, j] + alpha[i-1, j-1]) * probs[i, target[j]]
                else:
                    alpha[i, j] = (alpha[i-1, j] + alpha[i-1, j-1] + alpha[i-1, j-2]) * probs[i, target[j]]

        return alpha[t_max-1, s_max-1] + alpha[t_max-1, s_max-2]

    def ctc_score_tf(self, probs, tones, ep=1e-10):
        """ Returns log(CTC prob), where `tones` is the target tone string, and probs[i,j] is the prob that time i is
            a j-th tone. The CTC prob is the probability that an agent following probs would output
            a string representing tones.
            Params:
            probs: probs[i,j] is the probability that time i was a j-th tone.
            ep: since we return log(CTC prob) we don't want CTC prob to be 0 or negative, so we add ep whenever
                we take logs.
        """
        target_string = '0' + '0'.join(tones) + '0'
        target = [int(c) for c in target_string]
        
        # alpha[t, s] is the total probability of making string equivalent to target[0:s+1] at (end of) time t and
        # ending with target[s].
        # Not that 'a--bb' makes '-a-b' but not 'a--b-' does not (must end correctly).
        # 0 <= t < probs.shape[0] =: t_max, 0 <= s < len(target) =: s_max
        t_max = probs.shape[0]
        s_max = len(target)

        # Scaled alpha variables.
        # alpha_sc[t, s] = alpha[t, s] / C[t] where C[t] = sum_s alpha[t, s].
        # Computing alpha_sc and C prevents floating point underflows.
        # We compute log(C[t]).
        
        alpha_sc = [[0.0] * s_max for _ in range(t_max)]
        log_c = [0.0] * t_max

        # Initial conditions (unscaled, then rescale afterwards)
        alpha_sc[0][0] = probs[0, target[0]]
        alpha_sc[0][1] = probs[0, target[1]]
        # alpha[0, i] = 0 for i > 1

        s = sum(alpha_sc[0])
        log_c[0] = tf.math.log(s + ep)
        
        # Rescale
        alpha_sc[0][0] = alpha_sc[0][0] / (s + ep)
        alpha_sc[0][1] = alpha_sc[0][1] / (s + ep)

        # alpha[t, s] = 0 for

        for i in range(1, t_max):
            # alpha_sc[i, s] will first compute alpha[i, s] / C[i-1]. Then we'll rescale at the end.
            alpha_sc[i][0] = alpha_sc[i-1][0] * probs[i, 0]
            for j in range(1, s_max):
                if target[j] == 0 or j < 2 or target[j] == target[j-2]:
                    alpha_sc[i][j] = (alpha_sc[i-1][j] + alpha_sc[i-1][j-1]) * probs[i, target[j]]
                else:
                    alpha_sc[i][j] = (alpha_sc[i-1][j] + alpha_sc[i-1][j-1] + alpha_sc[i-1][j-2]) * probs[i, target[j]]

            s = sum(alpha_sc[i])
            #c[i] = s * c[i-1]
            log_c[i] = tf.math.log(s + ep) + log_c[i-1]
            # Rescale
            for j in range(s_max):
                alpha_sc[i][j] = alpha_sc[i][j] / (s + ep)

        return tf.math.log(alpha_sc[t_max-1][s_max-1] + alpha_sc[t_max-1][s_max-2] + ep) + log_c[t_max-1]

    def avg_wer_from_audios(self, xs, y_true, use_beam_search=True):
        if use_beam_search:
            y_pred = [self.beam_search_decode(x) for x in xs]
        else:
            y_pred = [self.predict_tones(x) for x in xs]
        return self.avg_word_error_rate(y_true, y_pred)
    
    def avg_word_error_rate(self, y_true, y_pred):
        return sum(self.word_error_rate(a, b) for a, b in zip(y_true, y_pred)) / len(y_true)
    
    def word_error_rate(self, y_true, y_pred):
        """ The word error rate between two strings a and b is edit_distance(a, b) / len(a). 
            y_true and y_pred are strings.
        """
        return self.edit_distance(y_true, y_pred) / len(y_true)
    
    def edit_distance(self, a, b):
        """ Lavenshtein distance (edit distance) between two strings a and b. 
            The minimum number of insertions, deletions or substitutions to get from a to b.
        """
        memo = {}
        def dist(i, j):
            # Equals edit_distance(a[i:], b[j:])
            if (i, j) in memo:
                return memo[(i, j)]
            if i == len(a):
                res = len(b)-j
            elif j == len(b):
                res = len(a)-i
            elif a[i] == b[j]:
                res = dist(i+1, j+1)
            else:
                res = 1 + min(dist(i+1, j), dist(i, j+1), dist(i+1, j+1))
            memo[(i,j)] = res
            return res
        return dist(0, 0)
                
