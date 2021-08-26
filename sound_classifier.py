

import librosa
import librosa.effects
import numpy as np

class SoundClassifier:

    def __init__(self):
        pass

    def info_from_filename(self, filename):
        # 'available_Female_Japan_11.mp3' will return ('available', 'Female', 'Japan', '11').
        return filename[:-4].split('_')

    def gender_from_filename(self, filename):
        return self.info_from_filename(filename)[2]

    def tones_from_filename(self, filename):
        return self.info_from_filename(filename)[1]
    
    def mfcc(self, folderpath, filename, pad=None, left_pad=None, right_pad=None, n_mfcc=100):
        """
          Librosa uses a default sampling rate of 22050. With hop_length=256, this means each column (fixed time)
          represents a time interval of 256/22050 seconds (about 10ms).
          Only 12-13 low order MFCC's are significant for capturing speech patterns.
        """
        import os
        audio, sr = librosa.load(os.path.join(folderpath, filename))
        # each column in mfcc is frame_length entries of audio.
        audio, index = librosa.effects.trim(audio, top_db=15, frame_length=256, hop_length=64)
        if left_pad is not None:
            audio = np.pad(audio, pad_width=(left_pad, 0))
        if right_pad is not None:
            audio = np.pad(audio, pad_width=(0, right_pad))
            
        if pad is not None:
            if audio.shape[0] > pad:
                audio = audio[:pad]
            else:
                audio = np.pad(audio, pad_width=(0, pad - audio.shape[0]))
        return librosa.feature.mfcc(audio, n_mfcc=n_mfcc, hop_length=256) # shape (n_mfcc, time)
        #return librosa.feature.melspectrogram(audio, n_mels=128, hop_length=256)

    def file_to_samples(self, folderpath, filename):
        mfcc = self.mfcc(folderpath, filename)
        gender = self.gender_from_filename(filename)
        mfcc_t = mfcc.transpose() # mfcc has shape (n_mfcc, t). we want snapshots in time.
        n_rows = mfcc_t.shape[0]
        x = [mfcc_t[i] for i in range(n_rows)]
        x = [x[i] + x[i+1] + x[i+2] + x[i+3] + x[i+4] + x[i+5] + x[i+6] + x[i+7] + x[i+8] + x[i+9] for i in range(n_rows-9)]
        y = [gender] * (n_rows - 9)
        return (x, y)

    def get_training(self, folder='C:/Users/Ahmad/Dropbox/Programming/ToneClassification/data', n_max=None, filter_=None, **kwargs):
        # pad: pad the audio to have length pad, can take e.g. pad = 100,000.
        # filter_: list of filenames which we don't want to load.
        from os import listdir
        import os
        #filenames = [f for f in listdir(folder) if self.gender_from_filename(f) in ['Male', 'Female']]
        filenames = [f for f in listdir(folder) if f not in filter_]
        import random
        random.shuffle(filenames)
        if n_max is not None:
            filenames = filenames[:n_max]
        xs = [self.mfcc(folder, filename, **kwargs) for filename in filenames]
        ys = [self.tones_from_filename(filename) for filename in filenames]
        return (filenames, xs, ys)
        

    def svc(self, n_max=None):
        folder = 'C:/Users/Ahmad/Dropbox/Programming/ToneClassification/data'
        from os import listdir
        
        filenames = [f for f in listdir(folder) if self.gender_from_filename(f) in ['Male', 'Female']]
        import random
        random.shuffle(filenames)
        if n_max is not None:
            filenames = filenames[:n_max]
        x_vals = []
        y_vals = []
        n = len(filenames)
        train_fnames, test_fnames = filenames[:int(n*0.8)], filenames[int(n*0.8):]
        for filename in train_fnames:
            xs, ys = self.file_to_samples(folder, filename)
            x_vals += xs
            y_vals += ys

        """
        from sklearn.model_selection import train_test_split

        x_train, x_test, y_train, y_test = train_test_split(x_vals, y_vals, test_size=0.2)
        from sklearn import svm
        clf = svm.SVC(kernel='rbf', gamma=2)
        clf.fit(x_train, y_train)

        from sklearn import metrics
        y_pred = clf.predict(x_test)
        print('Accuracy:',metrics.accuracy_score(y_test, y_pred))

        print('Random', clf.predict(x_test[:1]))
        """

        from sklearn.model_selection import train_test_split

        from sklearn import svm
        from sklearn.linear_model import LogisticRegression
        clf = svm.SVC(kernel='rbf', C=5)
        #clf = svm.SVC(kernel='poly')
        #clf = LogisticRegression(C=100)
        clf.fit(x_vals, y_vals)

        def pred(clf, filename, output=False):
            xs, ys = self.file_to_samples(folder, filename)
            if len(xs) == 0:
                return 'None'
            male = 0
            female = 0
            
            for x in xs:
                if clf.predict([x])[0] == 'Male':
                    male += 1 * abs(clf.decision_function([x])[0])
                else:
                    female += 1 * abs(clf.decision_function([x])[0])
            if output:
                print(f'female: {female:4.3f}, male: {male:4.3f}')
            
            if male > female:
                return 'Male'
            else:
                return 'Female'
            

        correct = 0; incorrect = 0
        for fname in train_fnames:
            pred_gender = pred(clf, fname)
            gender = self.gender_from_filename(fname)
            if gender != pred_gender:
                print(fname, pred_gender, gender)
                pred(clf, fname, True)
                if gender != 'None':
                    incorrect += 1
            if gender == pred_gender:
                correct += 1
        print('--------------------------Training accuracy:',correct,'correct out of',(correct+incorrect),'ratio:',correct*1.0/(correct+incorrect))

        correct = 0; incorrect = 0
        for fname in test_fnames:
            pred_gender = pred(clf, fname)
            gender = self.gender_from_filename(fname)
            if gender != pred_gender:
                print(fname, pred_gender, gender)
                pred(clf, fname, True)
                if gender != 'None':
                    incorrect += 1
            if gender == pred_gender:
                correct += 1
        print('--------------------------Test accuracy:',correct,'correct out of',(correct+incorrect),'ratio:',correct*1.0/(correct+incorrect))
        print('classes:', clf.classes_)
        #print(self.file_to_samples('data', 'available_Female_Japan_11.mp3'))

    def logistic_regression(self):
        folder = 'C:/Users/Ahmad/Dropbox/Programming/SoundClassification/data'
        from os import listdir
        filenames = [f for f in listdir(folder) if self.gender_from_filename(f) in ['Male', 'Female']]
        x_vals = []
        y_vals = []
        n = len(filenames)
        train_fnames, test_fnames = filenames[:int(n*0.8)], filenames[int(n*0.8):]
        for filename in train_fnames:
            xs, ys = self.file_to_samples(folder, filename)
            x_vals += xs
            y_vals += ys

        """
        from sklearn.model_selection import train_test_split

        x_train, x_test, y_train, y_test = train_test_split(x_vals, y_vals, test_size=0.2)
        from sklearn import svm
        clf = svm.SVC(kernel='rbf', gamma=2)
        clf.fit(x_train, y_train)

        from sklearn import metrics
        y_pred = clf.predict(x_test)
        print('Accuracy:',metrics.accuracy_score(y_test, y_pred))

        print('Random', clf.predict(x_test[:1]))
        """

        from sklearn.model_selection import train_test_split

        from sklearn import svm
        from sklearn.linear_model import LogisticRegression
        #clf = svm.SVC(kernel='rbf', C=10)
        #clf = svm.SVC(kernel='poly')
        clf = LogisticRegression(C=100)
        clf.fit(x_vals, y_vals)

        def pred(clf, filename):
            xs, ys = self.file_to_samples(folder, filename)
            if len(xs) == 0:
                return 'None'
            male = 0
            female = 0
            probs = clf.predict_proba(xs) # returns array with rows (female prob, male prob)
            import numpy as np
            avg_prob = np.mean(probs, axis=0) # mean of columns
            print(f'female: {avg_prob[0]:4.3f}, male: {avg_prob[1]:4.3f}')
            if avg_prob[0] > avg_prob[1]:
                return 'Female'
            else:
                return 'Male'
            """
            for x in xs:
                if clf.predict([x])[0] == 'Male':
                    male += 1 * abs(clf.decision_function([x])[0])
                else:
                    female += 1 * abs(clf.decision_function([x])[0])
            print('female',female,'male',male)
            if male > female:
                return 'Male'
            else:
                return 'Female'
            """

        correct = 0; incorrect = 0
        for fname in train_fnames:
            pred_gender = pred(clf, fname)
            gender = self.gender_from_filename(fname)
            if gender != pred_gender:
                print(fname, pred_gender, gender)
                if gender != 'None':
                    incorrect += 1
            if gender == pred_gender:
                correct += 1
        print('--------------------------Training accuracy:',correct,'correct out of',(correct+incorrect),'ratio:',correct*1.0/(correct+incorrect))

        correct = 0; incorrect = 0
        for fname in test_fnames:
            pred_gender = pred(clf, fname)
            gender = self.gender_from_filename(fname)
            if gender != pred_gender:
                print(fname, pred_gender, gender)
                if gender != 'None':
                    incorrect += 1
            if gender == pred_gender:
                correct += 1
        print('--------------------------Test accuracy:',correct,'correct out of',(correct+incorrect),'ratio:',correct*1.0/(correct+incorrect))
        print('classes:', clf.classes_)
        #print(self.file_to_samples('data', 'available_Female_Japan_11.mp3'))




#print("test")
#sc = SoundClassifier()
#sc.logistic_regression()
"""
ns = [300, 400, 500, 600, 700]
for n in ns:
    print('n=',n)
    sc.svc(n)
"""
#xs, ys = sc.get_training(n_max=10)

#sc.svc(None)
