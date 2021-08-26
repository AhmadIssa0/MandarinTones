

import re
import codecs # unicode file reading
import random
from forvo import *
from pinyin_tagger import ChineseTokenizer

# In windows use command: chcp 936
# to display Chinese characters in command prompt.

def extract_word_pinyin(text):
    """ Returns (word, pinyin) from a line of CDICT (Chinese Dictionary). """

    m = re.search("(\S+) (\S+) \[(.+)\] \/(.*)\/", text)
    if m:
        trad = m.group(1)
        simp = m.group(2)
        pinyin = m.group(3)
        meaning = m.group(4)
        return (simp, pinyin)
    else:
        return None

def word_pinyin_pairs(filename="C:/Users/Ahmad/Dropbox/Programming/ToneClassification/cedict_ts.u8.txt",
                      comment_marker='#', max_pairs=100000):
    """ Returns list of (word, pinyin) pairs from CDICT dictionary. """
    pairs = []
    with codecs.open(filename, encoding='utf-8', mode='r') as f:
        for line in f:
            if not line.startswith('#'):
                res = extract_word_pinyin(line)
                if res:
                    pairs.append(res)
            if len(pairs) >= max_pairs:
                break
    return pairs

def pinyin_to_tones(pinyin):
    return ''.join([x[-1] for x in pinyin.split(' ')])

pairs = [(word, pinyin_to_tones(pinyin)) for  word, pinyin in word_pinyin_pairs()]
print(pairs[-5:])

forvo = Forvo()
prons = forvo.get_pronunciation_info('作孽')
print(prons)

ct = ChineseTokenizer()
random.shuffle(pairs)
for word, tone in pairs[:2000]:
    if len(word) > 1 and len(word) < 5 and '不' not in word and '一' not in word and len(ct.pinyin_dict[word]) == 1:
        forvo.download_pronunciations(word, folderpath='C:/Users/Ahmad/Documents/Jupyter notebooks/tone_classification_data/data', pronunciation=ct.to_tones(word))

    
#print(prons)

#for x in prons:
#    print(x)

#forvo.download(prons[1][0], 'chinese2.mp3')


