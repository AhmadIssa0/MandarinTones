
import codecs
import re
from collections import defaultdict

class ChineseTokenizer:

    def __init__(self, cedict_filename='cedict_ts.u8.txt'):
        self.pinyin_dict = self.load_cedict(cedict_filename)


    def extract_word_pinyin(self, text):
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
    
    def load_cedict(self, filename):
        pinyin_dict = defaultdict(list)
        with codecs.open(filename, encoding='utf-8', mode='r') as f:
            for line in f:
                if not line.startswith('#'):
                    res = self.extract_word_pinyin(line)
                    if res:
                        word, pinyin = res
                        if pinyin not in pinyin_dict[word]:
                            pinyin_dict[word].append(pinyin)
        return pinyin_dict

    def tokenize(self, text, max_word_len=4):
        """ Tokenize `text` into words. Follows greedy algorithm which tries to maximize
            length of word which is contained in CDICT. Ignores punctuation.
        """
        tokens = []
        i = 0
        while i < len(text):
            word_len = max_word_len
            while word_len > 0:
                word = text[i:i+word_len]
                if len(self.pinyin_dict[word]) > 0:
                    break
                word_len -= 1
            if word_len == 0:
                # Didn't find character/word in CEDICT. Treat it as a single character.
                word_len = 1
            tokens.append(word)
            i += word_len
        return tokens
            
    def to_pinyin(self, text, all_poss=False, include_char=False):
        pinyin_poss = [(w, self.pinyin_dict[w]) for w in self.tokenize(text)]
        if all_poss:
            return pinyin_poss
        else:
            if include_char:
                pinyin = [(p[0], p[1][0]) for p in pinyin_poss if len(p[1]) > 0] # the first pronunciation is most likely.
                return pinyin
            else:
                pinyin = [p[1][0] for p in pinyin_poss if len(p[1]) > 0] # the first pronunciation is most likely.
                return ' '.join(pinyin)

    def to_tones(self, text, tone_sandhi=True, include_char=False):
        pinyin = self.to_pinyin(text, include_char=True)
        
        def sandhi(tones):
            res = tones.replace('3333', '2323')
            res = res.replace('333', '323') # could also be 223
            res = res.replace('33', '23')
            return res

        # extract tones and apply word-level sandhi
        pinyin = [[p[0], sandhi(''.join([c for c in p[1] if c in '12345']))] for p in pinyin]

        def next_tone(i, j): # i is word index, j is char in word index
            if len(pinyin[i][0]) > j+1:
                return pinyin[i][1][j+1]
            else:
                if i+1 < len(pinyin):
                    return pinyin[i+1][1][-1]
                else:
                    return ''

        def prev_char(i, j): # i is word index, j is char in word index
            if j > 0:
                return pinyin[i][0][j-1]
            else:
                if i-1 >= 0:
                    return pinyin[i-1][0][0]
                else:
                    return ''
                
        # 'bu' and 'yi' tone change rules
        for i in range(len(pinyin)-1, -1, -1):
            if i < len(pinyin)-1 and pinyin[i][0] == '不' and pinyin[i+1][1][0] == '4':
                pinyin[i][1] = '2' # change to second tone if next tone is a 4th tone

            # handle case where 'bu' is part of a word
            ind = pinyin[i][0].find('不')
            if ind >= 0 and ind < len(pinyin[i][0])-1 and pinyin[i][1][ind+1] == '4':
                pinyin[i][1] = pinyin[i][1][:ind] + '2' + pinyin[i][1][ind+1:]

            ind = pinyin[i][0].find('一')
            if ind >= 0 and not prev_char(i, ind) == '第':
                if next_tone(i, ind) == '4':
                    pinyin[i][1] = pinyin[i][1][:ind] + '2' + pinyin[i][1][ind+1:]
                elif len(next_tone(i, ind)) > 0 and next_tone(i, ind) in '123':
                    pinyin[i][1] = pinyin[i][1][:ind] + '4' + pinyin[i][1][ind+1:]
            
        
        # intra-word level sandhi
        for i in range(len(pinyin)-1,0,-1):
            if pinyin[i][1][0] == '3' and pinyin[i-1][1][-1] == '3':
                pinyin[i-1][1] = pinyin[i-1][1][:-1] + '2'
                
        if include_char:
            return pinyin
        else:
            return ''.join([p[1] for p in pinyin])
    
