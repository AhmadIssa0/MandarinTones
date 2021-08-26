# -*- coding: utf-8 -*-

from selenium import webdriver
import base64
from bs4 import BeautifulSoup

class Forvo:

    def __init__(self):
        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_argument('--headless')
        chrome_path = "C:/Users/Ahmad/Dropbox/Programming/ToneClassification/chromedriver_win32/chromedriver.exe"
        self.driver = webdriver.Chrome(executable_path=chrome_path, options=chrome_options)

    def __del__(self):
        self.driver.stop_client()
        self.driver.close()


    def get_pronunciation_info(self, word, lang='zh'):
        """ returns a list of eg. (mp3-url, 'Male', 'Luxembourg')
            from the forvo webpage.
        """
        url = f"https://forvo.com/word/{word}/#{lang}"
        self.driver.get(url)
        content = self.driver.page_source
        soup = BeautifulSoup(content)

        audios = []
        info = []
        audio_base_url="http://audio.forvo.com/audios/mp3/"
        for y in soup.find_all('header', id=lang):
            y = y.next_sibling.next_sibling # pulls out only audio for the given language
            for z in y.find_all('li'):
                nondec = [a.get('onclick').split(',')[-3][1:-1] for a in z.find_all('span', class_='play icon-size-xl')]
                
                loc = z.find('span', class_='from')
                if loc is not None:
                    audios += [audio_base_url + base64.b64decode(x).decode() for x in nondec]
                    info.append(loc.contents[0])

        # info is a list of '(Male from Luxembourg)', extract gender and location
        def extract(s):
            tokens = s[1:-1].split(' ')
            return (tokens[0], ' '.join(tokens[2:]))

        res = list(zip(audios, map(extract, info)))

        # filter out any bad urls, i.e. user didn't upload mp3 (was ogg or something).
        return [(x[0], x[1][0], x[1][1]) for x in res if x[0].endswith('mp3')]


    def download(self, url, file_name, file_path=''):
        import urllib
        import shutil
        #url = 'http://audio.forvo.com/audios/mp3/b/l/bl_9327468_94_765206.mp3'
        #file_name = 'luxembourg.mp3'
        

        user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'
        headers={'User-Agent':user_agent,} 

        request=urllib.request.Request(url,None,headers) #The assembled request
        # Download the file from `url` and save it locally under `file_name`:
        with urllib.request.urlopen(request) as response, open(file_path + '/' + file_name, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)

    def download_pronunciations(self, word, folderpath, pronunciation='', lang='zh'):
        # folderpath shouldn't end with '/'. Must use forward slash delimiters.
        i = 0
        for (url, gender, location) in self.get_pronunciation_info(word, lang):
            filename = f'{word}_{pronunciation}_{gender}_{location}_{i}.mp3'
            self.download(url, filename, folderpath)
            i += 1
        

"""
import time
forvo = Forvo()
folder = 'C:/Users/Ahmad/Dropbox/Programming/ToneClassification/data'
with open('words.txt') as f:
    lines = f.read().splitlines()
    lines = [word for word in lines if len(word) > 5]
    for word in lines[100:120]:
        print('word is:', word)
        forvo.download_pronunciations(word, folder)
        time.sleep(5)
"""
        
"""
forvo = Forvo()
prons = forvo.get_pronunciation_info('赞同')
print(prons)

for x in prons:
    print(x)

forvo.download(prons[1][0], 'chinese2.mp3')
"""

