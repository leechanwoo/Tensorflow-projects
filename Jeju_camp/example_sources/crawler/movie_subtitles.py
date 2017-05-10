# -*- coding: utf-8 -*-

import requests
import codecs
import urllib
from urllib.request import urlopen, urlretrieve, quote
from bs4 import BeautifulSoup

def get_filename():
    url = "http://cineaste.co.kr/bbs/board.php?bo_table=psd_caption" #korean
    # url = "http://cineaste.co.kr/bbs/board.php?bo_table=psd_caption&sca=%EC%98%81%EB%AC%B8" #english

    data = requests.get(url).text
    soup = BeautifulSoup(data, 'html.parser')
    td = soup.find_all('td', {'class':'list-subject'})
    #print(td)
    table_index = 5
    a = td[table_index].find('a')
    link = a['href'].encode('ascii')
    next_page = requests.get(link).text
    soup_np = BeautifulSoup(next_page, 'html.parser')
    a = soup_np.find('a', {'class':'list-group-item break-word'})
    filename = a['onclick'][10:-3]

    print("filename extracted")

    download_url = "http://cineaste.co.kr/skin/board/apms-caption/view_caption.php?bo_table=psd_caption&fname="

    subtitle_url = download_url + filename

    subtitle_data = requests.get(subtitle_url).text
    subtitle_soup = BeautifulSoup(subtitle_data, 'html.parser')
    subtitle_soup.find('body')
    print(subtitle_soup)

    save_file = filename[:-4]+".txt"
    return download_url, filename, save_file

def download_data(download_url, filename, save_file):
    urlretrieve(download_url+filename, save_file)
    

if __name__ == "__main__":

    # download_url, filename, save_file = get_filename()
    # download_data(download_url, filename, save_file)

    # with open(save_file, 'r', encoding='utf-8') as f:
    #     text = f.read()
    
    # print(text)