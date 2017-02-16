# -*- coding: utf-8 -*-

import requests
from bs4 import BeautifulSoup

if __name__ == "__main__":
    url = "http://cineaste.co.kr/bbs/board.php?bo_table=psd_caption"
    print type(url)
    data = requests.get(url).text
    soup = BeautifulSoup(data, 'html.parser')
    td = soup.find('td', {'class':'list-subject'})
    link = td.find('a')['href'].encode('ascii')
    print type(link)

    new_page = requests.get(link).text.encode('utf-8')
    new_soup = BeautifulSoup(new_page, 'html.parser')
    a_class = new_soup.find('a', {'class':"list-group-item break-word"})   #a class="list-group-item break-word"


    print a_class