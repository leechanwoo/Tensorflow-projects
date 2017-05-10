# -*- coding: utf-8 -*-

import requests
import codecs
import urllib
from urllib.request import urlopen, urlretrieve, quote
from bs4 import BeautifulSoup
import os
import re

if __name__ == "__main__":
    search_query = "https://www.youtube.com/results?search_query="
    watch_query = "https://www.youtube.com/watch?v="
    search_keywords = ['moana']

    for keyword in search_keywords:
        url = search_query + keyword
        youtube_page = requests.get(url).text
        parsed_page = BeautifulSoup(youtube_page, 'html.parser')
        item_list = parsed_page.find('ol', {'class':'item-section'})
        videos = item_list.findAll('div', {'class':'yt-lockup-video'})
        video_page_list = [ watch_query+item['data-context-item-id'] for item in videos]
        video_page = requests.get(video_page_list[0]).text
        parsed_page = BeautifulSoup(video_page,'html.parser')
        print(parsed_page)


# def download_data(s_data, save_file):
#     with open(os.getcwd() + "/"+ save_file, "w") as f:
#         f.write(s_data)

# def get_filename(url):
#     # url = "http://cineaste.co.kr/bbs/board.php?bo_table=psd_caption" #korean
#     # url = "http://cineaste.co.kr/bbs/board.php?bo_table=psd_caption&sca=%EC%98%81%EB%AC%B8" #english

#     data = requests.get(url).text 
#     # print(data)
    
#     soup = BeautifulSoup(data, 'html.parser')
#     td = soup.find_all('td', {'class':'list-subject'}) 
    
 
#     print(len(td)) 
    
#     for i in range(len(td)):
#         a = td[i].find('a')
#         # print(a)
    
#         link = a['href'].encode('ascii')
#         # print(link)
#         next_page = requests.get(link).text
#         print(next_page)

#         soup_np = BeautifulSoup(next_page, 'html.parser')
     
#         for a in soup_np.find_all('a'): 
#             try:
#                 if re.match('view_cap',a['onclick']):
#                     # if onclick attribute exist, it will match for searchDB, if success will print
#                     # 살펴보니 한글,영어,통합 3가지를 지원하는 부분이 있습니다. 모두 다운 받게 되어 있습니다.
#                     # 이쪽 소스를 수정하시면 필요한 부분만 다운 받을 수 있습니다. 
#                     # 그리고 onclick으로 지원하는 smi는 다운 받을 수 있습니다.
#                     # ( zip으로 다운 받는 부분은 추가 코딩이 필요합니다. ) 
#                     filename = a['onclick'][10:-3]
#                     print(filename)
#                     download_url = "http://cineaste.co.kr/skin/board/apms-caption/view_caption.php?bo_table=psd_caption&fname="
#                     subtitle_url = download_url + filename
#                     subtitle_data = requests.get(subtitle_url).text
#                     subtitle_soup = BeautifulSoup(subtitle_data, 'html.parser')
#                     body_data = subtitle_soup.find('body').text 
#                     save_file = filename[:-4]+".txt" 
#                     download_data(body_data, save_file)
#             except:
#                 pass
 

# if __name__ == "__main__":
#     # 페이지가 1~3155 로 되어 있습니다.
#     # 시간이 오래 걸릴 수 있으니 range 값을 조정하여 smi를 다운 받으세요.  
#     # for i in range(1, 3156):
#     #     get_filename("http://cineaste.co.kr/bbs/board.php?bo_table=psd_caption&page=" + str(i))

#     # get_filename("http://cineaste.co.kr/bbs/board.php?bo_table=psd_caption&page=" + str(1))