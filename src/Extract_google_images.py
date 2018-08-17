# -*- coding: utf-8 -*-
"""
Spyder Editor

Attempt to scrape google image to extract images to train a CNN to recognize
a sport's good

Script imported from https://gist.github.com/genekogan/ebd77196e4bf0705db51f86431099e57

This is a temporary script file.
"""
from selenium import webdriver
import json
import os
from urllib.request import Request, urlopen

searchterm = 'hockey skates' # will also be the name of the folder
number_of_images = 120

url = "https://www.google.co.in/search?q="+searchterm+"&source=lnms&tbm=isch"
# NEED TO DOWNLOAD CHROMEDRIVER, insert path to chromedriver inside parentheses in following line
browser = webdriver.Chrome('C:\\Users\\sam\\Chromedriver\\chromedriver.exe')
browser.get(url)
header={'User-Agent':"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36"}
counter = 0
succounter = 0

if not os.path.exists(searchterm):
    os.mkdir(searchterm)

for _ in range(500):
    browser.execute_script("window.scrollBy(0,10000)")

for x in browser.find_elements_by_xpath('//div[contains(@class,"rg_meta")]'):
    counter = counter + 1
    print("Total Count:", counter)
    print("Succsessful Count:", succounter)
    print("URL:",json.loads(x.get_attribute('innerHTML'))["ou"])

    img = json.loads(x.get_attribute('innerHTML'))["ou"]
    imgtype = json.loads(x.get_attribute('innerHTML'))["ity"]
    try:
        req = Request(img)
        raw_img = urlopen(req, timeout=10).read()
        File = open(os.path.join(searchterm , searchterm + "_" + str(counter) + "." + imgtype), "wb")
        File.write(raw_img)
        File.close()
        succounter = succounter + 1
    except:
            print("can't get img")
    if succounter > number_of_images:
        break
    
print(succounter, "pictures succesfully downloaded")
browser.close()