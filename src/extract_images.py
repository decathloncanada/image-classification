# -*- coding: utf-8 -*-
"""
Class to extract google images for a list of search terms and store it in a
folder, the name of the folder being the search term itself.

HT to: https://gist.github.com/genekogan/ebd77196e4bf0705db51f86431099e57

@author: AI team
"""
import inspect
import json
import os
import pandas as pd
from selenium import webdriver
from urllib.request import Request, urlopen

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)

class extract_images():
      
    #function to extract images for a given search term
    #function to extract images for a given search term
    def _get_images(self,search_term, nb_images, verbose, browser, header):
        url = "https://www.google.co.in/search?q="+search_term+"&source=lnms&tbm=isch"
        browser.get(url)
        
        #initialize counters
        counter = 0
        success = 0
        
        #we roll through the window, to be able to extract more than 100 images if desired
        for _ in range(500):
            browser.execute_script("window.scrollBy(0,10000)")
         
        #we loop through the images
        for x in browser.find_elements_by_xpath('//div[contains(@class,"rg_meta")]'):
            counter = counter + 1
            
            if verbose:
                print("Total Count:", counter)
                print("Succsessful Count:", success)
                print("URL:",json.loads(x.get_attribute('innerHTML'))["ou"])
        
            img = json.loads(x.get_attribute('innerHTML'))["ou"]
            imgtype = json.loads(x.get_attribute('innerHTML'))["ity"]
            try:
                req = Request(img)
                raw_img = urlopen(req, timeout=10).read()
                File = open(os.path.join(parentdir + '/data/image_dataset/train/' + search_term, search_term + "_" + str(counter) + "." + imgtype), "wb")
                File.write(raw_img)
                File.close()
                success = success + 1
            except:
                    pass
                
            if success >= nb_images:
                break
            
        print(success, "pictures succesfully downloaded")
        browser.close() 
        
    #main function: runs through search terms, and extract images 
    def run(self, search_terms, nb_images=100, delete_previous_images=True, 
            path_to_driver=None, verbose=True):
        """
        search_terms: list of terms we want to search
        nb_images: number of images we want to extract
        delete_previous_images: if we want to empty the folder before the search, to avoid duplicate images
        path_to_driver: path to the chrome driver; if None, we assume it is in the root folder
        """
        
        #we launch Chrome using selenium
        #if the path of the chrome driver is provided, we use it; otherwise, we assume it is in the root folder
        if path_to_driver is not None:
            browser = webdriver.Chrome(path_to_driver)
        else:
            browser = webdriver.Chrome(parentdir + '/chromedriver.exe') 
            
        header={'User-Agent':"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36"}
        
        for search_term in search_terms:
            folder = parentdir + '/data/image_dataset/train/' + search_term
            if os.path.exists(folder):
                #if we want to delete previously extracted images, to avoid duplicates
                if delete_previous_images:
                    print('Deleting previous images')
                    for file in os.listdir(folder):
                        try:
                            file_path = os.path.join(folder, file)
                            os.unlink(file_path) 
                        except:
                            print('Cannot delete file', '/data/image_dataset/train/' + search_term)
            #if no folder with the name = search term exists, create it        
            else:
                os.mkdir(parentdir + '/data/image_dataset/train/' + search_term)
                
            #extract the images
            self._get_images(search_term, nb_images=nb_images, verbose=verbose, browser=browser, header=header)                   
            

if __name__ == '__main__':
    #extract search terms
#    search_terms = list(pd.read_csv(parentdir + '/data/searchterms.csv', delimiter=',').columns)
    search_terms = ['soccer socks']
    extracter = extract_images()
    extracter.run(search_terms, nb_images=120, verbose=False)
    