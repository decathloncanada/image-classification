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
import shutil
from urllib.request import Request, urlopen

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)

class extract_images():
    
    def __init__(self):
        pass
    
    #method to delete images in a folder
    def _delete_images(self, folder):
        if os.path.exists(folder):
            for the_file in os.listdir(folder):
                file_path = os.path.join(folder, the_file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path): shutil.rmtree(file_path)
                except Exception as e:
                    print(e)
      
    #method to extract images for a given search term
    def _get_images(self,search_term, nb_images, folder_name, level, verbose, browser, header):
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
            if imgtype in ['jpg', 'jpeg']:
                try:
                    req = Request(img)
                    raw_img = urlopen(req, timeout=10).read()
                    File = open(os.path.join(parentdir + '/data/image_dataset/' + level + '/' + folder_name, folder_name + "_" + str(self.success_dict[folder_name]) + "." + imgtype), "wb")
                    File.write(raw_img)
                    File.close()
                    success = success + 1
                    self.success_dict[folder_name] += 1             
                except:
                        pass
                
            if success >= nb_images:
                break
            
        print(success, "pictures succesfully downloaded")
        browser.close() 
        
    #main method: runs through search terms, and extract images 
    def run(self, nb_images=100, delete_previous_images=True, 
            path_to_driver=None, verbose=True):
        """
        search_terms: list of terms we want to search, along with the name of the search folder, if it is for training or validation, and the number of images we want to extract
        nb_images: number of images we want to extract
        delete_previous_images: if we want to empty the directory before the search, to avoid duplicate images
        path_to_driver: path to the chrome driver; if None, we assume it is in the root folder
        """
        
        #read the search terms csv
        search_terms_df = pd.read_csv(parentdir + '/data/searchterms.csv', delimiter=',', encoding='latin1')              
        search_terms = search_terms_df['search_term'].unique()
        self.success_dict = {v:0 for v in search_terms_df.category.unique()}
        
        #if we want to delete previously extracted images, to avoid duplicates
        if delete_previous_images:
            print('Deleting previous images')            
            #training set
            folder = parentdir + '/data/image_dataset/train'
            self._delete_images(folder)
            #validation set
            folder = parentdir + '/data/image_dataset/val'
            self._delete_images(folder)
            #test set
            folder = parentdir + '/data/image_dataset/test'
            self._delete_images(folder)
            #augmented set
            folder = parentdir + '/data/augmented_dataset'
            self._delete_images(folder)
                    
        #create the name of the folders
        for name in search_terms_df['category'].unique():
            folder = parentdir + '/data/image_dataset/train/' + name
            if not os.path.exists(folder):
                os.makedirs(folder)
            folder = parentdir + '/data/image_dataset/val/' + name
            if not os.path.exists(folder):
                os.makedirs(folder)
            folder = parentdir + '/data/image_dataset/test/' + name
            if not os.path.exists(folder):
                os.makedirs(folder)
        
        #extract the images
        for search_term in search_terms:           
            #we launch Chrome using selenium
            #if the path of the chrome driver is provided, we use it; otherwise, we assume it is in the root folder
            if path_to_driver is not None:
                browser = webdriver.Chrome(path_to_driver)
            else:
                browser = webdriver.Chrome(parentdir + '/chromedriver.exe') 
                
            header={'User-Agent':"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36"}
                          
            nb_images = search_terms_df.loc[search_terms_df.search_term==search_term, 'number_imgs'].values[0]
            level = search_terms_df.loc[search_terms_df.search_term==search_term, 'set'].values[0]
            folder_name = search_terms_df.loc[search_terms_df.search_term==search_term, 'category'].values[0]
            self._get_images(search_term, nb_images=nb_images, folder_name=folder_name, level=level, verbose=verbose, browser=browser, header=header)                   
            del browser


if __name__ == '__main__':
    #extract search terms
    extracter = extract_images()
#    extracter.run(verbose=True)
    