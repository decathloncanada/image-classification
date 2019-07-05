# -*- coding: utf-8 -*-
import Augmentor
import os
import inspect
import shutil
import math

class DataAugmentor():
    
    def __init__(self, path='data/image_dataset/train', distortion=False, 
                 flip_horizontal=False, flip_vertical=False, random_crop=False,
                 random_erasing=False, rotate=False, resize=False, 
                 target_width=299, target_height=299):
        self.current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        self.parent_dir = os.path.dirname(self.current_dir)
        self.train_dir = os.path.join(self.parent_dir, path)
        self.categories = sorted(os.listdir(self.train_dir))
        self.categories_folder = [os.path.abspath(os.path.join(self.train_dir, i)) for i in self.categories]
        # options
        self.distortion=distortion
        self.flip_horizontal=flip_horizontal
        self.flip_vertical=flip_vertical
        self.random_crop=random_crop
        self.random_erasing=random_erasing
        self.rotate=rotate
        self.resize=resize
        self.target_width=target_width
        self.target_height=target_height
        # Create a pipeline for each class
        self.pipelines = {}
        for folder in self.categories_folder:
            print("Folder %s:" % (folder))
            self.pipelines[os.path.split(folder)[1]] = (Augmentor.Pipeline(folder))
            print("\n----------------------------\n")
            
        for p in self.pipelines.values():
            print("Class %s has %s samples." % (p.augmentor_images[0].class_label, len(p.augmentor_images)))
    
    def set_options(self, pipeline):   
        if self.distortion : pipeline.random_distortion(probability=1, grid_width=4, grid_height=4, magnitude=8)
        if self.flip_horizontal : pipeline.flip_left_right(probability=0.5)
        if self.flip_vertical : pipeline.flip_top_bottom(probability=0.5)
        if self.random_crop :  pipeline.crop_random(probability=0.5, percentage_area=0.75)
        if self.resize : pipeline.resize(probability=1, width=self.target_width, height=self.target_height, resample_filter="BILINEAR")
        if self.random_erasing : pipeline.random_erasing(probability=0.5, rectangle_area=0.25)
        if self.rotate : pipeline.rotate(0.5, max_left_rotation=10, max_right_rotation=10)
        
    def test_pipeline(self, class_size_approximation, test_class):
        pipeline = self.pipelines[test_class]
        self.set_options(pipeline)
        for _ in range(math.ceil(class_size_approximation/len(pipeline.augmentor_images)-1)):
            pipeline.process()
    
    def generate_images(self, class_size_approximation):
        for pipeline in self.pipelines.values():
            self.set_options(pipeline)
            for _ in range(math.ceil(class_size_approximation/len(pipeline.augmentor_images)-1)):
                pipeline.process()
    
    # Move output folder content to main folder 
    def move_outputs(self):
        for pipeline in self.pipelines.values():
            output_dir = pipeline.augmentor_images[0].output_directory
            class_label = pipeline.augmentor_images[0].class_label
            
            for image in os.listdir(pipeline.augmentor_images[0].output_directory):
                os.rename(os.path.join(output_dir, image), os.path.join(os.path.join(self.train_dir, class_label), image))
    
    # Delete output folder if present            
    def delete_outputs(self):
        for pipeline in self.pipelines.values():
            dirpath = pipeline.augmentor_images[0].output_directory
            if os.path.exists(dirpath) and os.path.isdir(dirpath):
                shutil.rmtree(dirpath)
        
if __name__ == '__main__':
    augmentor = DataAugmentor()
    #augmentor.test_pipeline(200, 'aerobatics')
    #augmentor.generate_images(200)
    #augmentor.move_outputs()
    #augmentor.delete_outputs()
