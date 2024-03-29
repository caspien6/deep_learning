import pandas as pd
import urllib.request
import uuid
from PIL import Image
import requests
from io import BytesIO
import glob
import numpy as np


image_id_rotation_csv = 'O:\\ProgrammingSoftwares\\anaconda_projects\\dp_nagyhazi\\data\\image_ids_and_rotation.csv'
train_annotation_csv = 'O:\\ProgrammingSoftwares\\anaconda_projects\\dp_nagyhazi\\data\\train-annotations-human-imagelabels.csv'
validation_annotation_csv = 'O:\\ProgrammingSoftwares\\anaconda_projects\\dp_nagyhazi\\data\\validation-annotations-human-imagelabels.csv'
test_annotation_csv = 'O:\\ProgrammingSoftwares\\anaconda_projects\\dp_nagyhazi\\data\\test-annotations-human-imagelabels.csv'
class_desc_csv = 'O:\\ProgrammingSoftwares\\anaconda_projects\\dp_nagyhazi\\data\\class-descriptions.csv'

class DataCollector:
    """ Collect Google Open Api v4 images.
    """
    
    def load_datas(self, image_id_url_csv, train_annotation_csv, validation_annotation_csv, test_annotation_csv, class_desc_csv):
        """Loading into memory already downloaded .csv files.

		Data collector requires 5 .csv files from Google open image v4 page 
		(https://storage.googleapis.com/openimages/web/download.html). This files
		contains label and url information about images.

		
		#Arguments
			image_id_url_csv: filepath to the downloaded .csv-s.  
				Complete Open Images section  Image IDs.
			
			train_annotation_csv: filepath to the downloaded .csv-s. 
				Subset with Image-Level Labels the Human verified train label
			
			validation_annotation_csv: filepath to the downloaded .csv-s. 
				Subset with Image-Level Labels the Human verified validation label
			
			test_annotation_csv: filepath to the downloaded .csv-s. 
				Subset with Image-Level Labels the Human verified train labels
			
			class_desc_csv: filepath to the downloaded .csv-s. 
				Subset with Image-Level Labels, metadata class names.
        """

        self.image_id_url = pd.read_csv(image_id_url_csv,engine='python', sep = ',')

        self.image_train_labels = pd.read_csv(train_annotation_csv,engine='python', sep = ',')
        self.image_validation_labels = pd.read_csv(validation_annotation_csv, engine='python',sep = ',')
        self.image_test_labels = pd.read_csv(test_annotation_csv,engine='python',sep = ',')

        self.class_description = pd.read_csv(class_desc_csv,engine='python',header = None, sep = ',')

        self.image_labels = pd.concat( [self.image_train_labels,self.image_validation_labels, self.image_test_labels] )
    
    def find_by_labelName(self,label_name):
    	"""It finds the pictures wich labels equals with the argument label_name's ID.
	    	#Argument:
	    		label_name:
	    			String type word which contained by class_description.
    	"""
        label_id = self.class_description[self.class_description[1] == label_name][0]
        label_id = label_id.values[0]
        
        self.labeled_df = self.image_labels.loc[(self.image_labels['LabelName'] == label_id) & (self.image_labels['Confidence'] == 1)]
        merged_df = self.image_id_url.merge(self.labeled_df, on='ImageID', how='inner')
        
        self.result_label_df = merged_df.loc[pd.notnull(merged_df['Thumbnail300KURL'])]
        return self.result_label_df
        
    
    
    def collect_small_images(self,image_urls,folder):
        """Long running iteration!!
			    Iterate through row-by-row, download and save image. If the Http header Content-Type == image/png
			    then it won't download the image. This is because blank image comes with "Image no longer exists", HTTP status code 200.
	       	#Arguments: 
	            - image_urls: Pandas DataFrame cleaned url links, skip if url link is None.
	            - folder: The folder where do you want to download the images.
	    """
        for index, row in image_urls.iterrows():

            if not pd.notnull(row['Thumbnail300KURL']):
                return
            url = row['Thumbnail300KURL']
            filename = folder + str(uuid.uuid1())+ '.jpg'
            # Checking png is important, because there are pictures which is no longer exist, therefore im getting back a picture
            # with http status code 200 but it is a blank pic.
            if 'image/png' not in urllib.request.urlopen(url).info()['Content-Type']:
                urllib.request.urlretrieve(url, filename)