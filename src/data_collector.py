import pandas as pd
import urllib.request
import uuid
from PIL import Image
import requests
from io import BytesIO

class DataCollector:
    

    def load_datas(self, image_id_url_csv, train_annotation_csv, validation_annotation_csv, test_annotation_csv, class_desc_csv):
        self.image_id_url = pd.read_csv(image_id_url_csv,engine='python', sep = ',')

        self.image_train_labels = pd.read_csv(train_annotation_csv,engine='python', sep = ',')
        self.image_validation_labels = pd.read_csv(validation_annotation_csv, engine='python',sep = ',')
        self.image_test_labels = pd.read_csv(test_annotation_csv,engine='python',sep = ',')

        self.class_description = pd.read_csv(class_desc_csv,engine='python',header = None, sep = ',')

        self.image_labels = pd.concat( [self.image_train_labels,self.image_validation_labels, self.image_test_labels] )
    
    def find_by_labelName(self,label_name):
        label_id = self.class_description[self.class_description[1] == label_name][0]
        label_id = label_id.values[0]
        
        labeled_df = self.image_labels.loc[(self.image_labels['LabelName'] == label_id) & (self.image_labels['Confidence'] == 1)]
        merged_df = self.image_id_url.merge(labeled_df, on='ImageID', how='inner')
        
        self.result_label_df = merged_df.loc[pd.notnull(merged_df['Thumbnail300KURL'])]
        return self.result_label_df
        
    
    '''Long running iteration!!
    Iterate through row-by-row, download and save image. If the Http header Content-Type == image/png
    then it won't download the image. This is because blank image comes with "Image no longer exists", HTTP status code 200.
        Parameters: 
            - image_urls: Pandas DataFrame cleaned url links, skip if url link is None.
            - folder: The folder where do you want to download the images.
    '''
    def collect_small_images(self,image_urls,folder):
        for index, row in image_urls.iterrows():

            if not pd.notnull(row['Thumbnail300KURL']):
                return
            url = row['Thumbnail300KURL']
            filename = folder + str(uuid.uuid1())+ '.jpg'
            # Checking png is important, because there are pictures which is no longer exist, therefore im getting back a picture
            # with http status code 200 but it is a blank pic.
            if 'image/png' not in urllib.request.urlopen(url).info()['Content-Type']:
                urllib.request.urlretrieve(url, filename)