# Grayscaled pictures colorisation  

Homework for the 2018 BME VIK Deep learning course.

I choosed the IMG01 problem: Grayscaled image colorisation.
There is a sample folder with all the files my program will use up, just these samples smaller than their original sizes.
In this work i am using up the pretrained Keras VGG16 network for a part of my model.

## Prerequisites  
You can use up the requirements.txt to create your environment. (it contains a cpu tensorflow)
Jupyter notebook  for the .ipynb files.
Opencv2 for video colorization.  
NVIDIA GPU + CUDA cuDNN  

## Getting Started

### Installation

- Clone this repository  
- Install Tensorflow and dependencies from https://www.tensorflow.org/install/
- If you want to use Google Open Image v4 pictures, then first you should download those:
  - [Human train labels](https://storage.googleapis.com/openimages/2018_04/train/train-annotations-human-imagelabels.csv)  
  - [Human valid labels](https://storage.googleapis.com/openimages/2018_04/validation/validation-annotations-human-imagelabels.csv)  
  - [Human test labels](https://storage.googleapis.com/openimages/2018_04/test/test-annotations-human-imagelabels.csv)  
  - [Image Ids and urls](https://storage.googleapis.com/openimages/2018_04/image_ids_and_rotation.csv)  
  - [Class descriptions](https://storage.googleapis.com/openimages/2018_04/class-descriptions.csv)  
- If you don't want to use Google images, then just use up some of your pictures. Separate them into train, valid, test folders.
- (Optional) Install python requirements:
    `pip install -r requirements.txt`
    
### Dataset preparation, collect images

You can prepare your dataset with the help of [data_collector.py](https://github.com/caspien6/deep_learning/blob/master/src/data_collector.py) and [Data Collection.ipynb](https://github.com/caspien6/deep_learning/blob/master/src/Data%20Collection.ipynb) files.  

If you want pictures from Google Image v4 you can download by label names: 
```
#Own .py file
import data_collector
from utility_methods import collect_and_separate_labels, collect_labels

data_hl = data_collector.DataCollector()

#This is a method which load up the required files for Google image into the memory. 
#You will need at least 8-10 Gigabyte memory space to perform this action
data_hl.load_datas(image_id_url_path, train_label_path, valid_label_path, test_label_path, class_description_path)

#Sample label names, you can choose names from the class description file
label_names = ['City', 'Cityscape', 'Town']

#This method will collect all 300kb Thumbnail image which label registered from the above label names.
#If you want to separate by label names into separate folders, then use collect_and_separate_labels method not collect_labels.
#This can be time consuming method.
collect_labels(data_hl, image_train_root_folder, label_names)
```
There is a sample data collecting part of my training script: [runner_stream.py](https://github.com/caspien6/deep_learning/blob/master/src/runner_stream.py), [runner_unet.py](https://github.com/caspien6/deep_learning/blob/master/src/runner_unet.py).

In the earier stage of the project i used a custom data loader class, there is a sample how to use it in the [runner.py](https://github.com/caspien6/deep_learning/blob/master/src/runner.py)

### Training
To train one of my model, you need to run the some of the scripts: [runner_stream.py](https://github.com/caspien6/deep_learning/blob/master/src/runner_stream.py), [runner_unet.py](https://github.com/caspien6/deep_learning/blob/master/src/runner_unet.py). 

Before training you need to set some of the parameters inside the runner scripts. These are:
- pts_hull_file: Path to pts_hull_file.npy. This contains the discretized a,b pairs.
- distribution_file: Path to prior_probs.npy. This file is to customize loss function with color empirical distribution.
- image_<type>_root_folder where type can be train,valid,test: Path to the <type> folder.  

