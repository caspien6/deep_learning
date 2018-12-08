import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def collect_and_separate_labels(data_hl, image_root_folder, label_names):
    """For data collection, you can call this method with multiple labels.
    It puts seperate folder each image by label_names.

        #Arguments:
            data_hl: DataCollector instance, what already loaded up with the .csv files.
            
            image_root_folder: Path to a folder where you want 
            to save the folders separated by label names.

            label_names: List of String. The class_desc_csv must contain every element.
    """
    for query_string in label_names:
        data_hl.find_by_labelName(query_string)
        image_folder = image_root_folder + query_string.replace(" ", "") + '/'
        print(image_folder)
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)
        data_hl.collect_small_images(data_hl.result_label_df , image_folder)
        
def collect_labels(data_hl, image_folder, label_names):
    """For data collection, you can call this method with multiple labels.
    It puts every image into the same folder.

        #Arguments:
            data_hl: DataCollector instance, what already loaded up with the .csv files.
            
            image_folder: Path to a folder where you want to save the downloaded pictures.

            label_names: List of String. The class_desc_csv must contain every element.
    """
    for query_string in label_names:
        data_hl.find_by_labelName(query_string)
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)
        data_hl.collect_small_images(data_hl.result_label_df , image_folder)

def save_plots(history):
    """
        Summarized method to save training history plots.
    """

    # summarize history for accuracy
    plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.savefig('model_accuracy.png')
    plt.figure()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.savefig('model_loss.png')

def save_plots_callback(logs):
    """
        Summarized method to save training log plots.
    """

    # summarize history for accuracy
    plt.figure()
    plt.plot(logs['acc'])
    plt.plot(logs['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.savefig('model_accuracy.png')
    plt.figure()
    # summarize history for loss
    plt.plot(logs['loss'])
    plt.plot(logs['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.savefig('model_loss.png')