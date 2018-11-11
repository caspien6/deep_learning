import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def collect_and_separate_labels(data_hl, image_root_folder, label_names):
    for query_string in label_names:
        data_hl.find_by_labelName(query_string)
        image_folder = image_root_folder + query_string.replace(" ", "") + '/'
        print(image_folder)
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)
        data_hl.collect_small_images(data_hl.result_label_df , image_folder)
        
def collect_labels(data_hl, image_folder, label_names):
    for query_string in label_names:
        data_hl.find_by_labelName(query_string)
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)
        data_hl.collect_small_images(data_hl.result_label_df , image_folder)

def save_plots(history):
    # summarize history for accuracy
    plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('model_accuracy.png')
    plt.figure()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('model_loss.png')