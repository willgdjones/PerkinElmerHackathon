import tensorflow as tf
import pickle
import numpy as np
from helpers import *
import cv2


def display_clf_statistics(clf):
    epoch_steps = [i*1 for i in range(len(clf.train_cost))]

    f,a = plt.subplots(1,2,figsize=(15,5))
    a[0].set_title("Accuracy per Epoch",size=15)
    a[0].set_ylabel('Accuracy')
    a[0].set_xlabel('Epoches')
    a[0].plot(epoch_steps,clf.train_accuracy, label='train')
    a[0].plot(epoch_steps,clf.test_accuracy,label='test')
    a[0].set_xticks(epoch_steps)
    a[0].set_xticks(a[0].get_xticks()[::5])
    a[0].set_yticks(np.arange(0,1.1,0.1))
    a[0].legend(loc='lower right')

    a[1].set_title("Cost per Epoch",size=15)
    a[1].set_ylabel('Cost')
    a[1].set_xlabel('Epoches')
    a[1].plot(epoch_steps,clf.train_cost, label='train')
    a[1].plot(epoch_steps,clf.test_cost, label='test')
    a[1].set_xticks(epoch_steps)
    a[1].set_xticks(a[1].get_xticks()[::5])
    a[1].set_yticks(np.arange(0,max(clf.train_cost)+5000,5000))
    a[1].legend(loc='upper right')
    return f,a


def generate_patches(n,size, save_interval, data_directory = './patches/'):
    data = []
    random_IDs = np.random.choice(image_IDs, n)
    for (i, ID) in enumerate(random_IDs):
        print ID
        if i % save_interval == 0 and i != 0:
            pickle.dump(data, open(data_directory + 'patches_n{}_size{}.py'.format(i,size), 'wb'))
            print 'patches_n{}_size{}.py saved'.format(i,size)
        I = Image(ID, generate_training=True)
        patches = I.generate_training_patches(size)
        data += patches

        


def softmax(z):
    return np.exp(z) / np.sum(np.exp(z), 1)

def display_example_segmentations():
    f, a = plt.subplots(2,3, figsize=(15,10))
    f.suptitle("Raw images with PerkinElmer segmentations", size=15)

    ID = np.random.choice(image_IDs)
    I = Image(ID)
    
    a[0][0].set_title('Raw image: '+ ID.split('-')[0])
    a[0][0].axis('off')
    cv2.rectangle(I.image,(0,0), (300,300), float(max(I.image.flatten())), 5)
    a[0][0].imshow(I.image)
    
    a[0][1].set_title('Clean segmentations: '+ ID.split('-')[0])
    a[0][1].axis('off')
    cv2.rectangle(I.clean,(0,0), (300,300), float(max(I.clean.flatten())), 5)
    a[0][1].imshow(I.clean)

    a[0][2].set_title('Discarded segmentations: '+ ID.split('-')[0])
    a[0][2].axis('off')
    cv2.rectangle(I.discarded,(0,0), (300,300), float(max(I.discarded.flatten())), 5)
    a[0][2].imshow(I.discarded)
    
    a[1][0].set_title('Zoomed Raw image: '+ ID.split('-')[0])
    a[1][0].axis('off')
    a[1][0].imshow(I.image[0:300,0:300])
    
    a[1][1].set_title('Zoomed Clean segmentations: '+ ID.split('-')[0])
    a[1][1].axis('off')
    a[1][1].imshow(I.clean[0:300,0:300])

    a[1][2].set_title('Zoomed Discarded segmentations: '+ ID.split('-')[0])
    a[1][2].axis('off')
    a[1][2].imshow(I.discarded[0:300,0:300])
        
def display_random_border_image():
    ID = np.random.choice(image_IDs)
    I = Image(ID)
    plt.figure(figsize=(15,15))
    plt.imshow(I.borders)
    
    
def display_random_patches(patches):
    f, a = plt.subplots(3,5, figsize=(15,10))

    random_idx = np.random.choice(range(len(patches)), 15)
    for (i ,pair) in enumerate(np.array(patches)[random_idx]):
        image = pair[0]
        label = pair[1]
        a.flatten()[i].set_title(['Background', 'Between', 'Cell'][np.array(label).argmax()])
        a.flatten()[i].imshow(image, vmin=0, vmax=0.5, interpolation='none')
        
        
        
def display_zoomed_segmentations(ID):

    seg = Segmenter('oldmodels/patches31-ep97-lr0.001', ID)
    image = Image(ID)
    cv2.rectangle(image.image, (550,350), (800,600), float(max(image.image.flatten())), 4)
    cv2.rectangle(image.borders, (550,350), (800,600), float(max(image.borders.flatten())), 4)
    cv2.rectangle(seg.segmented_image, (550,350), (800,600), float(max(seg.segmented_image.flatten())), 4)
    f,a = plt.subplots(2,3,figsize=(15,10))
    f.suptitle("Segmenting Image ID: {}".format(image.ID.split('-')[0]), size=30)
    a[0][0].imshow(image.image)
    a[0][0].set_title("Raw image")
    a[0][0].axis('off')
    a[0][1].imshow(image.borders)
    a[0][1].set_title("PerkinElmer Segmentation")
    a[0][1].axis('off')
    a[0][2].imshow(seg.segmented_image)
    a[0][2].set_title("Deep Learning Segmentation")
    a[0][2].axis('off')
    a[1][0].imshow(image.image[350:600,550:800])
    a[1][0].set_title("Zoomed Raw image")
    a[1][0].axis('off')
    a[1][1].imshow(image.borders[350:600,550:800])
    a[1][1].set_title("Zoomed PerkinElmer Segmentation")
    a[1][1].axis('off')
    a[1][2].imshow(seg.segmented_image[350:600,550:800])
    a[1][2].axis('off')
    a[1][2].set_title("Zoomed Deep Learning Segmentation")

