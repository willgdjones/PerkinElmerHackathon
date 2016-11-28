import tensorflow as tf
import pickle
import numpy as np
from helpers import *




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
    f, a = plt.subplots(3,4, figsize=(15,10))
    f.suptitle("Raw images with PerkinElmer segentations", size=15)

    for (i, ID) in enumerate(np.random.choice(image_IDs, 4)):
        I = Image(ID)
        a[0][i].set_title('Raw image: '+ ID.split('-')[0])
        a[0][i].axis('off')
        a[0][i].imshow(I.image)
        a[1][i].set_title('Clean segments: '+ ID.split('-')[0])
        a[1][i].axis('off')
        a[1][i].imshow(I.clean)
        a[2][i].set_title('Discarded segments: '+ ID.split('-')[0])
        a[2][i].axis('off')
        a[2][i].imshow(I.discarded)
        
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
        a.flatten()[i].imshow(image)

