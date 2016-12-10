import numpy as np
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
from sklearn.cross_validation import train_test_split
from tensorflow.contrib.layers.python.layers import fully_connected, convolution2d, max_pool2d, flatten
import time
from helpers import *

def lrn(x):
    return tf.nn.lrn(x, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)


class Image():
    def __init__(self,ID, generate_training=False):
        self.ID = ID
        borders = plt.imread('png/size1080/borders/' + self.ID + '.png')
        clean = plt.imread('png/size1080/clean/' + self.ID + '.png')
        discarded = plt.imread('png/size1080/discarded/' + self.ID + '.png')
        image = plt.imread('png/size1080/images/' + self.ID + '.png')
        self.borders = borders
        self.clean = clean
        self.discarded = discarded
        self.image = image
        if generate_training:
            self.between_pixel_IDs = self.get_between_pixel_IDs()
            self.background_pixel_IDs = self.get_background_pixel_IDs()
            self.clean_pixel_IDs = self.get_clean_pixel_IDs()
            self.discarded_pixel_IDs = self.get_discarded_pixel_IDs()
        
        
    def display(self):
        f, a = plt.subplots(2,2, figsize=(15,15))
        f.suptitle(self.ID, size=20)
        a[0][0].imshow(self.image)
        a[0][0].set_title("Raw image")
        a[0][1].imshow(self.clean)
        a[0][1].set_title("Clean")
        a[1][0].imshow(self.discarded)
        a[1][0].set_title("Discard")
        a[1][1].imshow(self.borders)
        a[1][1].set_title("Borders")

    # Pixels always run for 0-1166400
    def get_pixels(self):
        number_of_pixels = len(self.image.flatten())
        assert number_of_pixels == 1166400, "Number of pixels in wrong"
        return range(len(self.image.flatten()))
    
    def get_labels(self):
        return self.borders.flatten()

    def get_neighbourhood_coords(self,pixelID_list, size):
        coords_list = []
        for pixelID in pixelID_list:
            halfsize = size / 2
            i = pixelID / 1080
            j = pixelID % 1080
            coords = ((i-halfsize,i+halfsize+1), (j-halfsize,j+halfsize+1))
            coords_list.append(coords)
        return coords_list
    
    def get_background_pixel_IDs(self):
        return [i*1080 + j for i,j in  np.column_stack(np.where(self.borders * 255 == 0))]
    
    def get_between_pixel_IDs(self):
        return [i*1080 + j for i,j in  np.column_stack(np.where(self.borders * 255 == 75))]
    def get_discarded_pixel_IDs(self):
        return [i*1080 + j for i,j in  np.column_stack(np.where(self.borders * 255 == 150))]    
    
    def get_clean_pixel_IDs(self):
        return [i*1080 + j for i,j in  np.column_stack(np.where(self.borders * 255 == 250))]
        
    
    def get_patch_from_coords(self, coord_list):
        return [self.image[coord[0][0]:coord[0][1], coord[1][0]:coord[1][1]] for coord in coord_list]
    
    def generate_training_patches(self, size):
        halfwidth = size/2
        irange = range(halfwidth, 1080-halfwidth)
        jrange = range(halfwidth, 1080-halfwidth)
        valid_pixels = [1080*i + j for i in irange for j in jrange]
        
        valid_betweens_idx = np.array([self.free_from_discarded(coord_pair) for coord_pair in self.get_neighbourhood_coords(self.between_pixel_IDs, 11)])

        r_between = np.array(self.between_pixel_IDs)[valid_betweens_idx]
        
        n = len(r_between)
        
        between_neighbourhoods = self.get_neighbourhood_coords(list(set(self.between_pixel_IDs).intersection(set(valid_pixels))), size)
        
        background_nhoods = self.get_neighbourhood_coords(list(set(self.background_pixel_IDs).intersection(set(valid_pixels))), size)
        r_background_nhoods = self.sample(background_nhoods, n)
        
        
        clean_nhoods = self.get_neighbourhood_coords(list(set(self.clean_pixel_IDs).intersection(set(valid_pixels))), size)
        r_clean_nhoods = self.sample(clean_nhoods, n)
        
        labels = [[1,0,0] for x in range(n)] + [[0,1,0] for x in range(n)] + [[0,0,1] for x in range(n)]
        patches = r_background_nhoods + between_neighbourhoods + r_clean_nhoods
        neighbourhoods = [self.image[coord[0][0]:coord[0][1], coord[1][0]:coord[1][1]] for coord in patches]
        
        return zip(neighbourhoods, labels)
    
    def free_from_discarded(self,coord_pair):
        try:
            irange = range(coord_pair[0][0], coord_pair[0][1])
            jrange = range(coord_pair[1][0], coord_pair[1][1])
            return 150 not in self.get_labels()[np.array([(i-1)*1080 + j for i in irange for j in jrange])] * 255
        except:
            pdb.set_trace()

    def sample(self,sample_list,n):
        indexes = range(len(sample_list))
        samples = []
        while len(samples) < n:
            idx = np.random.choice(indexes)
            s = sample_list[idx]
            if self.free_from_discarded(s):
                samples.append(s)

        return samples
    
    
    

    
class Segmenter():
    def __init__(self, modelID, imageID):
        self.modelID = modelID
        self.imageID = imageID
        self.segmented_image = self.segment_image(self.imageID)
        self.image = Image(self.imageID)
        
    def segment_image(self, imageID):
        I = Image(imageID)
        size = 11
        halfwidth= size/2
        irange = range(halfwidth, 1080-halfwidth)
        jrange = range(halfwidth, 1080-halfwidth)


        valid_pixels = [1080*i + j for i in irange for j in jrange]

        valid_coords = I.get_neighbourhood_coords(valid_pixels, 11)
        valid_patches = I.get_patch_from_coords(valid_coords)
        valid_labels = I.get_labels()[valid_pixels]*255


        valid_encodings = []
        for i in range(len(valid_labels)):
            if valid_labels[i] == 0 or valid_labels[i] == 150:
                valid_encodings.append([1,0,0])
            elif valid_labels[i] == 75:
                valid_encodings.append([0,1,0])
            elif valid_labels[i] == 250:
                valid_encodings.append([0,0,1])
            else:
                pdb.set_trace() 

        tf.reset_default_graph()
        with tf.Session() as sess:
            model = self.modelID
            saver = tf.train.import_meta_graph('models/{}.meta'.format(model))
            saver.restore(sess, 'models/{}'.format(model))
            y = tf.get_collection('y')[0]
            Y = tf.get_collection('Y')[0]
            X = tf.get_collection('X')[0]
            count = tf.get_collection('count')[0]
            keep_prob = tf.get_collection('keep_prob')[0]
            cost = tf.get_collection('cost')[0]
            accuracy = tf.get_collection('accuracy')[0]
            optimiser = tf.get_collection('optimiser')[0]

            batch_size = 100
            number_of_examples = len(valid_pixels)
            nbatches = number_of_examples / batch_size
            total_output = []
            total_count = []
            for b in range(0,nbatches):
                batch_patches = valid_patches[b*batch_size:(b+1)*batch_size]
                batch_patches = np.array(batch_patches).reshape(-1,11,11,1)
                batch_encodings = valid_encodings[b*batch_size:(b+1)*batch_size]
                try:
                    c, output, acc, cnt = sess.run([cost, y, accuracy, count], feed_dict={X:batch_patches, Y: batch_encodings, keep_prob:1})
                except:
                    pdb.set_trace()
                total_output.extend(output)
                total_count += cnt
#                if b % 2000 == 0:
#                    print "{:.2f}% completed".format(100*float(b)/nbatches)

        predictions = [[np.exp(x[0]) / sum(np.exp(x)), np.exp(x[1]) / sum(np.exp(x)), np.exp(x[2]) / sum(np.exp(x))] for x in total_output]


        segmented_array  = [x.argmax() for x in total_output]
        segmented_image = np.array(segmented_array).reshape(1070,1070).astype(float)
        return segmented_image
    
    def display(self):
        f, a = plt.subplots(1,3, figsize=(15,5))
        f.suptitle("Image ID: {}".format(self.imageID), size=20)
        a[0].imshow(self.image.image)
        a[0].set_title("Raw image")
        a[0].axis('off')
        a[1].imshow(self.image.borders)
        a[1].set_title("PerkinElmer Segmentation ")
        a[1].axis('off')
        a[2].imshow(self.segmented_image)
        a[2].set_title("Deep Learning Segmentation")
        a[2].axis('off')
        f.show()
        
        
class Classifier():
    def __init__(self, params):
        self.params = params
        self.test_cost = [0]
        self.train_cost = [0]
        self.test_accuracy = [0]
        self.train_accuracy = [0]
        
    def display_statistics(self):
        epoch_steps = [i*1 for i in range(len(self.train_cost))]

        f,a = plt.subplots(1,2,figsize=(15,5))
        a[0].set_title("Accuracy per Epoch",size=15)
        a[0].set_ylabel('Accuracy')
        a[0].set_xlabel('Epoches')
        a[0].plot(epoch_steps,self.train_accuracy, label='train')
        a[0].plot(epoch_steps,self.test_accuracy,label='test')
        a[0].set_xticks(epoch_steps)
        a[0].set_xticks(a[0].get_xticks()[::5])
        a[0].set_yticks(np.arange(0,1.1,0.1))
        a[0].legend(loc='lower right')

        a[1].set_title("Cost per Epoch",size=15)
        a[1].set_ylabel('Cost')
        a[1].set_xlabel('Epoches')
        a[1].plot(epoch_steps,self.train_cost, label='train')
        a[1].plot(epoch_steps,self.test_cost, label='test')
        a[1].set_xticks(epoch_steps)
        a[1].set_xticks(a[1].get_xticks()[::5])
        a[1].set_yticks(np.arange(0,max(self.train_cost)+1000,1000))
        a[1].legend(loc='upper right')
        return f,a

        
    def train(self,verbose=False):
        
        data = pickle.load(open('patches/oldpatches/patches_{}.py'.format(self.params['patches_number']),'rb'))

        images = [x[0] for x in data]
        encodings = [x[1] for x in data]

        X_train, X_test, y_train, y_test = train_test_split(images, encodings, test_size=0.2, random_state=42)


        tf.reset_default_graph()
        number_of_examples = len(y_train)
        n_batches = number_of_examples / self.params['batch_size']

        X = tf.placeholder(tf.float32, shape = [None, 11, 11, 1])
        Y = tf.placeholder(tf.float32, shape = [None, 3])
        keep_prob = tf.placeholder(tf.float32)

        conv1 = convolution2d(X, 128, [5,5], padding='SAME' )
        lrn1 = lrn(conv1)
        lrn1 = tf.nn.dropout(lrn1, keep_prob)

        conv2 = convolution2d(lrn1, 128, [5,5], padding='SAME')
        lrn2 = lrn(conv2)
        lrn2 = tf.nn.dropout(lrn2, keep_prob)

        conv3 = convolution2d(lrn2, 64, [3,3], padding='SAME')
        lrn3 = lrn(conv2)
        lrn2 = tf.nn.dropout(lrn2, keep_prob)


        flat1 = flatten(lrn3)
        fc1 = fully_connected(flat1, 1000)
        fc1 = tf.nn.dropout(fc1, keep_prob)

        fc2 = fully_connected(flat1, 1000)
        fc2 = tf.nn.dropout(fc1, keep_prob)

        y = fully_connected(fc2, 3, None)

        cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(y,Y))
        optimiser = tf.train.AdamOptimizer(self.params['learning_rate']).minimize(cost)
        results = tf.cast(tf.equal(tf.arg_max(y,1), tf.arg_max(Y,1)), tf.float32)
        count = tf.reduce_sum(results)
        accuracy = tf.reduce_mean(results)
        init = tf.initialize_all_variables()

        saver = tf.train.Saver()
        tf.add_to_collection('y', y)
        tf.add_to_collection('Y', Y)
        tf.add_to_collection('X', X)
        tf.add_to_collection('keep_prob', keep_prob)
        tf.add_to_collection('cost', cost)
        tf.add_to_collection('accuracy', accuracy)
        tf.add_to_collection('optimiser', optimiser)
        tf.add_to_collection('count', count)

        
        s = time.time()
        with tf.Session() as sess:
            sess.run(init)
            for ep in range(self.params['epochs']):
                t = time.time()
                for i in range(n_batches+1):
                    b_X_train = np.array(X_train[i*self.params['batch_size']:(i+1)*self.params['batch_size']])
                    b_X_train = b_X_train.reshape(-1,11,11,1)
                    b_y_train = y_train[i*self.params['batch_size']:(i+1)*self.params['batch_size']]
                    _, trbc, trbcount = sess.run([optimiser, cost, count], feed_dict={X:b_X_train, Y: b_y_train, keep_prob:0.5})

                if ep % self.params['display_step'] == 0:
                    
                    tc = 0
                    tcount = 0
                    for i in range((len(X_test)/self.params['batch_size'])+1):
                        b_X_test = np.array(X_test[i*self.params['batch_size']:(i+1)*self.params['batch_size']])
                        b_X_test = b_X_test.reshape(-1,11,11,1)
                        b_y_test = y_test[i*self.params['batch_size']:(i+1)*self.params['batch_size']]
                        tbc, tbcount = sess.run([cost, count], feed_dict={X:b_X_test, Y: b_y_test, keep_prob:1})
                        tc += tbc
                        tcount += tbcount
                    tacc = float(tcount) / len(X_test)

                    self.test_cost.append(tc)
                    self.test_accuracy.append(tacc)
                    
                    trc = 0
                    trcount = 0
                    for i in range(n_batches+1):
                        b_X_train = np.array(X_train[i*self.params['batch_size']:(i+1)*self.params['batch_size']])
                        b_X_train = b_X_train.reshape(-1,11,11,1)
                        b_y_train = y_train[i*self.params['batch_size']:(i+1)*self.params['batch_size']]
                        trbc, trbcount = sess.run([cost, count], feed_dict={X:b_X_train, Y: b_y_train, keep_prob:1})
                        trc += trbc
                        trcount += trbcount
                    tracc = float(trcount) / len(X_train)
                    
                    self.train_cost.append(trc)
                    self.train_accuracy.append(tracc)

                    
                    if verbose:
                        print "Ep: {}, train c: {:.1f}, train acc: {:.3f}, test c: {:.1f},  test acc: {:.3f}, secs p/e {}, {:.1f} imgs p/s".format(ep,trc,tracc, tc, tacc,(time.time() - t),(number_of_examples) / (time.time() - t))
                    
                    saver.save(sess, self.params['models_dir'] + "/patches{}-ep{}-lr{}".format(self.params['patches_number'],ep,self.params['learning_rate']))
                self.final_message = "Total training time: {} at {} per epoch".format(time.time() - s, (time.time() - s) / self.params['epochs'])        
        
    