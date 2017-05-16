import numpy as np
import pickle
import skimage.transform as st
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import  Image, ImageOps, ImageEnhance

#######################################################################
## Helper functions.
#######################################################################
def data_iterator(data, labels, batch_size, num_epochs=1, shuffle=True):
    """
    A simple data iterator for samples and labels.
    @param data: Numpy tensor where the samples are in the first dimension.
    @param labels: Numpy array.
    @param batch_size:
    @param num_epochs:
    @param shuffle: Boolean to shuffle data before partitioning the data into batches.
    """
    img_size = data.shape[1]
    crop_xy = 10
    img_cropped_size = img_size - crop_xy
    data_size = data.shape[0]
    for epoch in range(num_epochs):
        # shuffle labels and features
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_samples = data[shuffle_indices]
            shuffled_labels = labels[shuffle_indices]
        else:
            shuffled_samples = data
            shuffled_labels = labels
        for batch_idx in range(0, data_size-batch_size, batch_size):
            batch_samples = shuffled_samples[batch_idx:batch_idx + batch_size]
            # batch_samples_distorted = pre_process(batch_samples, training=True)
            batch_labels = shuffled_labels[batch_idx:batch_idx + batch_size]
            batch_samples_distorted = np.zeros((batch_samples.shape[0], batch_samples.shape[1]-crop_xy, batch_samples.shape[2]-crop_xy, batch_samples.shape[3]))
            
            for id, image in enumerate(batch_samples):
                converted_img = np.reshape(image, (img_size,img_size,3))
                converted_img = Image.fromarray(converted_img)
                row = np.random.randint(0, crop_xy)
                col = np.random.randint(0, crop_xy)
                converted_img = converted_img.crop((row,col,img_cropped_size+row,img_cropped_size+col))
                 #Whiten and Standardise data
                #converted_img = (np.asarray((converted_img), dtype=np.float32))
                #converted_img -= np.mean(converted_img, axis = (0,1))
                #converted_img /= (np.std(converted_img, axis = (0,1))+1e-8)

                if np.random.uniform()>=0.5:
                    enhancer = ImageEnhance.Brightness(converted_img)
                    converted_img = enhancer.enhance(np.random.uniform(low=0.5, high=1.5))
                if np.random.uniform()>=0.5:
                    converted_img = ImageOps.mirror(converted_img)     #np.fliplr(converted_img)
                if np.random.uniform()>=0.5:
                    enhancer = ImageEnhance.Contrast(converted_img)
                    converted_img = enhancer.enhance(np.random.uniform(low=0.5, high=1.5))
                
                #Standardise data
                converted_img = (np.asarray((converted_img), dtype=np.float32))
                converted_img -= np.mean(converted_img, axis = (0,1))
                converted_img /= (np.std(converted_img, axis = (0,1))+1e-8) 
                batch_samples_distorted[id] = converted_img #np.reshape(converted_img, (img_cropped_size, img_cropped_size, 3))
            
            # do batch normalisation
            #batch_samples_distorted -= np.mean(batch_samples_distorted)
            #batch_samples_distorted /= np.std(batch_samples_distorted)  

            yield batch_samples_distorted, batch_labels

def data_iterator_samples(data, batch_size):
    """
    A simple data iterator f or samples.
    @param data: Numpy tensor where the samples are in the first dimension.
    @param batch_size:
    @param num_epochs:
    """
    img_size = data.shape[1]
    crop_xy = 10
    img_cropped_size = img_size - crop_xy
    data_size = data.shape[0]
    for batch_idx in range(0, data_size, batch_size):
        batch_samples = data[batch_idx:batch_idx + batch_size]
        batch_samples_distorted = np.zeros((batch_samples.shape[0], batch_samples.shape[1]-crop_xy, batch_samples.shape[2]-crop_xy, batch_samples.shape[3]))
        for id, image in enumerate(batch_samples):
            row = 5
            col = 5
            # converted_img = np.reshape(image, (img_size,img_size))
            converted_img = Image.fromarray(image)
            converted_img = converted_img.crop((row,col,img_cropped_size+row,img_cropped_size+col))
            # Standardise data
            converted_img = (np.asarray((converted_img), dtype=np.float32))
            converted_img -= np.mean(converted_img, axis=(0,1))
            converted_img /= (np.std(converted_img, axis=(0,1)) + 1e-8)
            batch_samples_distorted[id] = np.reshape(converted_img, (img_cropped_size, img_cropped_size, 3))
        
        yield batch_samples_distorted

def get_data(path, validation = 0.01):
    data = pickle.load(open(path, 'rb'))
    images = data['rgb']
    #images = np.reshape(images, (-1, 90, 90, 1))
    labels = data['gestureLabels']
    size = images.shape[0]
    validation = int(size*validation)

    shuffle_indices = np.random.permutation(np.arange(size))
    images = images[shuffle_indices]
    labels = labels[shuffle_indices]

    training_data = images[validation:]
    training_labels = labels[validation:]

    validation_data = images[:validation]
    validation_labels = labels[:validation]

    return (training_data, training_labels, validation_data, validation_labels)

def get_test_data(path):
    data = pickle.load(open(path, 'rb'))
    test_data = data['rgb']

    return (test_data)
                                              

def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size):
  """Construct a queued batch of images and labels.
  Args:
    image: 3-D Tensor of [height, width, 3] of type.float32.
    label: 1-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.
  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + labels from the example queue.
  num_preprocess_threads = 16
  images, label_batch = tf.train.shuffle_batch(
    [image, label],
    batch_size=batch_size,
    num_threads=num_preprocess_threads,
    capacity=min_queue_examples + 3 * batch_size,
    min_after_dequeue=min_queue_examples)

  # Display the training images in the visualizer.
  tf.summary.image('images', images)

  return images, tf.reshape(label_batch, [batch_size])
