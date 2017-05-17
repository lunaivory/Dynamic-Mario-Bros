import numpy as np
import matplotlib.pyplot as plt
import pickle
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from tqdm import tqdm
import skimage.util as util
from skimage import feature
from skimage.filters import sobel, gaussian
from skimage.color import rgb2gray

def extract_single_feature(data):

    # masks = get_masks(data, 'depth')

    #masked_rgb_images = np.zeros((data['rgb'].shape[0], data['rgb'].shape[1]-30, data['rgb'].shape[2], 3), dtype=np.uint8)
    masked_sobel_images = np.zeros((data['rgb'].shape[0], data['rgb'].shape[1]-30, data['rgb'].shape[2], 3), dtype=np.uint8)

    for i in tqdm(range(0, data['rgb'].shape[0])):
        #segmentedUser = data['segmentation'][i]
        #mask2 = np.mean(segmentedUser, axis=2) > 150  # For depth images.
        #mask3 = np.tile(mask2, (3, 1, 1))  # For 3-channel images (rgb)
        #mask3 = mask3.transpose((1, 2, 0))
        img = data['rgb'][i]
        #img = img * mask3
        img_cropped = util.crop(img, ((5,25),(0,0),(0,0)))
        masked_rgb_images[i] = img_cropped
    

    #for i in tqdm(range(0, data['rgb'].shape[0])):
    #    segmentedUser = data['segmentation'][i]
    #    mask2 = np.mean(segmentedUser, axis=2) > 150  # For depth images.
    #    mask3 = np.tile(mask2, (3, 1, 1))  # For 3-channel images (rgb)
    #    mask3 = mask3.transpose((1, 2, 0))
    #    img = data['rgb'][i]
    #    depth = data['depth'][i]
    #    #img = gaussian(img, sigma=2.0)
    #    img = img * mask3
    #    depth = depth * mask2
    #    #print(img.shape())
    #    temp = np.asarray(img)
    #    temp_depth = np.asarray(depth)
    #    temp = util.crop(temp, ((5,25),(0,0), (0,0)))
    #    temp_depth = util.crop(temp_depth, ((5,25),(0,0)))
    #    #temp = np.asarray(img_cropped)

    #    min = temp.min()
    #    max = temp.max()
    #    temp = (temp - min)/(max-min)

    #    min = temp_depth.min()
    #    max = temp_depth.max()
    #    temp_depth = (temp_depth - min)/(max-min)
        
     #   masked_sobel_images[i,:,:,0:3] = np.asarray(temp*255.0, dtype=np.uint8)
     #   masked_sobel_images[i,:,:,3] = np.asarray(temp_depth*255.0, dtype=np.uint8)

    # data_pp = {'sobel': masked_sobel_images}# , 'gestureLabels': data['gestureLabels']}
    # pickle.dump(data_pp, open("testData_pp_sobel.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    # del masked_sobel_images
    # del data_pp

   # masked_depth_images = np.zeros((data['rgb'].shape[0], data['rgb'].shape[1], data['rgb'].shape[2]))
   # for i in tqdm(range(0, data['depth'].shape[0])):
   #     segmentedUser = data['segmentation'][i]
   #     mask2 = np.mean(segmentedUser, axis=2) > 150  # For depth images.
   #     img = data['depth'][i]
   #     img = img * mask2
   #     masked_depth_images[i] = img
   #     min = masked_depth_images[i].min()
   #     max = masked_depth_images[i].max()
   #     masked_depth_images[i] = (masked_depth_images[i] - min) / (max - min)

    data_pp = {'gray': masked_sobel_images, 'gestureLabels': data['gestureLabels']}
    pickle.dump(data_pp, open("trainData_pp_rgb4D.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)


    return 
    # return masked_grey_images, masked_sobel_images, masked_depth_images


def get_masks(data, data_type):

    if data_type == 'rgb':
        masks = np.zeros((data[data_type].shape))
        for i in range(0, data[data_type].shape[0]):
            segmentedUser = data['segmentation'][i]
            mask2 = np.mean(segmentedUser, axis=2) > 150  # For depth images.
            mask3 = np.tile(mask2, (3, 1, 1))  # For 3-channel images (rgb)
            mask3 = mask3.transpose((1, 2, 0))
            masks[i] = mask3
        return masks

    elif data_type == 'depth':
        masks = np.zeros((data[data_type].shape))
        for i in range(0, data[data_type].shape[0]):
            segmentedUser = data['segmentation'][i]
            mask2 = np.mean(segmentedUser, axis=2) > 150  # For depth images.
            masks[i] = mask2
        return masks

    return

def preprocess_images(data_training):

    print('Extracting features')
    # masked_grey_images, masked_sobel_images, masked_depth_images = extract_single_feature(data_training)

    # data_pp = [masked_grey_images, masked_sobel_images, masked_depth_images]

    return extract_single_feature(data_training)



path = '/home/fzechini/Desktop/uie2/data/a2_dataTrain.pkl'
data = pickle.load(open(path, 'rb'))

preprocess_images(data)

# data_pp = {'sobel': sobel, 'gestureLabels':data['gestureLabels']}
# pickle.dump( data_pp, open( "trainData_pp_sobel.pkl", "wb" ), protocol=pickle.HIGHEST_PROTOCOL)
#
# data_pp = {'depth': depth}
# pickle.dump( data_pp, open( "trainData_pp_depth.pkl", "wb" ), protocol=pickle.HIGHEST_PROTOCOL)


