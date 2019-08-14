import imagehash
import skimage.measure  as ssim
from numbers import Number
import random
from keras.preprocessing import image 
from keras.applications.resnet50 import preprocess_input

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
from scipy import sum, average
from scipy.spatial.distance import directed_hausdorff



# feature map for the query image

def index_query(image_path, cnn_model):
    
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x1 = np.expand_dims(x, axis=0)
    x2 = preprocess_input(x1)
    print(x2.shape)
    query = cnn_model.predict(x2)
    print(query.shape)
    query = query.flatten()
    print(query.shape)
    imgplot = plt.imshow(img)
    plt.show()
    
    return query,img,x

# helper function to plot the topn results retrieved
def plot_results(selected_images, highlight = None):
    
    n = len(selected_images)
    
    fig = plt.figure(figsize=(20,10))
    
    for i in range(n):
        x = round(n / 5)
        a = fig.add_subplot(x,5,i+1)
        img = selected_images[i][0]
        #img = image.load_img(image_path, target_size=(224, 224))
        plt.imshow(img)
        if i == highlight:
            a.set_title("RELEVANT")
        else:
            a.set_title(str(selected_images[i][1]))
        
    plt.show()

    
def ssims(x,y):
    '''
    return similarity score
    '''
    ssimValue = ssim.compare_ssim(x,y,multichannel=True )
    #hashes=imagehash.dhash(i)-imagehash.dhash(j)

    return ssimValue
    
def simHash(x,y):
    '''
    return similarity score using Hash function
    '''
    
    hashes=imagehash.dhash(x)-imagehash.dhash(y)

    return hashes

def ssims2(x,y):
    '''
    return similarity similarity score between pairs of feature vector
    '''
    ssimValue = ssim.compare_ssim(x,y)
    #hashes=imagehash.dhash(i)-imagehash.dhash(j)

    return ssimValue
    
### similarity measure on feature space

# for the purpose of comparison, I use the Euclidean Distance between the feature vectors

def find_query_l2(query_index, indexes, topn):
    '''
    return euclidean distacnce  between pairs of feature vector
    '''
    retrieved = []

    for indx in indexes:
        search_image = indx[0]
        search_indx = indx[1]
        diff = query_index - search_indx 
        diff = diff**2
        match_score = sum(diff)
        retrieved.append((search_image, match_score))
        
    sorted_retreival = sorted(retrieved, key=lambda x: x[1])

    selected = sorted_retreival[:topn]
    return selected

# helper function to do the retreival
# inputs - the query feature vector computed from the cnn_model, 
#          the indexes image list 
#          and the number of results to retreive (topn)
# outputs - a list of tuples containing the retrieved image paths and their respective scores


## similarity measure on feature vector space

def find_query_ssim2(query_index, indexes, topn):
    '''
    input : query_index, indexes, topn.
    output: structural_similarity_index_value for top-n images
    '''
    
    retrieved = []

    for indx in indexes:
        search_image = indx[0]
        search_indx = indx[1]
       
        match_score = ssims2(query_index, search_indx)
        retrieved.append((search_image, match_score))
        
    sorted_retreival = sorted(retrieved, key=lambda x: x[1])
    
    selected = sorted_retreival[-topn:]
    selected = sorted(selected, key=lambda x: x[1], reverse=True)
    return selected


def find_query_ssim3(query_index, indexes, topn):
    '''
    output : structural_similarity_index/euclidean_distance for top-n images
    input  : query_index, indexes, topn.
    
    '''
    
    retrieved = []

    for indx in indexes:
        search_image = indx[0]
        search_indx = indx[1]
        
        score1 = ssims2(query_index, search_indx)
        diff = query_index - search_indx 
        diff = diff**2
        score2 = sum(diff)
        match_score = score1/score2
        
        retrieved.append((search_image, match_score))
        
    sorted_retreival = sorted(retrieved, key=lambda x: x[1])
    
    selected = sorted_retreival[-topn:]
    selected = sorted(selected, key=lambda x: x[1], reverse=True)
    return selected

def find_query(query_index,model, indexes, topn):
    
    '''
    input : query image vector,neural model, indexes, topn.
    output: similarity score for top-n images
    '''
    
    retrieved = []

    for indx in indexes:
        search_image = indx[0]
        search_indx = indx[1]
        diff = query_index - search_indx 
        
        diff = diff**2
        diff = np.reshape(diff,(1,2048))
        match_score =model.predict(diff.reshape(1,-1))# model.predict(diff.reshape(-1,1))
        retrieved.append((search_image, match_score))
        
    sorted_retreival = sorted(retrieved, key=lambda x: x[1])
    
    selected = sorted_retreival[-topn:]
    selected = sorted(selected, key=lambda x: x[1], reverse=True)
    return selected
  

## similarity measure on raw image input
    
def find_query_hash(query_index, indexes, topn):
    '''
    input: query image feature vector,indexed image data base, number of similar images to be retrieved
    out: similarity score for topn using differencehash
    '''
    
    retrieved = []

    for indx in indexes:
        search_image = indx[0]
        search_indx = indx[1]
        match_score = simHash(query_index, search_image)
        retrieved.append((search_image, match_score))
        
    sorted_retreival = sorted(retrieved, key=lambda x: x[1])

    selected = sorted_retreival[:topn]
    return selected

def find_query_ssim(query_index, indexes, topn):
    '''
    input: query image feature vector,indexed image data base, number of similar images to be retrieved
    out: similarity score for topn
    '''
    
    retrieved = []

    for indx in indexes:
        search_image = indx[0]
        search_indx = indx[1]
        x = image.img_to_array(search_image)
        match_score = ssims(x,query_index)
        #diff = query_index - search_indx 
        #diff = diff**2
        #match_score = sum(diff)
        retrieved.append((search_image, match_score))
        
    sorted_retreival = sorted(retrieved, key=lambda x: x[1])

    selected = sorted_retreival[-topn:]
    return selected

def find_query_l2raw(query_index, indexes, topn):
    '''
    input: query image feature vector,indexed image data base, number of similar images to be retrieved
    out: euclidean distance for topn
    '''
    retrieved = []

    for indx in indexes:
        search_image = indx[0]
        search_indx = indx[1]
        x = image.img_to_array(search_image)
        #match_score = ssims(x,query_index)
        diff = query_index - x 
        diff = diff**2
        match_score = sum(diff)
        retrieved.append((search_image, match_score))
        
    sorted_retreival = sorted(retrieved, key=lambda x: x[1])

    selected = sorted_retreival[:topn]
    return selected


def similarity_score(x,y,i,j):
    '''
    return similarity score
    '''
    ssimValue = ssim.compare_ssim(x,y,multichannel=True )
    hashes=imagehash.dhash(i)-imagehash.dhash(j)

    if hashes != 0:  # it is an integer or a float
        return ssimValue/hashes
    else:
        return ssimValue