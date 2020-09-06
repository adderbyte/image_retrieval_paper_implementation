-----------------------------------

### content based image retrieval
------------------------------

<p> Here is the scenario: Given a query image, retrieve similar images from an image database ?</p>

 Assuming you have an image of an object but you don't know what name it is called, or what characteristics it has. Then, you might want to do an image search on Yandex, Google or Baidu to help retrieve similar images to your query image. Once retrieved, you could know everything you want about your query image based on the similar images retrieved-assuming the retrieval system works well and retrieves the correct/actual matching images.

This repository is an attempt to develop a mini image retrieval system. This has important use in medicine, industrial automation and other domains. It will be shown that Neural networks are very good at computing similarities between objects activation feature vectors  than the metric similarity measures (eg euclidean distance) we are familiar with: this fact will be important for the image retrieval task.


3 factors are important:

   * using the activation layers as feature vectors
   * Having a labelled data for training image pair (feature vectors as input and similarity score as target)
   * The target score will be derived using [structural similarity index](https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf)/ euclidean distance with some heuristics to get easy and difficult pair scores
   




<p> As an example, this is a query image, selected randomly from the internet:  </p>

![alt-text](https://github.com/adderbyte/content_based_image_retrieval/blob/master/data_file/test2.jpg)

Task: retrieve the top-10 similar objects to the random image using euclidean distance as similarity measure:

![alt-text](https://github.com/adderbyte/content_based_image_retrieval/blob/master/data_file/screen_top.png)

Example, retrieve top-10 similar objects using  [structural similarity index](https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf) as similarity measure. This gives better performance. Could you spot the difference ?:

![alt-text](https://github.com/adderbyte/content_based_image_retrieval/blob/master/data_file/ssims.png)

Or  use ssim divided by the euclidean distance. Try to retrieve top-30 images to cinfirm that this does better:
![alt-text](https://github.com/adderbyte/content_based_image_retrieval/blob/master/data_file/ssim_divided_by_euclid.png)


Or using Neural network for  similarity score prediction:

![alt-text](https://github.com/adderbyte/content_based_image_retrieval/blob/master/data_file/mlp_real.png)

The Neural model was trained  just for 5 epochs!


-----------------------------------

    How to Reproduce this result
------------------------------
    
    * Clone this repository
    * Download the preprocessed data (see link to the data folder below)
    * Place the ipython notebook `CBIR.ipynb` in same folder as the downloaded data
    * Run the ipython notebook (note: you could retrain your own neural model)
    * No need to rerun the data preprocessing part of the notebook (if you want, you could)
    * Enjoy !


-----------------------------------

    Pre-processed Data for Running Model
------------------------------
The folder containing the preprocessed data is in the link below:

[Public Data Folder](https://yadi.sk/d/eVz5JYGK1HHxFQ)



-----------------------------------

    Dataset
------------------------------
The training was based on the caltech 101 object category data set.

[Caltech Category Objects Database](http://www.vision.caltech.edu/Image_Datasets/Caltech101/Caltech101.html#Download)


