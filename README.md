

-----------------------------------
content based image retrieval
------------------------------

Given a query image retrieve similar images from the database.


Example, this is a query image, selected randomly from the internet:

![alt-text](https://github.com/adderbyte/content_based_image_retrieval/blob/master/data_file/test2.jpg)

Example,retrieve top-10 similar objects as below using euclidean distance as similarity measure:

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
    * No need to rerun the data preprocessing part of the notebook (if you want to, you could)
    * Enjoy !


-----------------------------------

    Pre-processed Data for Running Model
------------------------------
To folder containing the preprocessed data is in the link below:

[Public Data Folder](https://yadi.sk/d/eVz5JYGK1HHxFQ)



-----------------------------------

    Dataset
------------------------------
The training was based on the caltech 101 object category data set.

[Caltech Category Objects Database](http://www.vision.caltech.edu/Image_Datasets/Caltech101/Caltech101.html#Download)


