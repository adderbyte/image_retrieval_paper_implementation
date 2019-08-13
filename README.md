

-----------------------------------
content based image retrieval
------------------------------

Given a query image retireve similar images from the data base.


Example, this is a query image, selected randomly from the internet:

![alt-text](https://github.com/adderbyte/content_based_image_retrieval/blob/master/data_file/test2.jpg)

Example,retrieve top-10 similar objects as below using metric distance:

![alt-text](https://github.com/adderbyte/content_based_image_retrieval/blob/master/data_file/screen_top.png)

Example, retrieve top-10 similar objects using  [structural similarity index](https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf) as similarity measure. This gives better performance. Could you spot the difference ?:

![alt-text](https://github.com/adderbyte/content_based_image_retrieval/blob/master/data_file/ssims.png)

Or as use ssim divided by the suclidean distance:
![alt-text](https://github.com/adderbyte/content_based_image_retrieval/blob/master/data_file/ssims.png)


Or as use Multilayer perceptron (MLP):

![alt-text](https://github.com/adderbyte/content_based_image_retrieval/blob/master/data_file/mlp.png)

The Multilayer perceptron has not be trained properly and fine tuned. When this is done the MLP should perform
better than standard metric distance.


-----------------------------------

    Dataset
------------------------------
The training was based on the caltech 101 object category data set.

[Caltech Category Objects Database](http://www.vision.caltech.edu/Image_Datasets/Caltech101/Caltech101.html#Download)
