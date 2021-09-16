# FacialRecognition
From the given images folders I have created a csv file for training specifying the path of the images in 3 formats anchor(passport images), positive(image of the same person), negative(images of different person).

For testing a csv file is also created and path of some of the images are kept and a label is given specifying whether the person is same or not.

The images are trained on a siamese network using triplet loss. The forward_prediction function in the siamese net is used for prediction only.
