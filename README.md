# Comparisong of SIFT and CNN

This project compares using SIFT with color features and both trained and pretrained CNNs in a butterfly classification task.

## TODO / thoughts

- [ ] Improve data preprocessing when training the neural network by doing random cropping?
- [X] Make sure the tensor color dimensions are appropriate for OpenCV
- [ ] Test reducing feature vector size for pretrained imagenet

## The butterfly dataset

The hierarchy of the data as presented in order in the index files is

species(1-200), genus(1-116), subfamily(1-23), family(1-5)


## Notes about training the CNNS

All models appear to converge after 10 epochs based on accuracy obtained on development set.