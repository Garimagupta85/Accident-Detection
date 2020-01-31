# Accident-Detection

### Description

CCTV surveillance cameras are installed in the majority of roads,highways these days, therefore it generates millions and millions of hours of data, thus, capture a variety of real-world incidents. Road accidents are one of the most severe and fatal incidents, which disrupt the smooth flow of traffic as well leading to wastage of time and resources. Detection of accidents not only help us to save the life of victims, but also helps in reducing traffic congestion.In this, we have proposed a framework for accident detection based on Hierarchical Recurrent Neural Network. The framework localizes and identifies the presence of road accidents in the captured video. The framework contains a time-distributed model which learns both the temporal and spatial features of the video,making the framework more efficient. The proposed approach is evaluated on the data-set, built by obtaining recorded road accident videos from youtube. Results demonstrate the applicability of our approach performs, accident detection and localization effectively.

## Getting Started

<h3>Procedure</h3>

1) Run `create_dataset.py` for converting the video to images.
2) Then run `main.py` to train the model.
3) Finally, run `model.py` for testing your model.

### Code Requirements

You can install Conda for python which resolves all the dependencies for machine learning.

```
pip install requirements.txt
```


## Built With

* [Keras](https://keras.io/) - The DL framework used
* [numpy](https://numpy.org/) - Data Mainupulation
* [matplotlib](https://matplotlib.org/) - Data Visualization



## Authors

##### 1) [Garima Gupta](https://github.com/Garimagupta85)
##### 2) [Ritwik Singh](https://github.com/ritwik-singh)


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
