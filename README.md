# **Behavioral Cloning** 

## Writeup

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/center_1.png
[image2]: ./examples/left_1.png
[image3]: ./examples/right_1.png
[image4]: ./examples/center_2.png
[image5]: ./examples/left_2.png
[image6]: ./examples/right_2.png

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* readme.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The first model I tried is LeNet, which consists of two convolution layers with 5x5 filter size and two max pooling layers with 2x2 pool size.  
The code of LeNet model using Keras:
```python
def LeNet():
    """
    Build LeNet Model
    """
    model = Sequential()
    model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    model.add(Conv2D(6, (5, 5), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.5))
    model.add(Conv2D(6, (5, 5), activation='relu'))
    model.add(MaxPooling2D())
    # model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))

    return model
```
Here is the model architecture table:  
**LeNet Model:**  
| Layer                 |     Description                                | 
|:---------------------:|:----------------------------------------------:| 
| Input                 | (160, 320, 3) images                           | 
| Lambda                | x/255.0 - 0.5  data normalization              | 
| Cropping2D            | cropping=((70, 25), (0, 0)))                   | 
| Conv2D                | filters: 6, kernel_size: (5, 5),  'relu'       |
| MaxPooling2D          | default value, pool_size: (2, 2)               |
| Dropout               | 0.5                                            |
| Conv2D                | filters: 6, kernel_size: (5, 5),  'relu'       |
| MaxPooling2D          | default value, pool_size: (2, 2)               |
|                       |                                                |
| Flatten               | ()                                             |
| Dense                 | Output = 120                                   |
| Dense                 | Output = 84                                    |
| Dense                 | Output = 1                                     |


#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 65). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 24). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 118).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road by using multiple cameras and one lap of recovery driving from the sides.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model similar to the LeNet model.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. 

To combat the overfitting, I decreased the epochs to 2.

Then I defined the `generator()` method which behaves like an iterator, to improve the training performance by using lower memory.

I changed the deep learning model from LeNet to NVIDIA to improve the decrease the loss.

The final step was to run the simulator to see how well the car was driving around track one.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture I used is NVIDIA, which consists of 5 convolution layers with 5x5 filter size and 2x2 stride, and 4 full connection layers.  
The code of NVIDIA model using Keras:
```python
def Nvidia():
    """
    Build NVIDIA Model
    """
    model = Sequential()
    model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((65, 20), (0, 0))))
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))

    # model.add(MaxPooling2D())
    model.add(Flatten())
    # model.add(Dropout(0.5))
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    return model
```
Here is the model architecture table:  
**LeNet Model:**  
| Layer                 |     Description                                          | 
|:---------------------:|:--------------------------------------------------------:| 
| Input                 | (160, 320, 3) images                                     | 
| Lambda                | x/255.0 - 0.5  data normalization                        | 
| Cropping2D            | cropping=((65, 20), (0, 0)))                             | 
| Conv2D                | filters: 24, kernel_size: (5, 5), strides: (2, 2) 'relu' |
| Conv2D                | filters: 36, kernel_size: (5, 5), strides: (2, 2) 'relu' |
| Conv2D                | filters: 48, kernel_size: (5, 5), strides: (2, 2) 'relu' |
| Conv2D                | filters: 64, kernel_size: (3, 3),  'relu'                |
| Conv2D                | filters: 64, kernel_size: (3, 3),  'relu'                |
|                       |                                                          |
| Flatten               | ()                                                       |
| Dense                 | Output = 100                                             |
| Dense                 | Output = 50                                              |
| Dense                 | Output = 10                                              |
| Dense                 | Output = 1                                               |


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded one laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center, with a correction of 0.2 to the angle. 

```python
for sample in batch_samples:
    # center, left, right
    filenames = [image_path + sample[i].split('/')[-1] for i in range(3)]
    images.extend([cv2.imread(name) for name in filenames])
    angle = float(sample[3])
    angles.extend([angle, angle + correction, angle - correction])
```

![alt text][image2]
![alt text][image3]

These images show what a recovery looks like:

![alt text][image4]
![alt text][image5]
![alt text][image6]

I randomly shuffled the data set and put 20% of the data into a validation set. 
```python
def get_samples_parameter(data_path):
    """
    Read from csv file
    """
    samples = []
    with open(data_path) as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            samples.append(line)
    train, validation = train_test_split(samples, test_size=0.2)
    return train, validation
```

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. 

Lambda layer was used to normalize the data. And Cropping2D was used to clip the images with((65, 20), (0, 0)).

I used an adam optimizer so that manually training the learning rate wasn't necessary.

Generator was used to avoid out of memory, the generator method is defined in model.py(lines 32 ~ 53). Then the parameters used in `model.fit_generator()` are defined as:
```python
    steps_per_epoch = len(train_samples * 3) / batch_size
    validation_steps = len(validation_samples * 3) / batch_size
    train_generator = generator(train_samples, image_dir, batch_size=batch_size, correction=correction)
    validation_generator = generator(validation_samples, image_dir, batch_size=batch_size, correction=correction)

    # get model
    model = eval(model)()
    model.compile(loss='mse', optimizer='adam')
    model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, validation_data=validation_generator, validation_steps=validation_steps, epochs=2, workers=1)

```

The video of testing result on NVIDIA model is `./result.mp4`.
