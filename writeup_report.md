# ** Advanced Lane Finding** 

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/figure1.png "Undistorted"
[image2]: ./test_images/test3.jpg "Road Transformed"
[image3]: ./output_images/figure2.png "Color Threshold"
[image4]: ./output_images/figure3.png "Binary Image"
[image5]: ./output_images/figure4.png " Warped Image"
[image6]: ./output_images/figure5.png "Fit Sliding"
[image7]: ./output_images/figure6.png "Fit Polynomial"
[image8]: ./output_images/figure7.png "Output"
[video1]: ./project_video_out.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in lines 7 through 35 of the file called `CameraCalibration.py`).  

The first step is to prepare "object points", corresponding to (x, y, z) coordinates of the chessboard corners in world space. It is assumed that the chessboard pattern is fixed on the (x, y) plane at z=0, such that the object points are the same for all calibration images. 
Then, the coordinates are replicated in an array called `objp`, and a copy of it is appended to another array called `objpoints` every time all chessboard corners in a test image are succesfully detected.  
Subsequently, `imgpoints` are appended with the (x, y) pixel position of each reference corner in the image plane with every successful detection.  

Arrays `objpoints` and `imgpoints` are used to compute the camera calibration and distortion coefficients:

```python
ret, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)  
```

Distortion correction was applied to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

A combination of color(Hue and Saturation channels) and gradient thresholds (Sobel X) was used to generate a binary image (thresholding steps at lines 21 through 49 in `LaneTracking.py`).  Here is an example of my output for this step showing the different components that will result in the final binary image:

![alt text][image3]

And here is the final binary image:

![alt text][image4]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp()`, which appears in lines 42 through 80 in the file `LaneTracking.py`.  The `warp()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points. The source and destination points are determined in the following manner:

```python
    src = np.float32(
        [[((img_size[0] / 6) - 10), img_size[1]],
         [(img_size[0] / 2) - 90, img_size[1] / 2 + 120],
         [(img_size[0] / 2 + 90), img_size[1] / 2 + 120],
         [(img_size[0] * 5 / 6) + 40, img_size[1]]])

    dst = np.float32(
        [[((img_size[0] / 6) - 10), img_size[1]],
         [((img_size[0] / 6) - 10), 0],
         [(img_size[0] * 5 / 6) + 40, 0],
         [(img_size[0] * 5 / 6) + 40, img_size[1]]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 203, 720      | 203, 720      | 
| 550, 480      | 203, 0        |
| 730, 480      | 1106, 0       |
| 1106, 460     | 1106, 720     |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image5]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

A sliding window algorithm (function `sliding_window_search()` in the code `LaneTracking.py`, lines 83 through 169 ) was applied to find the lane lines for the first time or when the sanity checks (curvature, parallelism, distance) were not passed, i.e. when the detection has to be reset. The number of windows was identified as another parameter to be tuned. The lane lines are ther fit to a second order polynomial:

![alt text][image6]

With polynomial coefficients for the left and right lane: 

```sh
Left:
[  1.39487809e-05  -1.89359267e-02   3.67777138e+02]

Right:
[  3.01424963e-04  -4.29438392e-01   1.16210514e+03]
```

After the first detection (or after a reset) the method for finding lines is switched to a polynomial finder (function `polynomial_search()`in the code `LaneTracking.py`, lines 172 through 227)that uses as input previous frames polynomial coefficients.

![alt text][image7]

In case that any of the sanity checks were not passed, the filtered (averaged) results of n (another parameter to be tuned, as first try n=12) previous frames is used instead until the detection quality improves.

The class `Lane()` was implemented in the code `LaneTracking.py` (lines 311 through 415) in order to keep track of the previous states, perform the sanity checks and call the line search and curvature functions.

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

It was done in the code `LaneTracking.py`, lines 238 through 248 :

```python
def curvature(height, fit):
    # Returns the radius of curvature
    # Takes picture height and fitted polynomial coefficients as input
    y, x = calc_x_fit(height, fit)
    y_eval = y[-1]
    # Fit new polynomials to x,y in world space
    fit_cr = np.polyfit(y * ym_per_pix, x * xm_per_pix, 2)
    # Calculate the new radius of curvature
    curverad = ((1 + (2 * fit_cr[0] * y_eval * ym_per_pix + fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * fit_cr[0])
    # Now our radius of curvature is in meters
    return curverad
```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 420 through 428 in my code in `LaneTracking.py` in the function `pipeline()`.  

Here is an example of my result on a test image:

![alt text][image8]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_out.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

One of the issues I faced during the project was the selection of the source and destination points for the perspective transform. I realised thad trying to go "to deep" into the perspective could cause artifacts in the warped image. The optimal depth for the transform needs to be further investigated.

The selection of an appropriate combination of thresholding methods is probably the most important feature to be considered for improving the robustness of the line detection. Futher methods need to be investigated for the pipeline was not robust enough to deal wit the challenge_video file. 

In the harder_challenge_video file the glare in certain frames caused complete "blindness". Human drivers can also be dazzled, which can lead to accidents and certainly requires countermeasures to keep driving safely (this could be partially addressed if a different technology is also included for redundancy, e.g. LIDARS).

This pipeline certainly needs more work in order to make it robuts enough for real world deployment.

