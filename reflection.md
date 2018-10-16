# **Finding Lane Lines on the Road** 

## Project Goals



**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report

---

[//]: # (Image References)

[image1]: ./test_images_output/dbg1.png "color"
[image2]: ./test_images_output/dbg2.png "color"
[image3]: ./test_images_output/dbg3.png "color"
[image4]: ./test_images_output/dbg4.png "color"
[image5]: ./test_images_output/dbg5.png "color"
[image6]: ./test_images_output/solidWhiteRight.png "color"

---

### Reflection



### 1. Describe your pipeline. 

The pipeline is defined in the function called "lane_Detect_pipline", consisting of 10 steps explained below.

1. grayscale 
* Convert the image from color to grayscale image.

2. Gaussian blur 
* Remove noise with 3x3 patch.

3. Canny edge extractor 
* Extract edges from denoised grayscale image.
![alt text][image1]

4. Masking 
* Apply ROI mask.
![alt text][image2]

5. Hough transform 
* Extract straight lines from edge image.
![alt text][image3]

6. My_ransac
* A simple ransac to find the dominant line models as the LEFT/RIGHT lane. This mini ransac use the first order normal euquation to find the least square error solution for each line candidate. The fitness is computed by the length of each recruited line segment. The longest line model wins and gets selected to removed most of noisy lines espeically appearing near the center of the image.
* The advantage for this implementation is that it is easy to be upgraded to detect 2nd order polynomial curve, which appears in the challenge video clip.  
![alt text][image4]

7. Line segments to points conversion
* Converted list of lines into list of points so that segments of dashed lines will be connected.

8. Add endpoints
* Added endpoints to stretch the detected lines to the boudary within the ROI. This will ensure the lans starts at the bottom of the image and ends near the center of the image.
![alt text][image5]

9. Draw lines
* Draw the detect lane (in two lines) in the line_image.
* To show left/right lines, I purposely plotted them in different colors, whcih can be easily turned back to red if that is required.

10. Weighted image
* Overlay the line_image on the input images by weighted sum
![alt text][image6]



### 2. Identify potential shortcomings with your current pipeline

1. One observed problem is that each line in the images will be extracted into parallel edges. This could be solved by adding one extra step in the piepine during the ransac which chooses top two lien candidate in LEFT/RIGHT and merges them into single lines if found parallel.

2. As stated, the current implementation uses only first order equation, which would fail to find curves as appearing in the challenge video.


### 3. Suggest possible improvements to your pipeline

1. With one extra variable in the ransac, the pipeline will be robust to "turning" lanes with curvatures.

3. However, ransac on first order polynominal will be costly in search a balloned number of polynomial candidates. Cost can be lowered by perfroming two stage ransac: find left/right stright lines in the bottom of image first; then fitting 1st order polynomial with the left/right lines as the starting members.