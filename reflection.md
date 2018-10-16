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

Step 1. grayscale 
  * Convert the image from color to grayscale image.

Step 2. Gaussian blur 
  * Remove noise with 3x3 kernel.

Step 3. Canny edge extractor 
  * Extract edges from denoised grayscale image.
  
![alt text][image1]


Step 4. Masking 
  * Apply ROI mask.
  
![alt text][image2]


Step 5. Hough transform 
  * Extract straight lines from edge image.
  
![alt text][image3]


Step 6. Ransac
  * A simple ransac to find the dominant models as the LEFT/RIGHT lines. This ransac uses all line models deteced from Hough transform and find the line candidate via least square error solution. The fitness is computed by the length of each recruited line segment. The longest line model wins out and gets selected to removed most of noisy lines espeically appearing near the center of the image.
  * The advantage for this implementation is that it can be easy upgraded to detect polynomials, which appears in the challenge video clip.  

![alt text][image4]

Step 7. Line segments to points conversion
  * Converted list of lines into list of points so that segments of dashed lines will be connected.

Step 8. Add endpoints
  * Added end points to stretch the detected lines to the boudary within the ROI. This will ensure the lane starts at the bottom of the image and ends near the center of the image (stretched to ROI).

![alt text][image5]


Step 9. Draw lines
  * Draw the detect lane (in two lines) in the line_image.
  * To show left/right lines, I purposely plotted them in different colors, whcih can be easily turned back to red if that is required.

Step 10. Weighted image
  * Overlay the line_image on the input images by weighted sum

![alt text][image6]



### 2. Identify potential shortcomings with your current pipeline

* One observed problem is that each line in the images will be extracted into parallel edges. This could be solved by adding one extra step in the piepine during the ransac which chooses top two lien candidate in LEFT/RIGHT and merges them into single lines if found parallel.

* As stated, the current implementation uses only first order equation, which would fail to find curves as appearing in the challenge video.


### 3. Suggest possible improvements to your pipeline

* With one extra variable in the ransac, the pipeline will be robust to "turning" lanes with curvatures.

* However, ransac on first order polynominal will be costly in search a balloned number of polynomial candidates. Cost can be lowered by perfroming two stage ransac: find left/right stright lines in the bottom of image first; then fitting 1st order polynomial with the left/right lines as the starting members.
