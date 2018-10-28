
# coding: utf-8

# # Self-Driving Car Engineer Nanodegree
# 
# 
# ## Project: **Finding Lane Lines on the Road** 
# ***
# In this project, you will use the tools you learned about in the lesson to identify lane lines on the road.  You can develop your pipeline on a series of individual images, and later apply the result to a video stream (really just a series of images). Check out the video clip "raw-lines-example.mp4" (also contained in this repository) to see what the output should look like after using the helper functions below. 
# 
# Once you have a result that looks roughly like "raw-lines-example.mp4", you'll need to get creative and try to average and/or extrapolate the line segments you've detected to map out the full extent of the lane lines.  You can see an example of the result you're going for in the video "P1_example.mp4".  Ultimately, you would like to draw just one line for the left side of the lane, and one for the right.
# 
# In addition to implementing code, there is a brief writeup to complete. The writeup should be completed in a separate file, which can be either a markdown file or a pdf document. There is a [write up template](https://github.com/udacity/CarND-LaneLines-P1/blob/master/writeup_template.md) that can be used to guide the writing process. Completing both the code in the Ipython notebook and the writeup template will cover all of the [rubric points](https://review.udacity.com/#!/rubrics/322/view) for this project.
# 
# ---
# Let's have a look at our first image called 'test_images/solidWhiteRight.jpg'.  Run the 2 cells below (hit Shift-Enter or the "play" button above) to display the image.
# 
# **Note: If, at any point, you encounter frozen display windows or other confounding issues, you can always start again with a clean slate by going to the "Kernel" menu above and selecting "Restart & Clear Output".**
# 
# ---

# **The tools you have are color selection, region of interest selection, grayscaling, Gaussian smoothing, Canny Edge Detection and Hough Tranform line detection.  You  are also free to explore and try other techniques that were not presented in the lesson.  Your goal is piece together a pipeline to detect the line segments in the image, then average/extrapolate them and draw them onto the image for display (as below).  Once you have a working pipeline, try it out on the video stream below.**
# 
# ---
# 
# <figure>
#  <img src="examples/line-segments-example.jpg" width="380" alt="Combined Image" />
#  <figcaption>
#  <p></p> 
#  <p style="text-align: center;"> Your output should look something like this (above) after detecting line segments using the helper functions below </p> 
#  </figcaption>
# </figure>
#  <p></p> 
# <figure>
#  <img src="examples/laneLines_thirdPass.jpg" width="380" alt="Combined Image" />
#  <figcaption>
#  <p></p> 
#  <p style="text-align: center;"> Your goal is to connect/average/extrapolate line segments to get output like this</p> 
#  </figcaption>
# </figure>

# **Run the cell below to import some packages.  If you get an `import error` for a package you've already installed, try changing your kernel (select the Kernel menu above --> Change Kernel).  Still have problems?  Try relaunching Jupyter Notebook from the terminal prompt.  Also, consult the forums for more troubleshooting tips.**  

# ## Import Packages

# In[4]:


#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Read in an Image

# In[5]:


#reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')

#printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)
plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')


# ## Ideas for Lane Detection Pipeline

# **Some OpenCV functions (beyond those introduced in the lesson) that might be useful for this project are:**
# 
# `cv2.inRange()` for color selection  
# `cv2.fillPoly()` for regions selection  
# `cv2.line()` to draw lines on an image given endpoints  
# `cv2.addWeighted()` to coadd / overlay two images
# `cv2.cvtColor()` to grayscale or change color
# `cv2.imwrite()` to output images to file  
# `cv2.bitwise_and()` to apply a mask to an image
# 
# **Check out the OpenCV documentation to learn about these and discover even more awesome functionality!**

# ## Helper Functions

# Below are some helper functions to help get you started. They should look familiar from the lesson!

# In[6]:


import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)


# ## Test Images
# 
# Build your pipeline to work on the images in the directory "test_images"  
# **You should make sure your pipeline works well on these images before you try the videos.**

# In[7]:


import os
files = os.listdir("test_images/")
print(files)


# ## Build a Lane Finding Pipeline
# 
# 

# Build the pipeline and run your solution on all test_images. Make copies into the `test_images_output` directory, and you can use the images in your writeup report.
# 
# Try tuning the various parameters, especially the low and high Canny thresholds as well as the Hough lines parameters.

# In[81]:


# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images_output directory.

def my_ransac(Lines, err_pix = 4):
    """
    Identify the dominant left/right lines
    
    'Lines', a list line segments in [x1,y1,x2,y2]
    
    return top left/right lines
    """
    # excludes extreme lines, slope = 0/inf
    Lines = Lines[(Lines[:,1] - Lines[:,3])*(Lines[:,0] - Lines[:,2]) != 0 ]
    #Lines = Lines[np.argsort(np.max(Lines[:,[1,3]], axis = 1))[::-1],:]
    
    # use line length as the weighting for fitness 
    LLen = np.linalg.norm(Lines[:,:2] - Lines[:,2:4], axis = 1)
    
    # fit all lines to find top line modele 
    fitness, inliers, model = [], [], []
    for i, L in enumerate(Lines):
        # get line model from LSE solution
        A = np.vstack((L[[0,2]], [1,1])).T
        y = L[[1,3]].reshape(2,1)
        A1 = np.linalg.inv(A)
        coef = np.matmul(A1,y).T
        y1, y2 = Lines[:,1], Lines[:,3]
        pt1 = np.vstack((Lines[:,0], np.ones_like(Lines[:,0])))
        pt2 = np.vstack((Lines[:,2], np.ones_like(Lines[:,2])))
        ii =((np.abs(np.matmul(coef, pt1) - y1) + np.abs(np.matmul(coef, pt2) - y2)) < err_pix).squeeze()
        fitness += [np.sum(LLen[ii])]
        model += [coef]
        inliers += [Lines[ii]]
    fitness = np.array(fitness)    
    model = np.array(model).squeeze().T
    ii = np.where(model[0] < 0)[0]
    Li = ii[np.argmax(fitness[ii])]
    ii = np.where(model[0] > 0)[0]
    Ri = ii[np.argmax(fitness[ii])]
    
    return inliers[Li], inliers[Ri], model[:,Li], model[:,Ri]

def line2points(line):
    # break down into points
    pts = np.vstack((line[:,:2], line[:,2:4]))
    pts = pts[np.argsort(pts[:,1]),:]
    
    # merge similar points
    gap = np.append(np.any(pts[:-1,:] - pts[1:,:] == 0, axis = 1), False)
    ii = np.where(gap)[0]
    pts[ii+1] = (pts[ii] + pts[ii+1])/2
    pts = pts[~gap]
    return pts

def my_draw_lines(line_img, pts, color, thickness=2):
    """
    Draw lines in line_img
    
    'line_img', the canvas to draw the lines
    'pts', a list of points in [x, y]
    
    return drawn line_img
    """
    for pt1, pt2 in zip(pts[:-1], pts[1:]):
        cv2.line(line_img, tuple(pt1), tuple(pt2), color, thickness)
    return line_img

def my_hough_lines(masked_image, pp):
    """
    `masked_image`, input image masked for a Canny transform.
    'pp', pipeline parameters    
    Returns lines detected in the masked_images
    """
    lines = cv2.HoughLinesP(masked_image, pp['rho'], pp['theta'], pp['threshold'], np.array([]),                minLineLength = pp['min_line_len'], maxLineGap = pp['max_line_gap'])
    return lines.squeeze()

def add_endpoints(pts, model, y0, y1):
    xFromY = lambda model, y : int((y - model[1])/model[0] + 0.5)
    if y1 > pts[0,1]:
        pts = np.vstack(([xFromY(model, y1), int(y1)], pts))
    if y0 < pts[-1,1]:
        pts = np.vstack((pts, [xFromY(model, y0), int(y0)]))
    return pts

#np.vstack(([xFromY(model, y1), y1], line, [xFromY(model, y0), y0]))

def lane_detect_pipeline(image, pp, dbug = 100):
    """
    processing pipeline to extract lanes from images
    
    color -> gray level -> Gaussian blur -> canny -> masking -> hough -> ransac -> 
        add endpoints -> draw lines -> weighted sum
    
    'image', raw input 
    'pp', pipeline parameters
    
    return "lane overlayed" image
    """
    
    # output image
    line_img = np.zeros_like(image)
    
    # edge detection
    img = canny(gaussian_blur(grayscale(image), pp['kernel_size']), 
                pp['canny']['low_threshold'], pp['canny']['high_threshold'])
    if (dbug == 0):
        return img

    # masking
    nrows, ncols, _ = image.shape
    vertices = np.array([[[0, nrows], [ncols/2 - ncols/32 , nrows/2 + nrows/8 ],                          [ncols/2 + ncols/32 , nrows/2 + nrows/8], [ncols, nrows]]], dtype = np.int32)
    masked_image = region_of_interest(img, vertices)
    if (dbug == 1):
        return masked_image
    
    # hough filtering
    lines = my_hough_lines(masked_image, pp['hough'] )
    if (dbug == 2):
        for l in lines:
            my_draw_lines(line_img, line2points(np.array([l])), [255, 0, 0])
        return line_img
    
    # ransac Left/right selection
    LL, RL, Lmodel, Rmodel = my_ransac(lines)
    
    if (dbug == 3):
        for l in LL:
            my_draw_lines(line_img, line2points(np.array([l])), [255, 0, 0])
        return line_img
    if (dbug == 4):
        for l in RL:
            my_draw_lines(line_img, line2points(np.array([l])), [255, 0, 0])
        return line_img
     
    # convert to point list and add end point    
    Lpts = add_endpoints(line2points(LL), Lmodel, nrows/2 + nrows/8, nrows) #line_img.shape[0]/2 + line_img.shape[0]/16,  lime_img.shape[0])
    Rpts = add_endpoints(line2points(RL), Rmodel, nrows/2 + nrows/8, nrows) #line_img.shape[0]/2 + line_img.shape[0]/16,  lime_img.shape[0])

    # draw broken lines
    my_draw_lines(line_img, Lpts, [255, 0, 0], pp['line_thickness'])
    my_draw_lines(line_img, Rpts, [0, 0, 255], pp['line_thickness'])
    
    # weighted output
    results = weighted_img(line_img, image)

    return results    


# In[38]:


# parameters    
pp = { 
    'kernel_size': 3,
    'line_thickness': 8,
    'canny': { 'low_threshold': 50, 'high_threshold': 150 },
    'hough': { 'rho': 1, 'theta': np.pi/180., 'threshold':  8, 'min_line_len':  4, 'max_line_gap': 4 }
}    

file_i = 'test_images/'+ files[1]
imge = mpimg.imread(file_i)
plt.imshow(lane_detect_pipeline(image, pp, 0))
print(file_i)
plt.imshow(image)


# In[39]:


plt.imshow(lane_detect_pipeline(image, pp, 0))


# In[40]:


plt.imshow(lane_detect_pipeline(image, pp, 1))


# In[51]:


plt.imshow(lane_detect_pipeline(image, pp, 2))


# In[53]:


plt.imshow(lane_detect_pipeline(image, pp, 3))


# In[54]:


plt.imshow(lane_detect_pipeline(image, pp, 4))


# In[76]:


plt.imshow(lane_detect_pipeline(image, pp, 5))


# In[77]:


plt.imshow(lane_detect_pipeline(image, pp))


# ## Test on Videos
# 
# You know what's cooler than drawing lanes over images? Drawing lanes over video!
# 
# We can test our solution on two provided videos:
# 
# `solidWhiteRight.mp4`
# 
# `solidYellowLeft.mp4`
# 
# **Note: if you get an import error when you run the next cell, try changing your kernel (select the Kernel menu above --> Change Kernel). Still have problems? Try relaunching Jupyter Notebook from the terminal prompt. Also, consult the forums for more troubleshooting tips.**
# 
# **If you get an error that looks like this:**
# ```
# NeedDownloadError: Need ffmpeg exe. 
# You can download it by calling: 
# imageio.plugins.ffmpeg.download()
# ```
# **Follow the instructions in the error message and check out [this forum post](https://discussions.udacity.com/t/project-error-of-test-on-videos/274082) for more troubleshooting tips across operating systems.**

# In[78]:


# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML


# In[79]:


pp = { 
    'kernel_size': 3,
    'line_thickness': 8,
    'canny': { 'low_threshold': 50, 'high_threshold': 150 },
    'hough': { 'rho': 1, 'theta': np.pi/180., 'threshold':  8, 'min_line_len':  4, 'max_line_gap': 4 }
}    

def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    result = lane_detect_pipeline(image, pp)
    
    return result


# Let's try the one with the solid white lane on the right first ...

# In[80]:


white_output = 'test_videos_output/solidWhiteRight.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
get_ipython().run_line_magic('time', 'white_clip.write_videofile(white_output, audio=False)')


# Play the video inline, or if you prefer find the video in your filesystem (should be in the same directory) and play it in your video player of choice.

# In[82]:


white_output = 'test_videos_output/solidWhiteRight.mp4'


# In[83]:


HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(white_output))


# ## Improve the draw_lines() function
# 
# **At this point, if you were successful with making the pipeline and tuning parameters, you probably have the Hough line segments drawn onto the road, but what about identifying the full extent of the lane and marking it clearly as in the example video (P1_example.mp4)?  Think about defining a line to run the full length of the visible lane based on the line segments you identified with the Hough Transform. As mentioned previously, try to average and/or extrapolate the line segments you've detected to map out the full extent of the lane lines. You can see an example of the result you're going for in the video "P1_example.mp4".**
# 
# **Go back and modify your draw_lines function accordingly and try re-running your pipeline. The new output should draw a single, solid line over the left lane line and a single, solid line over the right lane line. The lines should start from the bottom of the image and extend out to the top of the region of interest.**

# Now for the one with the solid yellow lane on the left. This one's more tricky!

# In[ ]:


yellow_output = 'test_videos_output/solidYellowLeft.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4').subclip(0,5)
clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4')
yellow_clip = clip2.fl_image(process_image)
get_ipython().run_line_magic('time', 'yellow_clip.write_videofile(yellow_output, audio=False)')


# In[ ]:


HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(yellow_output))


# ## Writeup and Submission
# 
# If you're satisfied with your video outputs, it's time to make the report writeup in a pdf or markdown file. Once you have this Ipython notebook ready along with the writeup, it's time to submit for review! Here is a [link](https://github.com/udacity/CarND-LaneLines-P1/blob/master/writeup_template.md) to the writeup template file.
# 

# ## Optional Challenge
# 
# Try your lane finding pipeline on the video below.  Does it still work?  Can you figure out a way to make it more robust?  If you're up for the challenge, modify your pipeline so it works with this video and submit it along with the rest of your project!

# In[ ]:


challenge_output = 'test_videos_output/challenge.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip3 = VideoFileClip('test_videos/challenge.mp4').subclip(0,5)
clip3 = VideoFileClip('test_videos/challenge.mp4').subclip(0,10)
challenge_clip = clip3.fl_image(process_image)
get_ipython().run_line_magic('time', 'challenge_clip.write_videofile(challenge_output, audio=False)')


# In[ ]:


HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(challenge_output))

