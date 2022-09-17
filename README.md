<h1>Fake Detection System</h1>

Table of content
1. Demo and Screenshots
2. Overview
3. Motivation 
4. Dataset
5. Directory Tree
6. Technical Aspect
7. Installation Deployment
8. Technologiees Used
9. To Do
10. Credits

<h4> Demo and Screenshots</h4>
<p>The Demo of Fake Detection tool is available at this link:-

  
 The screenshots of the results page for detection of fake facial image, fake image and fake video are:-
  
 ![HomePage](https://user-images.githubusercontent.com/38296253/190548443-d57d1e6f-ef8e-4109-8cfc-0cae8d188688.PNG)
  


</p>
<h4> Overview </h4>
FDT web app is a python package toolkit and a single platform for detecting fake media. Users can upload facial image, image or video from their local system or by placing URL link of a image or a video. The tool detects it as a fake with a red bounding box or real with a green bounding box and with a confidence score and its metadata details. It is also integrated with Twitter for streaming image tweets and categorize it into real or fake, impact as an influencer or amateur, sentiments as 1(Positive), 0(Neutral) and -1(Megative), etc, and presents other statistics through pie charts. 

<h4>Motivation</h4>
To deal with the fakes in the wild and in Indian context, an open source and easy-to-use GUI is available for Fake Detection. 
<h4> Dataset </h4>
Dataset is available on this drive link: -
For Fake Face and Deepfake Detection, Dataset directory tree are as follow:
The Dataset should have two folders: train and valid. Both should contain folders named 'fake' and 'real'.

                                                         |--------|-----real-------0.jpg , 1.jpg , xx.jpg
                                                         |--------|-----fake-------1.jpg , 4.jpg , xx.jpg
                                                         |
                               |--------train-----|
   Datasets:----------|
                               |--------valid-----|
                                                         |
                                                         |--------|-----real-------10.jpg , 13.jpg , xx.jpg
                                                         |--------|-----fake-------12.jpg , 23jpg , xx.jpg
                                                         
  
  For Image Manipulation Detection, Dataset directory tree are as follows:
  
  The Dataset should have two folders: train and valid. Both should contain folders named 'images' and 'masks'. The images folder contain manipulated images and their corresponding ground truths are inn masks folder. 
  
                                                         |--------|-----images-------0.jpg , 1.jpg , xx.jpg
                                                         |--------|-----masks-------0.jpg , 1.jpg , xx.jpg
                                                         |
                               |--------train-----|
   Datasets:----------|
                               |--------valid-----|
                                                         |
                                                         |--------|-----images-------10.jpg , 13.jpg , xx.jpg
                                                         |--------|-----masks-------10.jpg , 13jpg , xx.jpg

  
  


<h4> Directory Tree </h4>

<h4> Technical Aspect </h4>
<h4> Installation Deployment </h4>
<h4> Technologiees Used </h4>
<h4> To Do </h4>
<h4> Credits </h4>
