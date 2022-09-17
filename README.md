<h1>Fake Detection System</h1>

Table of content
1. Demo and Screenshots
2. Overview
3. Motivation 
4. Dataset
5. Directory Tree
6. Installation
7. Deployment
8. Technologiees Used
9. To Do
10. Credits

<h4> Demo and Screenshots</h4>
The Demo of Fake Detection tool is available at this link:-

  
 The screenshots of the results page for detection of fake facial image, fake image and fake video are:-
  

<h4> Overview </h4>
FDT web app is a python package toolkit and a single platform for detecting fake media. Users can upload facial image, image or video from their local system or by placing URL link of a image or a video. The tool detects it as a fake with a red bounding box or real with a green bounding box and with a confidence score and its metadata details. It is also integrated with Twitter for streaming image tweets and categorize it into real or fake, impact as an influencer or amateur, sentiments as 1(Positive), 0(Neutral) and -1(Megative), etc, and presents other statistics through pie charts. 

<h4>Motivation</h4>
To deal with the fakes in the wild and in Indian context, an open source and easy-to-use GUI is available for Fake Detection. 
<h4> Dataset </h4>
Dataset is available on this drive link: -
For Fake Face and Deepfake Detection, Dataset directory tree are as follow:
The Dataset should have two folders: train and valid. Both should contain folders named 'fake' and 'real'.

  ![image](https://user-images.githubusercontent.com/38296253/190840044-61ae334d-736c-4260-877d-2327beb1b65f.png)
                                                           
  For Image Manipulation Detection, Dataset directory tree are as follows:
  
  The Dataset should have two folders: train and valid. Both should contain folders named 'images' and 'masks'. The images folder contain manipulated images and their corresponding ground truths are inn masks folder.

  ![image](https://user-images.githubusercontent.com/38296253/190840029-4458e94b-9af3-49f8-a5c1-73c7b1dd285f.png)

  
  
  


<h4> Directory Tree </h4>

<h4> Technical Aspect </h4>
<h4> Installation  </h4>
To install dependencies required, Download this file - environment.yml
Use this command to create environment from the environment.yml file - conda env create -f environment.yml
 <h4> Deployment </h4>
<h4> Technologiees Used </h4>
<h4> To Do </h4>
<h4> Credits </h4>
