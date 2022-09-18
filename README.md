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
 <img src="https://user-images.githubusercontent.com/38296253/190883352-9dc177b8-c802-41fa-89d7-a6b996dd5c46.PNG" width="300" height="300">
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
This tool is mainly divided into 3 tasks:
1. For Image Manipulation Detection, Trained UNet Model on Defacto Dataset that comprises approx 30K manipulated images (created using copy-move, splicing, inpainting, etc) with their corresponding ground truths.
2. For Fake Face Detection, Trained ResNet50+Vit model on Dataset that comprises of approx 134K fake and approx 125K real faces in both Indian and Non-Indian context.
3. For Deepfake Detection, Trained EfficientNet B7 model on FaceForensics++ dataset.  
<h4> Installation  </h4>
The Code is written in Python 3.7. Also, Download and install Anaconda follow steps given in this link:<br>
      https://docs.anaconda.com/anaconda/install/ <br>
To install dependencies required, Download this file - environment.yml <br>
Use this command to create environment from the environment.yml file- <br>conda env create -f environment.yml
 <h4> Deployment </h4>
 To run this web app, Go to app folder and run python main_file.py
 
<h4> Technologiees Used </h4>
 ![image](https://user-images.githubusercontent.com/38296253/190848411-b39b8984-58fb-4b8d-b193-e2afe43f8b57.png)
 ![image](https://user-images.githubusercontent.com/38296253/190848468-b376733f-8cd1-4d16-91f6-7e553841dba1.png)


<h4> To Do </h4>
<h4> Credits </h4>
