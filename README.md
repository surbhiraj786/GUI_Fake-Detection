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
 <img src="https://user-images.githubusercontent.com/38296253/190884375-aae133aa-e387-481b-95b2-2dae3eb0f77c.PNG" width="300" height="300">
 <img src="https://user-images.githubusercontent.com/38296253/190883469-e41b98e3-e4fb-4d99-bffb-4804358a0b67.PNG" width="300" height="300">
 <img src="https://user-images.githubusercontent.com/38296253/190886226-b6f547bc-2798-4ccd-b218-fcf3dbc31936.PNG" width="300" height="300">
 <img src="https://user-images.githubusercontent.com/38296253/190886193-030e942a-afe4-4d45-a989-849e62ae2c54.PNG" width="300" height="300">
 <img src="https://user-images.githubusercontent.com/38296253/190886273-39d71438-8493-4122-87ad-f20524953696.PNG" width="300" height="300">
 <img src="https://user-images.githubusercontent.com/38296253/190886297-93abce95-139a-4bc2-99dc-441065927a06.PNG" width="300" height="300">

<h4> Ove![ResultFakeFace](https://user-images.githubusercontent.com/38296253/190886193-030e942a-afe4-4d45-a989-849e62ae2c54.PNG)
rview </h4>
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
1. For Image Manipulation Detection, Trained U-Net Model on Defacto Dataset that comprises approx 30K manipulated images (created using copy-move, splicing, inpainting, etc) with their corresponding ground truths.
2. For Fake Face Detection, Trained ResNet50+Vit model on Dataset that comprises of approx 134K fake and approx 125K real faces in both Indian and Non-Indian context.
3. For Deepfake Detection, Trained EfficientNet B7 model on FaceForensics++ dataset.  
<h4> Installation  </h4>
The Code is written in Python 3.7. Also, Download and install Anaconda follow steps given in this link:<br>
      https://docs.anaconda.com/anaconda/install/ <br>
To install dependencies required, Download this file - environment.yml <br>
Use this command to create environment from the environment.yml file- <br>conda env create -f environment.yml

 <h4> Deployment </h4>
 Step 1- conda activate env<br>
 To run this web app, Go to app folder and run python main_file.py . It will give localhost address -  http://127.0.0.1:5001/ where the tool is hosted at this address.
 Example is shown as: 
 <img src="https://user-images.githubusercontent.com/38296253/190883776-acd3512d-cb37-431b-9195-7b527a77b64a.PNG" width="300" height="200">

 
<h4> Technologiees Used </h4>
<img src="https://user-images.githubusercontent.com/38296253/190848411-b39b8984-58fb-4b8d-b193-e2afe43f8b57.png" width="100">
<img src="https://user-images.githubusercontent.com/38296253/190848468-b376733f-8cd1-4d16-91f6-7e553841dba1.png" width="100">

 

<h4> To Do </h4>
<h4> Credits </h4>
