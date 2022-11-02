import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torchvision
import tarfile
from torchvision.datasets.utils import download_url
from torch.utils.data import random_split
import numpy as np
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, ToPILImage
import torchvision.transforms as tt
import PIL
from imgaug import augmenters as iaa
import imgaug as ia
import cv2 as cv
import math
import argparse
from facenet_pytorch.models.mtcnn import MTCNN
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from efficientnet_pytorch import EfficientNet
import torch.nn as nn
import torch.nn.functional as F
from torch import functional as Func
import PIL
from imgaug import augmenters as iaa
import cv2
import glob
import re
import moviepy.editor as moviepy
from facenet_pytorch.models.mtcnn import MTCNN
from tqdm.notebook import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from fastbook import *
from fastai.vision import *
from fastai import *
import pathlib
from fastai.vision.all import *
from torchvision import transforms as t_2
from matplotlib import pyplot
import warnings
import resnet50_vit_model
from resnet50_vit_model import VIT, PatchEmbedding, multiHeadAttention, residual, mlp, TransformerBlock, Transformer, \
    Classification
from torch.autograd import Variable
import os
from werkzeug.utils import secure_filename
from flask import Flask, flash, request, redirect, send_file, render_template
import requests
import download_video
import util
from PIL import Image
from PIL.ExifTags import TAGS
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore")

UPLOAD_FOLDER = os.path.join('static', 'videos')
MY_UPLOAD_FOLDER = os.path.join('uploaded')
Fake_Folder_d_4 = os.path.join('static', 'videos')
UPLOAD_IMAGE_FOLDER = os.path.join('static', 'Full Images')
UPLOAD_FACE_FOLDER = os.path.join('static', 'test_images')
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config["MY_UPLOAD_FOLDER"] = MY_UPLOAD_FOLDER


##########################################################################################################


class ImgAugTransform:
    """Test-Time Transformations"""

    def __init__(self):
        self.aug = iaa.Sequential([
            iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 3.0))),
            iaa.Fliplr(0.5),
            iaa.Affine(rotate=(-20, 20), mode='symmetric'),
            iaa.Sometimes(0.25,
                          iaa.OneOf([iaa.Dropout(p=(0, 0.1)),
                                     iaa.CoarseDropout(0.1, size_percent=0.5)])),
            iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True)
        ])

    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img)


transforms = ImgAugTransform()

stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
transforms = torchvision.transforms.Compose([
    ImgAugTransform(),
    lambda x: PIL.Image.fromarray(x),
    torchvision.transforms.RandomVerticalFlip(),
    tt.RandomHorizontalFlip(),
    tt.RandomResizedCrop((224, 224), interpolation=2),
    tt.ToTensor(),
    tt.Normalize(*stats, inplace=True)
])

test_tfms = transforms

aug_6 = A.Compose({A.RandomResizedCrop(224, 224)})

##########################################################################################################

"""GPU Handling"""


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


device = get_default_device()

model_4 = torch.load("29_d56.pth", map_location=device)
model_4.to(device)  # Put Model on CPU or GPU
model_4.eval()  # Put Model into Evaluation Mode

model_5 = torch.load("model_2.pth", map_location=device)
model_5.to(device)  # Put Model on CPU or GPU
model_5.eval()  # Put Model into Evaluation Mode

model_6 = torch.load(os.path.join("Trained_Models", "Fake_Face_Detection_Model.pth"), map_location=device)
model_6.to(device)
model_6.eval()

model_7 = torch.load(os.path.join("Trained_Models", "Resnet_VIT_Updated_New_F1.pth"), map_location=device)
model_7.to(device)
model_7.eval()
###############
##########################################################################################################


##########################################################################################################


"""Some Utility Functions"""


def tryint(s):
    try:
        return int(s)
    except:
        return s


def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [tryint(c) for c in re.split('([0-9]+)', s)]


def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)


##########################################################################################################


##########################################################################################################

"""Predict whether the video is fake or real"""


def predict_on_video(video_path, batch_size, input_size, strategy=np.mean, d6=20, d16=1, d_40='a'):
    v_cap = cv2.VideoCapture(video_path)
    v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Count Number Of Frames
    k = 1
    d23 = 20  # Every nth frame is used for prediction(Here every d23 i.e 10th frame is used for prediction)
    d_53 = 0
    d_53 = d_53 + 1

    parent_direc = os.path.join('static', 'videos')
    image_direc = "video_40"
    video_direc = "boxes_video_40"
    y_preds_2 = []

    d15 = 0
    d21 = 0

    path_d_2 = os.path.join(parent_direc, image_direc)  # Path to single image
    # os.mkdir(path_d_2)
    path_d_3 = os.path.join(parent_direc, video_direc)
    # os.mkdir(path_d_3)
    input_folder = path_d_2
    c67 = []
    c72 = 0
    c73 = 0
    c67.append(0.5)

    # Delete Contents of Image Directory And Video Directory #

    for i in range(v_len):
        # Load frame

        success = v_cap.grab()
        if i % d23 == 0:
            success, frame_4 = v_cap.retrieve()
        else:
            continue
        if not success:
            continue
        mtcnn = MTCNN(keep_all=True, min_face_size=176, thresholds=[0.85, 0.90, 0.90],
                      device=device)  # Initialize Face Detector
        frame = cv2.cvtColor(frame_4, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame.astype(np.uint8))

        frame = img
        # Detect face

        boxes, landmarks = mtcnn.detect(frame, landmarks=False)  # Detect Faces
        try:
            if len(boxes) != None:
                pass
        except:
            continue
        b1 = img
        for i in range(0, len(boxes)):
            x, y, width, height = boxes[i]
            max_width, max_height = b1.size
            # Crop Face From Image ##############
            boxes[i][0] -= width / 10
            boxes[i][1] -= height / 10
            boxes[i][0] = max(0, boxes[i][0])
            boxes[i][1] = max(0, boxes[i][1])
            boxes[i][2] = min(boxes[i][2] + (width / 10), max_width)
            boxes[i][3] = min(boxes[i][3] + (height / 10), max_height)
            b4 = b1.crop(boxes[i])
            ###########################################
            # img = test_tfms(b4)### Apply testing time transforms
            # img = img.unsqueeze(0)### Convert Single image ito a batch
            # img = to_device(img,device) ### Put it on GPU if available else on CPU
            # out_3 = model_4(img) #### Pass Image through the mode0l

            # out_3=torch.softmax(out_3.squeeze(),0)
            # d34 = out_3.cpu().detach().numpy()[0]

            b4.save("test_face_image.jpg")  # Save Cropped Image

            image = plt.imread("test_face_image.jpg")
            # print("ghnd")

            image = Image.fromarray(image).convert('RGB')
            # print("gfnk")
            image = aug_6(image=np.array(image))['image']
            image = np.transpose(image, (2, 0, 1)).astype(np.float32)
            image = torch.tensor(image, dtype=torch.float)

            # img = train_tfms(b4)##### Apply Transformations
            img = image.unsqueeze(
                0)  # Converting the Single Image into a batch of Single Image-----(3,380,380) to(1,3,380,380)
            img = to_device(img, device)  # Put the Image On the GPU if available
            out_3 = model_5(img)  # Pass through the model
            print("Video  probaility:", out_3)
            out_3 = torch.softmax(out_3.squeeze(), 0)
            print("Video predicted probaility value:", out_3)
            out_6 = out_3.cpu().detach().numpy()[1]
            print("numpy value:",out_6)
            y_preds_2.append(out_6)  # Append the Output of the Model to the list

            k = k + 1

            if out_6 > 0.5:
                label = "Fake"
            else:
                label = "Real"

            color = (0, 0, 255) if label == "Fake" else (0, 255, 0)

            # Correct This

            cv2.rectangle(frame_4, (int(boxes[i][0]), int(boxes[i][1])), (int(boxes[i][2]), int(boxes[i][3])), color,
                          2)  # Apply Bounding Box to the Frame,Red -Fake,Green-Real
            #####
            if d21 < 2:
                cv2.imwrite(os.path.join(path_d_2, str(d15) + ".jpg"), frame_4)  # Save Image to later get its shape
            c67.append(frame_4)  # Append the frames to later combine together to form a video
            # c72,c73=frame.shape
            d15 = d15 + 1
            d21 = d21 + 1

    if d21 == 0:
        cv2.imwrite(os.path.join(path_d_2, str(d15) + ".jpg"), frame_4)
        d15 = d15 + 1

    try:
        frame = cv2.imread(os.path.join(input_folder, "0.jpg"))  # Load a frame
        height, width, _ = frame.shape  # Find the shape of the frame
        j = 52
        out = cv2.VideoWriter(os.path.join(path_d_3, d_40), cv2.VideoWriter_fourcc(*'DIVX'), 2,
                              (width, height))  # Initialize a Video
        # c4 = os.listdir(input_folder)### Get the names of the frames that have been used for prediction
        # sort_nicely(c4) ### Sort them Numerically
        i = 0
        for i in range(len(c67)):
            out.write(c67[i])  # Write to the Video
        out.release()  # Release the Video
    except:
        pass

    return np.array(y_preds_2).mean(), path_d_3


####################################################
#global tag_name 
#global Metainfo 



###Function for Metadata Information of Image
def metadata_image(filename):
    filename = 'fimt.' + filename.split(".")[-1]
    print("Metadata function file name: ", filename)
    #d21 = 0
    # final_image = os.path.join(UPLOAD_FACE_FOLDER, filename)
    print("Metadata File Path: ",os.path.join(UPLOAD_IMAGE_FOLDER, filename))
    final_image = os.path.join(UPLOAD_IMAGE_FOLDER, filename)
    img = Image.open(final_image)
    exifdata = img.getexif()
    global tag_name, Metainfo
    tag_name = []
    Metainfo = []
    model_name = "None"
    company_name = "None"
    colorname = "None"
    software = "None"
    GPS = "None"
    Orient = "None"
    DATE = "None"
    ResU = "None"
    Exif = "None"
    YRes = "None"
    XRes = "None"
    for tagid in exifdata:
        #print(tagid)
        # getting the tag name instead of tag id
        tagname = TAGS.get(tagid, tagid) 
        value = exifdata.get(tagid)
        tag_name.append(tagname)
        Metainfo.append(value)


        
        if tagname == 'Model':
          model_name = exifdata.get(tagid)
        elif tagname == 'Make':
          company_name = exifdata.get(tagid)
        elif tagname == 'ColorSpace':
          colorname = exifdata.get(tagid)
        elif tagname == 'Software':
          software = exifdata.get(tagid)
        elif tagname == 'GPSInfo':
          GPS = exifdata.get(tagid)
        elif tagname == 'Orientation':
          Orient = exifdata.get(tagid)
        elif tagname == 'DateTime':
          DATE = exifdata.get(tagid)
        elif tagname == 'YResolution':
          YRes = exifdata.get(tagid)
        elif tagname == 'XResolution':
          XRes = exifdata.get(tagid)
        elif tagname == 'ExifOffset':
          Exif = exifdata.get(tagid)
        elif tagname == 'ResolutionUnit':
          ResU = exifdata.get(tagid)
    #print("tags:",tag_name)
    #print("Metadat:", Metainfo)
    print("Model Name:",model_name)

    return model_name, company_name, colorname, software,GPS, Orient, DATE,YRes,XRes, Exif, ResU


###########################################################

## function for metadata of video 

import ffmpeg
global vid_tag, vid_val
vid_tag = []
vid_val = []

def get_videometadata(filename):
  #filename = 'fimt.' + filename.split(".")[-1]
  print("Metadata function of video file name: ", filename)

    #d21 = 0
    # final_image = os.path.join(UPLOAD_FACE_FOLDER, filename)
  #print("Metadata video File Path: ",os.path.join(UPLOAD_FOLDER, filename))
  final_image = os.path.join(app.config['UPLOAD_FOLDER'], filename)
  print("Metadata video File Path to pass: ",final_image)
  #img = Image.open(final_image)
    
  vid = ffmpeg.probe(final_image)
  #print(vid)
  print( vid['streams'])

  videometadata = vid['streams'][0]
  #print(videometadata)
  for tagid in videometadata:
    tagname = TAGS.get(tagid, tagid)
    vid_tag.append(tagname)
    value = videometadata.get(tagid)
    vid_val.append(value)
  

  
  print("Tag of video:",vid_tag)
  print("Values of video:",vid_val)
  a0 = vid_tag[0]
  print("One value:",a0)
  return vid_tag,vid_val    



##########################################################################################################
########################################################################################################
# predicting whether the Face is Fake or Not
'''
def predict_on_face(image_path, batch_size, input_size):

    d21=0
    frame_4= cv2.imread(os.path.join("test_images",image_path))### Load Image
    ### If cropping is not required
    if(crop==False):

        image = plt.imread(os.path.join("test_images",image_path))#### Load Image
        image = Image.fromarray(image).convert('RGB')#### Convert Image to RGB
        image = aug_3(image=np.array(image))['image']##### Apply Transformations
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        image = torch.tensor(image, dtype=torch.float)###### Convert Image to Tensor
        pred = model(image.unsqueeze(0).to(device)).argmax()#### Pass through Model
        print("Predicted Value:", pred)

        pred1 = model(image.unsqueeze(0).to(device))
        print(pred1)
        if(pred==1):
            label="Fake"
        else:
            label="Real"

        color = (0, 0, 255) if label=="Fake" else (0, 255, 0)

        print(label + " "+ "Face Found")
        exit()


    mtcnn = MTCNN(keep_all=True,min_face_size=176,thresholds=[0.85,0.90,0.90],device=device)#### Load Face Detector
    frame = cv2.cvtColor(frame_4, cv2.COLOR_BGR2RGB)
    frame_4 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    img = Image.fromarray(frame.astype(np.uint8))


    frame=img
    k=0

    # Detect face
    try:
        boxes,landmarks = mtcnn.detect(frame, landmarks=False)##### Detect Faces
        b1 = img
    #print(boxes)
    except:
        print("Couldn't Find a Face!!!!")##### If mtcnn fails,this print statement is executed
        exit()
   ###### Loop over all faces
    try:
        for i in range(0,len(boxes)):
            #### Crop Faces from Images
            x,y,width,height =boxes[i]
            max_width,max_height = b1.size
            boxes[i][0]-=width/10
            boxes[i][1]-=height/10
            boxes[i][0] = max(0,boxes[i][0])
            boxes[i][1] = max(0,boxes[i][1])
            boxes[i][2] =min(boxes[i][2]+(width/10),max_width)
            boxes[i][3] =min(boxes[i][3]+(height/10),max_height)
            b4 = b1.crop(boxes[i])
            b4.save("test_face_image.jpg")##### Save Cropped Image


            image = plt.imread("test_face_image.jpg")#### Read Cropped Image
            image = Image.fromarray(image).convert('RGB')#### Convet to RGB
            image = aug_3(image=np.array(image))['image']##### Apply Transformations
            image = np.transpose(image, (2, 0, 1)).astype(np.float32)
            image = torch.tensor(image, dtype=torch.float)##### Convert to tensor
            pred = model(image.unsqueeze(0).to(device)).argmax()##### Pass thorugh The Model



            if(pred==1):
                label="Fake"
            else:
                label="Real"

            color = (0, 0, 255) if label=="Fake" else (0, 255, 0)
            cv2.rectangle(frame_4, (boxes[i][0],boxes[i][1]), (boxes[i][2],boxes[i][3]), color, 2)##### Draw Bounding Box
            cv2.imwrite(os.path.join("test_faces_output",image_path),frame_4)##### Save the Image
            print(label + " "+ "Face Found")
    except:
        print("-----------------***********No Faces Found***********-----------------")
        return

'''

########################################################################################################
########################################################################################################
"""Flask App"""


@app.route('/')
def predict_2():
    """Home Page"""
    return render_template('website_home_copy.html')


@app.route('/uploader', methods=['GET', 'POST'])
def file_uploader():
    if request.method == 'POST':

        # If there is video url, then only process this
        video_url = request.form.get("video_url")
        # if "youtube" or "youtu.be" in video_url:
        if video_url != "" and util.valid_youtube_video_url(video_url):
            file_name = download_video.youtube_video_downloader(video_url, os.path.join(app.config['UPLOAD_FOLDER']))
            if file_name != -1:
                posts = download_file(file_name)
                return render_template('masonry.html', value=posts, filename=posts)
            else:
                return redirect("/")
        else:
            image_url = request.form.get("image_url")
            is_url_valid = util.valid_image_url(image_url) if image_url != "" else False
            print("\033[92m" + "image url: " + image_url + "\033[0m")
            print("\033[92m" + f"is Valid: {is_url_valid}"  + "\033[0m")
            filename, file_type = "", ""

            if is_url_valid:
                filename = util.save_image_from_url(image_url=image_url, folder_destination=UPLOAD_IMAGE_FOLDER)
            else:
                # BELOW CODE IS TO DETECT FAKE FACE IMAGE AND FAKE IMAGE
                if request.files['file1'].filename == "":
                    return redirect("/")

                is_video = util.check_video_or_image_file(request.files['file1'].filename)
                if is_video == 1:
                    # For Video file
                    filename, file_type = util.save_image_from_form(request.files, os.path.join(app.config['UPLOAD_FOLDER']))
                else:
                    # For Image file
                    filename, file_type = util.save_image_from_form(request.files, UPLOAD_IMAGE_FOLDER)
            
            if file_type == "video":
                posts = download_file(filename)
                #posts = get_videometadata(filename)
                return render_template('masonry.html', value=posts, filename=posts)
            else:
                label, score, fake_image_path = Fake_Face(filename)
                #a,b,c,d,e,f,g,h,i,j,k = metadata_image(filename)

                #print("Modelbbbb",a)



                final_image = os.path.join(UPLOAD_IMAGE_FOLDER, 'fimt.' + filename.split(".")[-1])

                if label == -1:
                    # this image is for FIM
                    print("Now this is FIM")
                    aug_3 = A.Compose({A.Resize(255, 255)})

                    temp = pathlib.PosixPath
                    pathlib.PosixPath = pathlib.WindowsPath

                    dice = Dice()

                    learn = load_learner(os.path.join("Trained_Models", "Copy_Move_FIM.pkl"))

                    original_image = plt.imread(final_image)
                    original_image = Image.fromarray(original_image).convert('RGB')  # Convet to RGB
                    original_image = aug_3(image=np.array(original_image))['image']  # Apply Transformations
                    original_image = Image.fromarray(original_image)
                    original_image.save(os.path.join('static', 'Test_FIM', 'original_image.jpg'))

                    b1 = learn.predict(os.path.join('static', 'Test_FIM', 'original_image.jpg'))
                    # plt.imshow(b1[0].permute(1,2,0))
                    final_mask = b1[0].permute(1, 2, 0)
                    im_1 = t_2.ToPILImage()(np.uint8(final_mask))
                    im_1.save(os.path.join('static', 'Test_FIM', 'final_mask_image_Copy_Move.jpg'))
                    
                    learn = load_learner(os.path.join("Trained_Models", "Splicing_FIM.pkl"))
                    b1 = learn.predict(os.path.join('static', 'Test_FIM', 'original_image.jpg'))
                    # plt.imshow(b1[0].permute(1,2,0))
                    final_mask = b1[0].permute(1, 2, 0)
                    im_2 = t_2.ToPILImage()(np.uint8(final_mask))
                    im_2.save(os.path.join("static", "Test_FIM", "final_mask_image_Splicing.jpg"))

                    learn = load_learner(os.path.join("Trained_Models", "Inpainting_FIM.pkl"))
                    b1 = learn.predict(os.path.join('static', 'Test_FIM', 'original_image.jpg'))

                    # plt.imshow(b1[0].permute(1,2,0))
                    final_mask = b1[0].permute(1, 2, 0)
                    im_3 = t_2.ToPILImage()(np.uint8(final_mask))
                    im_3.save(os.path.join('static', 'Test_FIM', 'final_mask_image_Inpainting.jpg'))

                    posts = {
                        'Original_Image': os.path.join('static', 'Test_FIM', 'original_image.jpg'),
                        'Splicing_Image': os.path.join('static', 'Test_FIM', 'final_mask_image_Splicing.jpg'),
                        'Inpainting_Image': os.path.join('static', 'Test_FIM', 'final_mask_image_Inpainting.jpg'),
                        'Copy_Move_Image': os.path.join('static', 'Test_FIM', 'final_mask_image_Copy_Move.jpg'),
                    }

                    # plt.show()
                    return render_template('FIM_Output.html', value=posts, filename=posts)
                else:
                    a,b,c,d,e,f,g,h,i,j,k = metadata_image(filename)
                    print("Now this is FFM")
                    posts = {
                        'Original_Image': final_image,
                        'label': label,
                        'score': score,
                        'x':a,
                        'b1':b,
                        'c1':c,
                        'd1':d,
                        'e1':e,
                        'f1':f,
                        'g1':g,
                        'h1':h,
                        'i1':i,
                        'j1':j,
                        'k1':k,                        
                                   
                        'fake_img_path': fake_image_path
                    }
                    #posts = metadata_image(filename)
                    return render_template('FFM_Output.html', value=posts, filename=posts)
    else:
            return redirect(request.url)


###################################
# FIM routing on App


@app.route('/predict_FIM', methods=['GET', 'POST'])
def predict_3():
    """Fake Image routing"""
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            print('no file')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            print('no filename')
            return redirect(request.url)
        else:
            filename = secure_filename(file.filename)
            print(filename)
            file.save(os.path.join(UPLOAD_IMAGE_FOLDER, 'fimt' + '.' + filename.split('.')[-1]))
            print("saved file successfully")
            # send file name as parameter to downlad
            return redirect('/FIM_Pata_Karo/' + 'fimt' + '.' + filename.split('.')[-1])
    return render_template('upload_FIM.html')


@app.after_request
def add_header(r):
    r.headers['X-UA-Compatible'] = "IE=Edge,chrome=1"
    r.headers['Cache-Control'] = "no-cache, no-store , must-revalidate"
    r.headers['Pragma'] = "no-cache"
    r.headers['Expires'] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r


def get_my_x(fname: Path): pass


def get_my_y(fname: Path):
    pass


def acc_camvid(*_): pass


def get_y(*_): pass


@app.route("/FIM_Pata_Karo/<filename>", methods=['GET'])
def detect_FIM(filename):
    final_image = os.path.join(UPLOAD_IMAGE_FOLDER, filename)
    aug_3 = A.Compose({A.Resize(255, 255)})

    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath

    dice = Dice()

    learn = load_learner(os.path.join("Trained_Models", "Copy_Move_FIM.pkl"))

    original_image = plt.imread(final_image)
    original_image = Image.fromarray(original_image).convert('RGB')  # Convet to RGB
    original_image = aug_3(image=np.array(original_image))['image']  # Apply Transformations
    original_image = Image.fromarray(original_image)
    original_image.save(os.path.join('static', 'Test_FIM', 'original_image.jpg'))

    b1 = learn.predict(os.path.join('static', 'Test_FIM', 'original_image.jpg'))
    # plt.imshow(b1[0].permute(1,2,0))
    final_mask = b1[0].permute(1, 2, 0)
    im_1 = t_2.ToPILImage()(np.uint8(final_mask))
    im_1.save(os.path.join('static', 'Test_FIM', 'final_mask_image_Copy_Move.jpg'))

    learn = load_learner(os.path.join("Trained_Models", "Splicing_FIM.pkl"))
    b1 = learn.predict(os.path.join('static', 'Test_FIM', 'original_image.jpg'))
    # plt.imshow(b1[0].permute(1,2,0))
    final_mask = b1[0].permute(1, 2, 0)
    im_2 = t_2.ToPILImage()(np.uint8(final_mask))
    im_2.save(os.path.join("static", "Test_FIM", "final_mask_image_Splicing.jpg"))

    learn = load_learner(os.path.join("Trained_Models", "Inpainting_FIM.pkl"))
    b1 = learn.predict(os.path.join('static', 'Test_FIM', 'original_image.jpg'))

    # plt.imshow(b1[0].permute(1,2,0))
    final_mask = b1[0].permute(1, 2, 0)
    im_3 = t_2.ToPILImage()(np.uint8(final_mask))
    im_3.save(os.path.join('static', 'Test_FIM', 'final_mask_image_Inpainting.jpg'))

    posts = {
        'Original_Image': os.path.join('static', 'Test_FIM', 'original_image.jpg'),
        'Splicing_Image': os.path.join('static', 'Test_FIM', 'final_mask_image_Splicing.jpg'),
        'Inpainting_Image': os.path.join('static', 'Test_FIM', 'final_mask_image_Inpainting.jpg'),
        'Copy_Move_Image': os.path.join('static', 'Test_FIM', 'final_mask_image_Copy_Move.jpg'),
    }

    # plt.show()
    return render_template('FIM_Output.html', value=posts, filename=posts)


########################################################
## Directing to Fake Faces
# @app.route('/uploadfileFace', methods=['GET', 'POST'])
# def upload_file1():
#     if request.method == 'POST':
#         # check if the post request has the file part
#         if 'file' not in request.files:
#             print('no file')
#             return redirect(request.url)
#         file = request.files['file']
#         # if user does not select file, browser also
#         # submit a empty part without filename
#         if file.filename == '':
#             print('no filename')
#             return redirect(request.url)
#         else:
#             filename = secure_filename(file.filename)
#             print(filename)
#             file.save(os.path.join(UPLOAD_FACE_FOLDER, 'fimt1' + '.' + filename.split('.')[-1]))  #######Throwing error
#             print("saved file successfully")
#             # send file name as parameter to download
#             return redirect('/Fake_Face/' + 'fimt1' + '.' + filename.split('.')[-1])
#     return render_template('upload_file_face.html')


@app.after_request
def add_header(r):
    r.headers['X-UA-Compatible'] = "IE=Edge,chrome=1"
    r.headers['Cache-Control'] = "no-cache, no-store , must-revalidate"
    r.headers['Pragma'] = "no-cache"
    r.headers['Expires'] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r


aug_3 = A.Compose({
    A.RandomResizedCrop(224, 224)
})


##########Added######################
def convertImageToTensor(final_image):
    image_transforms = tt.Compose([
        tt.Resize((384, 384)),
        tt.ToTensor(),
        tt.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    ])
    image = Image.open(final_image)
    imageTensor = image_transforms(image).float()
    imageTensor = Variable(imageTensor, requires_grad=False)
    imageTensor = imageTensor.unsqueeze(0)  # converted in the form of batch
    return imageTensor


####################################
'''
def get_my_x(fname:Path): pass

def get_my_y(fname:Path):
    pass


def acc_camvid(*_): pass

def get_y(*_): pass

'''






# @app.route("/Fake_Face/<filename>", methods = ['GET'])
def Fake_Face(filename, crop=True, model_number=7):
    filename = 'fimt.' + filename.split(".")[-1]
    print("function file name: ", filename)
    d21 = 0
    # final_image = os.path.join(UPLOAD_FACE_FOLDER, filename)
    print(os.path.join(UPLOAD_IMAGE_FOLDER, filename))
    final_image = os.path.join(UPLOAD_IMAGE_FOLDER, filename)


    print("FAKE_FACE FUNCTION", final_image)

    frame_4 = cv2.imread(final_image)  # Load Image
    # print("frame_4 = ")
    # print(frame_4)

    mtcnn = MTCNN(keep_all=True, min_face_size=176, thresholds=[0.95, 0.95, 0.95],
                  device=device)  # Load Face Detector
    frame_5 = cv2.cvtColor(frame_4, cv2.COLOR_BGR2RGB)
    img2 = Image.fromarray(frame_5.astype(np.uint8))

    frame_5 = img2
    try:
        boxes, landmarks = mtcnn.detect(frame_5, landmarks=False)  # Detect Faces
        print("xyz")
        # b1 = img
    # print(boxes)
    except:
        print("Couldn't Find a Face!!!!")  # If mtcnn fails,this print statement is executed
        return -1,None,''
    #

    # Loading Model
    # model=torch.load(os.path.join("Trained_Models","Fake_Face_Detection_Model.pth"), map_location=device)

    ########Added########################
    # model_number = 6## 6 or 7  6-Fake_Face_Detection_Model, 7- Resnet50_VIT_Model
    #####################################

    ### If cropping is not required
    crop = True  ####Added
    if crop == False:

        image = plt.imread(final_image)  #### Load Image
        image = Image.fromarray(image).convert('RGB')  #### Convert Image to RGB
        image.save("test_face_image.jpg")

        if (model_number == 6):
            image = aug_3(image=np.array(image))['image']  ##### Apply Transformations
            # original_image1 = Image.fromarray(image)
            # original_image1.save(os.path.join('test_faces_output','original_image1.jpg'))
            image = np.transpose(image, (2, 0, 1)).astype(np.float32)
            image = torch.tensor(image, dtype=torch.float)  ###### Convert Image to Tensor
            pred = model_6(image.unsqueeze(0).to(device)).argmax()  #### Pass through Model
            print("Predicted Value:", pred)

            pred1 = model_6(image.unsqueeze(0).to(device))
            print(pred1)
            if (pred == 1):
                label = "Fake"
            else:
                label = "Real"

            color = (0, 0, 255) if label == "Fake" else (0, 255, 0)

            print(label + " " + "Face Found")
            # exit()
            return label, "test_face_image.jpg"
        ############Added######################
        elif (model_number == 7):
            print("Inside model_7")
            image = convertImageToTensor(final_image)  ##originally were sending image path
            print("Image converted")
            image = image.to(device)
            print(image.shape)
            preds = model_7(image)  # get the softmax probabilities
            #preds = preds.cpu()

            ###
            pred6 = torch.softmax(preds.squeeze(), 0)
            pred6 = pred6.cpu().detach().numpy()[0]
            ###
            pred = torch.argmax(preds, 1)[0]
            print("predicted value: 0- Fake, 1-Real", preds,"numpy val",pred6)
            if (pred == 1):
                label = "Real"
            else:
                label = "Fake"

            color = (0, 0, 255) if label == "Fake" else (0, 255, 0)

            print(label + " " + "Face Found")
            # exit()
            return label, pred6, "test_face_image.jpg"
        ###########################

    mtcnn = MTCNN(keep_all=True, min_face_size=176, thresholds=[0.85, 0.90, 0.90],
                  device=device)  # Load Face Detector
    frame = cv2.cvtColor(frame_4, cv2.COLOR_BGR2RGB)
    frame_4 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    img = Image.fromarray(frame.astype(np.uint8))

    frame = img
    k = 0

    # Detect face
    try:
        boxes, landmarks = mtcnn.detect(frame, landmarks=False)  # Detect Faces
        b1 = img
    # print(boxes)
    except:
        print("Couldn't Find a Face!!!!")  ##### If mtcnn fails,this print statement is executed
        return -1,None,''
    ###### Loop over all faces
    try:
        for i in range(0, len(boxes)):
            #### Crop Faces from Images
            x, y, width, height = boxes[i]
            max_width, max_height = b1.size
            boxes[i][0] -= width / 10
            boxes[i][1] -= height / 10
            boxes[i][0] = max(0, boxes[i][0])
            boxes[i][1] = max(0, boxes[i][1])
            boxes[i][2] = min(boxes[i][2] + (width / 10), max_width)
            boxes[i][3] = min(boxes[i][3] + (height / 10), max_height)
            b4 = b1.crop(boxes[i])
            b4.save("test_face_image.jpg")  ##### Save Cropped Image

            if model_number == 6:
                image = plt.imread("test_face_image.jpg")  #### Read Cropped Image
                image = Image.fromarray(image).convert('RGB')  #### Convet to RGB
                image = aug_3(image=np.array(image))['image']  ##### Apply Transformations
                image = np.transpose(image, (2, 0, 1)).astype(np.float32)
                image = torch.tensor(image, dtype=torch.float)  ##### Convert to tensor
                pred = model_6(image.unsqueeze(0).to(device)).argmax()  ##### Pass thorugh The Model

                if (pred == 1):
                    label = "Fake"
                else:
                    label = "Real"

            elif model_number == 7:
                print("Inside model_7,crop=true")
                image = convertImageToTensor("test_face_image.jpg")  ##originally were sending image path
                print("Image converted")
                image = image.to(device)
                print(image.shape)
                preds = model_7(image)  # get the softmax probabilities
                #preds = preds.cpu()
                ####
                pred6 = torch.softmax(preds.squeeze(), 0)
                pred6 = pred6.cpu().detach().numpy()[0]
            
                ###
                
                
                pred = torch.argmax(preds, 1)[0]
                print("predicted value1: 0- Fake, 1-Real", preds," numpy value",pred6)
                if (pred == 1):
                    label = "Real"
                else:
                    label = "Fake"

            color = (0, 0, 255) if label == "Fake" else (0, 255, 0)
            print("Boxes: ", boxes)
            cv2.rectangle(frame_4, (boxes[i][0], boxes[i][1]), (boxes[i][2], boxes[i][3]), color,
                          2)  # Draw Bounding Box
            cv2.imwrite(os.path.join("test_faces_output", filename), frame_4)  ##### Save the Image
            # print(label + " "+ "Face Found")
            return label, pred6, os.path.join("test_faces_output", filename)
    except Exception as e:
        print("-----------------***********No Faces Found***********-----------------")
        print(e)
        return -1,None,''


@app.route('/predict_FFM', methods=['GET', 'POST'])
def predict_4():
    print("here")
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            print('no file')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            print('no filename')
            return redirect(request.url)
        else:
            filename = secure_filename(file.filename)
            print(filename)
            file.save(os.path.join(UPLOAD_FACE_FOLDER, 'fimt' + '.' + filename.split('.')[-1]))
            print("saved file successfully")
            # send file name as parameter to downlad
            return redirect('/FFM_Pata_Karo/' + 'fimt' + '.' + filename.split('.')[-1])
    return render_template('upload_FFM.html')


# Fake image
@app.route("/FFM_Pata_Karo/<filename>", methods=['GET'])
def detect_FFM(filename):
    final_image = os.path.join(UPLOAD_FACE_FOLDER, filename)
    label,score, fake_image_path = Fake_Face(filename)
    a,b,c,d,e,f,g,h,i,j,k = metadata_image(filename)
    print("Modelaa",a)
    if label == -1:
        posts = {
            'Original_Image': "error.jpg",
            'label': -1,
            'score': 0,
            'fake_img_path': "error.jpg"
        }
    else:
        posts = {
            'Original_Image': final_image,
            'label': label,
            'score': score,
            'x':a,
            'b1':b,
            'c1':c,
            'd1':d,
            'e1':e,
            'f1':f,
            'g1':g,
            'h1':h,
            'i1':i,
            'j1':j,
            'k1':k,                                                
            #'x':a,
            #'y':b,
            #'z':c,
            #'p':d,
            'fake_img_path': fake_image_path

        }
    return render_template('FFM_Output.html', value=posts, filename=posts)


'''
    posts = {
                 'Original_Image1' : os.path.join(test_faces_output,'original_image1.jpg'),
                 #'Splicing_Image': os.path.join('static','Test_FIM','final_mask_image_Splicing.jpg'),
                 #'Inpainting_Image':os.path.join('static','Test_FIM','final_mask_image_Inpainting.jpg'),
                 #'Copy_Move_Image': os.path.join('static','Test_FIM','final_mask_image_Copy_Move.jpg'),
                }
'''
# plt.show()
# return render_template('FIM_Output.html',value=posts,filename=posts)
'''
def detect_FIM(filename):

    final_image = os.path.join(UPLOAD_FACE_FOLDER, filename)
    aug_3 = A.Compose({
         A.Resize(255,255)
        })


    temp=pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath

    #dice = Dice()



    model=torch.load(os.path.join("Trained_Models","Fake_Face_Detection_Model.pth"), map_location=device)
    #learn = load_learner(os.path.join("Trained_Models","Copy_Move_FIM.pkl"))

    original_image=plt.imread(final_image)
    original_image = Image.fromarray(original_image).convert('RGB')#### Convet to RGB
    original_image = aug_3(image=np.array(original_image))['image']##### Apply Transformations
    original_image = Image.fromarray(original_image)
    original_image.save(os.path.join('static','Test_FIM','original_image.jpg'))




    b1 = learn.predict(os.path.join('static','Test_FIM','original_image.jpg'))
    #plt.imshow(b1[0].permute(1,2,0))
    final_mask = b1[0].permute(1,2,0)
    im_1 = t_2.ToPILImage()(np.uint8(final_mask))
    im_1.save(os.path.join('static','Test_FIM','final_mask_image_Copy_Move.jpg'))


    learn = load_learner(os.path.join("Trained_Models","Splicing_FIM.pkl"))
    b1 = learn.predict(os.path.join('static','Test_FIM','original_image.jpg'))
    #plt.imshow(b1[0].permute(1,2,0))
    final_mask = b1[0].permute(1,2,0)
    im_2 = t_2.ToPILImage()(np.uint8(final_mask))
    im_2.save(os.path.join("static","Test_FIM","final_mask_image_Splicing.jpg"))



    learn = load_learner(os.path.join("Trained_Models","Inpainting_FIM.pkl"))
    b1 = learn.predict(os.path.join('static','Test_FIM','original_image.jpg'))


    #plt.imshow(b1[0].permute(1,2,0))
    final_mask = b1[0].permute(1,2,0)
    im_3 = t_2.ToPILImage()(np.uint8(final_mask))
    im_3.save(os.path.join('static','Test_FIM','final_mask_image_Inpainting.jpg'))

    posts = {
                 'Original_Image' : os.path.join('static','Test_FIM','original_image.jpg'),
                 'Splicing_Image': os.path.join('static','Test_FIM','final_mask_image_Splicing.jpg'),
                 'Inpainting_Image':os.path.join('static','Test_FIM','final_mask_image_Inpainting.jpg'),
                 'Copy_Move_Image': os.path.join('static','Test_FIM','final_mask_image_Copy_Move.jpg'),
                 }

    #plt.show()
    return render_template('FIM_Output.html',value=posts,filename=posts)


'''

'''
@app.route('/')
def hello1():
    return render_template('masonry.html')

input_size=224

@app.route("/downloadfile1/<filename>", methods = ['GET'])
def download_file(filename):
    num_d_3,final_video = predict_on_video(video_path=os.path.join(UPLOAD_FACE_FOLDER,filename),batch_size=10,input_size=input_size,d6=0,d_40=filename[:-4]+'.mp4')

    clip = moviepy.VideoFileClip(os.path.join(UPLOAD_FOLDER,'boxes_video_40',filename[:-4:]+'.mp4'))
    b2 = clip.write_videofile(os.path.join(UPLOAD_FOLDER,'boxes_video_40',filename))
    f_type="Real"
    if(num_d_3<0.5):
        f_type = "Real"
    else:
        f_type= "Fake"
    posts = {
                 'input_f_n' : os.path.join(UPLOAD_FOLDER,filename),
                 'output_f_n': os.path.join(UPLOAD_FOLDER,'boxes_video_40',filename),
                 'final_video_id':52,
                 'predicted_fake_score':0.6,
                 'title': str(f_type)
                 }

    return render_template('masonry.html',value=posts,filename=posts)


@app.route('/return-files/<filename>')
def return_files_tut_real(filename):
    #file_path =filename[input_f_n]
    file_path=filename
    return send_file(file_path, as_attachment=True, attachment_filename='')

@app.route('/return-files/<filename>')
def return_files_tut_fake(filename):
    #file_path =filename[input_f_n]
    file_path=filename
    return send_file(file_path, as_attachment=True, attachment_filename='')


'''


##########################################################


@app.route('/uploadfile', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            print('no file')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            print('no filename')
            return redirect(request.url)
        else:
            filename = secure_filename(file.filename)
            print(filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'fimt' + '.' + filename.split('.')[-1]))
            print("saved file successfully")
            # send file name as parameter to downlad
            return redirect('/downloadfile/' + 'fimt' + '.' + filename.split('.')[-1])
    return render_template('upload_file.html')


@app.route('/')
def hello():
    return render_template('masonry.html')


input_size = 224

# fistname, lastname, address
# imageurl, videourl, image file only



# @app.route("/downloadfile/<filename>", methods=['GET'])
def download_file(filename):
    print("Download file name: ", filename)
    # f.save(os.path.join(UPLOAD_IMAGE_FOLDER, 'fimt' + '.' + filename.split('.')[-1]))
    # file_path = os.path.join(UPLOAD_IMAGE_FOLDER, 'fimt' + '.' + filename.split('.')[-1])
    # num_d_3, final_video = predict_on_video(video_path=os.path.join(UPLOAD_FOLDER, filename), batch_size=10,
    #                                         input_size=input_size, d6=0, d_40=filename[:-4] + '.mp4')
    # num_d_3, final_video = predict_on_video(video_path=file_path, batch_size=10,
    #                                         input_size=input_size, d6=0, d_40=filename[:-4] + '.mp4')
    num_d_3, final_video = predict_on_video(
        video_path=os.path.join(app.config['UPLOAD_FOLDER'], filename), 
        batch_size=10,
        input_size=input_size, d6=0, d_40=filename[:-4] + '.mp4'
    )

    get_videometadata(filename)
    print("num_d_3:", num_d_3)
    print(f"Filename: {filename}")
    print(filename[:-4:])

    clip = moviepy.VideoFileClip(os.path.join(UPLOAD_FOLDER, 'boxes_video_40', filename[:-4:] + '.mp4'))
    b2 = clip.write_videofile(os.path.join(UPLOAD_FOLDER, 'boxes_video_40', filename))
    f_type = "Real"

    if num_d_3 < 0.5:
        f_type = "Real"
    else:
        f_type = "Fake"

   
    get_videometadata(filename)
    posts = {
        'input_f_n': os.path.join(UPLOAD_FOLDER, filename),
        'output_f_n': os.path.join(UPLOAD_FOLDER, 'boxes_video_40', filename),
        'final_video_id': 52,
        'predicted_fake_score': num_d_3,
        'title': str(f_type),
        'a':vid_tag[0],
        'a1':vid_val[0],
        'b':vid_tag[1],
        'b1':vid_val[1],
        'c':vid_tag[2],
        'c1':vid_val[2],
        'd':vid_tag[3],
        'd1':vid_val[3],
        'e':vid_tag[4],
        'e1':vid_val[4],
        'f':vid_tag[5],
        'f1':vid_val[5],
        'g':vid_tag[6],
        'g1':vid_val[6],
        'h':vid_tag[7],
        'h1':vid_val[7],
        'i':vid_tag[8],
        'i1':vid_val[8],
        'j':vid_tag[9],
        'j1':vid_val[9],
        'k':vid_tag[10],
        'k1':vid_val[10],
        'l':vid_tag[11],
        'l1':vid_val[11],
        'm':vid_tag[12],
        'm1':vid_val[12],
        'n':vid_tag[13],
        'n1':vid_val[13],
        'o':vid_tag[14],
        'o1':vid_val[14],
        'p':vid_tag[15],
        'p1':vid_val[15],
        'q':vid_tag[16],
        'q1':vid_val[16],
        'r':vid_tag[17],
        'r1':vid_val[17],
        's':vid_tag[18],
        's1':vid_val[18],
        't':vid_tag[19],
        't1':vid_val[19],
        'u':vid_tag[20],
        'u1':vid_val[20],
        'v':vid_tag[21],
        'v1':vid_val[21],
        'w':vid_tag[22],
        'w1':vid_val[22],
        'x':vid_tag[23],
        'x1':vid_val[23],
        'y':vid_tag[24],
        'y1':vid_val[24],
        'z':vid_tag[25],
        'z1':vid_val[25],
        'A':vid_tag[26],
        'A1':vid_val[26],
        'B':vid_tag[27],
        'B1':vid_val[27],
        'C':vid_tag[28],
        'C1':vid_val[28],
        'D':vid_tag[29],
        'D1':vid_val[29],
        'E':vid_tag[30],
        'E1':vid_val[30],
        'F':vid_tag[31],
        'F1':vid_val[31],
        'G':vid_tag[32],
        'G1':vid_val[32]
        
        
        
    }
    return posts

    # return render_template('masonry.html', value=posts, filename=posts)


@app.route('/return-files/<filename>')
def return_files_tut_face(filename):
    # file_path =filename[input_f_n]
    file_path = filename
    return send_file(file_path, as_attachment=True, attachment_filename='')


@app.route('/return-files/<filename>')
def return_files_tut_real(filename):
    # file_path =filename[input_f_n]
    file_path = filename
    return send_file(file_path, as_attachment=True, attachment_filename='')


@app.route('/return-files/<filename>')
def return_files_tut_fake(filename):
    # file_path =filename[input_f_n]
    file_path = filename
    return send_file(file_path, as_attachment=True, attachment_filename='')


if __name__ == '__main__':
    app.run(port=5001, debug=True)
