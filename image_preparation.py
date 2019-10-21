from menpo.feature import hog, ndfeature # you can import no_op and dsift as well (hog is best for imfrared images though)
from menpo import io as mio
from tqdm import tqdm
import numpy as np
import pandas as pd
from menpofit.aam import HolisticAAM as AAM
from menpofit.aam import LucasKanadeAAMFitter as Fitter
from menpofit.aam import WibergInverseCompositional as WIC
from menpofit.aam import SimultaneousInverseCompositional as SIC
import menpo
import menpodetect
import dlib
import os 
from PIL import Image
from matplotlib import pyplot as plt
import joblib
import pickle
import cv2
from skimage.feature import hog

HOG_PATH = "thermalfaceproject/hog_detector.svm"

def label_images(df):
    ind = 231
    df['label'] = ' '
    df.label[:ind] = 'angry'
    df.label[ind:2*ind] = 'sad'
    df.label[2*ind:3*ind] = 'surprised'
    df.label[3*ind:4*ind] = 'happy'
    return df

def load_images(emotions):
    images = []
    for element in emotions:
        IMAGE_PATH = "/Users/joanna_frankiewicz/Desktop/Database/FaceDB_Emotions/images/"+element 

        @ndfeature
        def float32_hog(x):
            return hog(x).astype(np.float32)

        print("Importing images")

        for i in tqdm(mio.import_images(IMAGE_PATH)):
                
            i = i.crop_to_landmarks_proportion(0.1) # Crop images to Landmarks --> only Face on resulting image    
            if i.n_channels > 2: # Convert multichannel images to greyscale
                i = i.as_greyscale()
                        
            images.append(i)   

    print("Succesfully imported %d Images" % len(images))

    train_images_df = pd.DataFrame(images, columns=['image'])
    train_images_df = label_images(train_images_df)
    return train_images_df

def get_fitter(df):
    #Training the AAM (Active Appearance model - a computer vision algorithm for matching a statistical model of object shape and appearance to a new image)
    LANDMARK_GROUP = "LJSON" #or "PTS"
    features = hog

    aam = AAM(
        df.image,
        group=LANDMARK_GROUP
        )

    fitter_alg = WIC 
    n_shape = 0.95 #  --> fraction of shape accuracy to remain (dimensionality reduction through PCA)
    n_appearance = 0.95 #  --> fraction of appearance accuracy to remain (dimensionality reduction through PCA)

    fitter = Fitter(aam=aam, 
                    lk_algorithm_cls=fitter_alg,
                    n_shape=n_shape, 
                    n_appearance=n_appearance)

    return fitter

def face_2_pointcloud(faces):
    if len(faces):
            face = np.array([faces[0].as_vector()[1], faces[0].as_vector()[0],
                             faces[0].range()[1], faces[0].range()[0]]).astype(np.uint16)
            print("Face detected. > ", face)
    else:
        face = np.array([0, 0, 0, 0]).astype(np.uint16)
        print("NO Face detected.")
    
    points = np.array([[face[1], face[0]],
                         [face[1]+face[3], face[0]],
                         [face[1]+face[3], face[0]+face[2]],
                         [face[1], face[0]+face[2]]])

    adjacency_matrix = np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]])

    return menpo.shape.PointDirectedGraph(points, adjacency_matrix)

def save_faces(emotions, face_detector, fitter):
    for element in emotions:
        TEST_IMG_PATH = "/Users/joanna_frankiewicz/Desktop/Database/FaceDB_Emotions/images/"+element
        for filename in os.listdir(TEST_IMG_PATH):
            if filename.endswith(".jpg"):
                img_path = TEST_IMG_PATH+"/"+filename
                test_img = mio.import_image(img_path)
                try:
                    test_face_bb = face_2_pointcloud(face_detector(test_img))
                    fitting_result = fitter.fit_from_bb(test_img, test_face_bb, max_iters=25)
                    
                    # get points of bounding_box. left top and bottom right
                    p_left_top = fitting_result.final_shape.bounds()[0]
                    p_right_bottom = fitting_result.final_shape.bounds()[1]

                    image_width, image_height = test_img.width, test_img.height

                    # clip value to image range
                    p_left_top[0] = np.clip([p_left_top[0]], 0, image_height)[0]
                    p_left_top[1] = np.clip([p_left_top[1]], 0, image_width)[0]
                    p_right_bottom[0] = np.clip([p_right_bottom[0]], 0, image_height)[0]
                    p_right_bottom[1] = np.clip([p_right_bottom[1]], 0, image_width)[0]

                    img_tmp = test_img.pixels.squeeze()[int(p_left_top[0]):int(p_right_bottom[0]), int(p_left_top[1]):int(p_right_bottom[1])]*255
                    img_tmp = cv2.resize(img_tmp, (144, 144))
                    result = cv2.imwrite("/Users/joanna_frankiewicz/Desktop/Database/FaceDB_Emotions/result/"+element+"/"+filename, img_tmp)
                except:
                    print('Invalid shape of image')


emotions = ['angry', 'sad', 'surprised', 'happy']
images = load_images(emotions)
fitter = get_fitter(images)
face_detector = menpodetect.DlibDetector(dlib.simple_object_detector(HOG_PATH))
save_faces(emotions, face_detector, fitter)


