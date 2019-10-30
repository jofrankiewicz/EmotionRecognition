from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os 

datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')


emotions = ['sad', 'angry', 'happy', 'surprised']

def generate_images(emotions):
    for element in emotions:
        TEST_IMG_PATH = "/Users/joanna_frankiewicz/Desktop/Database/FaceDB_Emotions/result/"+element
        for filename in os.listdir(TEST_IMG_PATH):
            if filename.endswith(".jpg"):
                img = load_img(TEST_IMG_PATH+'/'+filename)  # this is a PIL image
                x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
                x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

                i = 0
                for batch in datagen.flow(x, batch_size=1, save_to_dir='/Users/joanna_frankiewicz/Desktop/Database/FaceDB_Emotions/modification/'+element, save_prefix=element, save_format='jpeg'):
                    i += 1
                    if i > 30:
                        break  # otherwise the generator would loop indefinitely