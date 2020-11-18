import os, random, cv2
import numpy as np

'''
    A script for preparing and using the training dataset. 

'''

class dataHandler:

    def __init__(self):
        self.DATA_FP = os.path.abspath(os.path.join(os.curdir, 'data'))


    def get_covid_data_fp(self): # get covid data filepaths
        
        covid_dir = os.path.join(self.DATA_FP, 'CT_COVID')
        fp_list = [os.path.join(covid_dir, fp) for fp in os.listdir(covid_dir)]
        
        return fp_list


    def get_non_covid_data_fp(self):
        
        non_covid_dir = os.path.join(self.DATA_FP, 'CT_NonCOVID')
        fp_list = [os.path.join(non_covid_dir, fp) for fp in os.listdir(non_covid_dir)]

        return fp_list
    
    def get_all_data_labeled(self, shuffle=False):

        dataset = [ (fp, 1.0) for fp in self.get_covid_data_fp()]
        dataset += [ (fp, 0.0) for fp in self.get_non_covid_data_fp()]

        if shuffle:
            random.shuffle(dataset)

        return dataset

    def load_image(self, fp, resize=False, grayscale=False, add_noise=False, randomly_rotate=False):
        
        fp = os.path.abspath(fp) # normalize the path

        if not os.path.isfile(fp):
            print("Could not find filepath at location: ", fp )


        img = cv2.imread(fp)

        if add_noise:
            noise_mask = np.random.normal(0, .1, img.shape)
            img = img + noise_mask
           
        if randomly_rotate:
            # randomly rotate the image 90degrees 0 to 3 times. 
            times_to_rotates = range(0,4)
            img = np.rot90(img)

        if resize:
            img = cv2.resize(img, (299,299), interpolation=cv2.INTER_CUBIC)

        if grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = np.reshape(img, (img.shape[0], img.shape[1]))

        return img