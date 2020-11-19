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
           
        if randomly_rotate:
            # randomly rotate the image 90degrees 0 to 3 times. 
            times_to_rotate = np.random.randint(0,4)
            img = np.rot90(img, times_to_rotate)
            
        if resize: # resize with padding
            resized_img = img.copy()
            # resize without padding
            #img = cv2.resize(img, (299,299), interpolation=cv2.INTER_AREA)
            
            # resize so longer dimension == 299.0
            
            # this won't completely preserve the aspect ration because the longer dimension is not necessarily divisible by 299,
            # but should lead to very close results
            
            #aspect_ratio = img.shape[1]/img.shape[0]

            # get which dimension is larger and resize based on that
            if img.shape[0] > img.shape[1]: # height is greater than width
                new_width = 299
                new_height = int(np.floor((299 * img.shape[1]) / img.shape[0]))
            else:
                new_height= 299
                new_width = int(np.floor((299 * img.shape[0]) / img.shape[1]))
            
            resized_img = cv2.resize(resized_img, (new_width, new_height), interpolation=cv2.INTER_AREA)
                
            # now add padding so both dimensions are 299
            
            pad_left = int((299 - resized_img.shape[1]) /2)
            pad_top = int((299 - resized_img.shape[0]) / 2)
            
            pad_right = int(299 - pad_left - resized_img.shape[1])
            pad_bot = int(299 - pad_top - resized_img.shape[0])
            
            img = cv2.copyMakeBorder(resized_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=0)
            
            
        if add_noise: # REFERENCE: https://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv
            img = np.float32(img) # convert to float first 
            mean = 0.0
            var = 1.0
            sigma = var**0.5
            gauss_noise = np.random.normal(mean, sigma,(img.shape[0], img.shape[1], img.shape[2]))
            gauss_noise = np.reshape(gauss_noise, (img.shape[0],img.shape[1],img.shape[2]))
            img = img + gauss_noise
            
            img = cv2.cvtColor(img.astype('float32'), cv2.COLOR_RGB2BGR) # convert back to bgr
            
            #cv2.randn(gaussian_noise, (0,0,0),(10,10,10))
            #img = img + gaussian_noise
            
        if grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = np.reshape(img, (img.shape[0], img.shape[1]))
            

        
        if len(img.shape) == 2: # only x and y dimensions
            #img = np.resize(img, (img.shape[0], img.shape[1], 3))
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        
        return img