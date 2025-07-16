import matplotlib.pyplot as plt
import h5py
import numpy as np
import glob
from natsort import natsorted
import os 
import argparse
from wsi_core.WholeSlideImage import WholeSlideImage

parser = argparse.ArgumentParser()
parser.add_argument('--input-path',type=str, default='/ailab/group/pjlab-smarthealth03/transfers_cpfs_test/liumianxin/CLAM_DATA2/patches/', help='input for .h5')
parser.add_argument('--output-path',type=str,default='/ailab/user/liumianxin/transfer/AI-FFPE-main/', help='output for png')
args = parser.parse_args()
image_list = []
im_cntr1 = 0

im_cntr2_list = []

input_path = args.input_path
output_path = args.output_path

#input_path = "./path2svs/"
#output_path = "/path2png/"
output_path = os.path.join(output_path,"png_patches/testA/")
print(output_path)
if not os.path.exists(output_path):
    os.makedirs(output_path)
h5_counter = 0
exception_list = []

for filem in natsorted(glob.glob(input_path+"*.h5")):
    print("h5 count",h5_counter)
    h5_counter+=1
    # print(filem) 
    # try: 
    if h5_counter==2:
        png_cntr = 0      
        fname = filem.split('/')[-1]
        slidename = fname.split('.')[0]
        print(slidename)
        
        WSIpath = glob.glob("/ailab/public/pjlab-smarthealth03/liumianxin/HE/"+slidename+'.ndpi')
        if len(WSIpath)==0:
            WSIpath = glob.glob("/ailab/public/pjlab-smarthealth03/liumianxin/HE/"+slidename+'.svs')
        print(WSIpath[0])
        ext = WSIpath[0].split('.')[-1]
        if ext=='ndpi':
            patch_level = 2
        else:
            patch_level = 1
        WSI_object = WholeSlideImage(WSIpath[0])
        wsi = WSI_object.getOpenSlide()

        hdf = h5py.File(filem) 
        dset = hdf['coords']
        coords = dset[:]

        for i in coords:
            img = np.array(WSI_object.wsi.read_region(i, patch_level, (224,224)))
            plt.imsave(output_path+slidename+"_"+str(png_cntr) +".png",img)
            png_cntr+=1
            print(png_cntr)
    # except:
    #     exception_list.append(filem.split("/")[-1])
    #     print("Exception occured!!!")
    #     pass



#im_counter = 0 
#for image in sorted(glob.glob(filename_list+"/*")):
    #print(image.split("/")[-1])
    #if domain_type in image:
        #imagename = "/a"+str(im_counter)
        #shutil.copy(image,output_folder_name+"/"+image.split("/")[-1])
        #im_counter += 1
