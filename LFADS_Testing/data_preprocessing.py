import cv2 as cv
import os
import numpy as np
 
directory = '/home/013057356/LFADS/LFADS_Testing/LFADS_dataset/TTK_data/TikTok_dataset2/TikTok_dataset/'
write_directory = './LFADS_dataset/quant_imgs/'

images = '/home/013057356/LFADS/LFADS/LFADS_Testing/Small_Demo_Dataset/Images/'
binarys = '/home/013057356/LFADS/LFADS/LFADS_Testing/Small_Demo_Dataset/Matting/'
matting = '/home/013057356/LFADS/LFADS/LFADS_Testing/Small_Demo_Dataset/2ndmatting/'
proper_images = '/home/013057356/LFADS/LFADS_Testing/LFADS_dataset/LFADS_Full_Human_Dataset/proper_images/'
proper_images_matting = '/home/013057356/LFADS/LFADS_Testing/LFADS_dataset/LFADS_Full_Human_Dataset/proper_images_matting/'
# iterate over files in
# that directory
def binary():
	for filename in os.scandir(directory):
    		if filename.is_file():
    			grayImage = cv.imread(filename.path)
    			for x in range(np.shape(grayImage)[0]):
    				for y in range(np.shape(grayImage)[1]):
    					if grayImage[x][y][0] != 0 and grayImage[x][y][1] != 0 and grayImage[x][y][2] != 0:
    						grayImage[x][y][0] = 255;
    						grayImage[x][y][1] = 255;
    						grayImage[x][y][2] = 255;
    		cv.imwrite('./Segmentation_dataset/1803290511/Binary_00000000/'+filename.name,grayImage)
    		print(filename.name)


def testing(directory,write_directory):

	for filename in os.scandir(directory):
		if filename.is_file():
			frame = []
			img = cv.imread(filename.path)
			Z = img.reshape((-1,3))
			Z = np.float32(Z)
			criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
			K = 8
			ret,label,center=cv.kmeans(Z,K,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)
			center = np.uint8(center)
			res = center[label.flatten()]
			res2 = res.reshape((img.shape))
			print(filename.name,'            ',np.shape(res2))
			cv.imwrite(write_directory+filename.name,res2)

def transfer(directory,binary_dir):
	count = 0
	for subdir, dirs, files in os.walk(directory):
	    for file in files:
	    	if os.path.isdir(subdir):
	    		# print(images_dir+os.path.split(os.path.split(subdir)[0])[1]+'_'+file)
        		dirname = os.path.basename(subdir)
        		if dirname == 'masks':
        			img = cv.imread(os.path.join(subdir, file))
        			print('masks:',binary_dir+os.path.split(os.path.split(subdir)[0])[1]+'_'+file,'            ',np.shape(img))
        			cv.imwrite(binary_dir+os.path.split(os.path.split(subdir)[0])[1]+'_'+file,img);
        		# if dirname == 'images':
        		# 	img = cv.imread(os.path.join(subdir, file))
        		# 	print('image:',os.path.split(os.path.split(subdir)[0])[1]+'_'+file,'            ',np.shape(img))
        		# 	cv.imwrite(binary_dir+os.path.split(os.path.split(subdir)[0])[1]+'_'+file,img);


def convert(binary_dir,images_dir,matting_dir):

	for filename in os.scandir(images_dir):
		if filename.is_file():
			for matting_filename in os.scandir(binary_dir):
				if matting_filename.is_file():
					if filename.name == matting_filename.name:
						img = cv.imread(filename.path)
						mask = cv.imread(matting_filename.path)
						mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
						print(matting_dir+filename.name)
						matting = cv.bitwise_and(img,img,mask = mask)
						cv.imwrite(matting_dir+filename.name,matting)


convert(binarys,images,matting)
