import numpy as np
import h5py
from sklearn.mixture import GaussianMixture as GMM
import pandas as pd
from PIL import Image,ImageDraw
import cv2
lap = 0
"""
find_equidistant will get an HDF5 file as input. The file has many groups, from each group we have to get the strokes. For each stroke, we need to get the equidistant points and store back in an HDF format itself, number of points = 20.
"""
"""def get_results_from_HDF(h5file):  #This function takes the HDF file output from the find_equidistant(h5file) function and extracts data in the required form and must be given to return_list_for_csv_file(feature_array) so as to store the data in CSV format.
	f = h5py.File(h5file, 'r')
	keys=f.keys()
	#print keys
	total=len(keys)
	#print("Total ",total," keys")
	
	feature_array = []
	
	for t in range(total):
	
		group=f.get(keys[t])
		features=np.array(group)
		#features=group
		feature_array.append(features)
		
		
	return feature_array	

def return_list_for_csv_file(feature_array):   #CSV Format returned.
	listA = []
	for i in range(len(feature_array)):
		for j in range(len(feature_array[i])):
	
			listA.append(feature_array[i][j])
			
	return listA"""


"""def find_min(feature_array):
	minimum = 1000000000
	#print len(feature_array)
	for i in range(len(feature_array)):
		if(len(feature_array[i]) <minimum):
			minimum = len(feature_array[i])
			#if(minimum == 0):
				#print i
			
	return minimum"""  ## This function was used to find the set with the min number of elements in it.


def render_strokes_as_image(allstrokes,box,filename,point_size):
    maxx = int(box[0])    # This box is the dimensions of the box enclosing the word from which this letter came from.
    minx = int(box[1])
    maxy = int(box[2])
    miny = int(box[3])
    width = maxx - minx + 15
    height = maxy - miny + 15
    img=Image.new("RGB",[width,height],color='white')     #Syntax : PIL.Image.new(mode, size, color=0)  mode- eg. RGB, size- tuple (width, height) colour- background colour. Returns an Image object
    #print 'bla1'
    draw=ImageDraw.Draw(img)
    double_size=point_size*2
    for s in allstrokes:
    	#print 'bla'
    	try:
    		l = len(s)
    	except:
    		continue
    		
        for p in range(l):
            x=s[p][0]-minx
            y=s[p][1]-miny
            seg_mark=s[p][2]               #These can be neglected. ##IGNORE.
            draw.ellipse(((x-point_size,y-point_size),(x+point_size,y+point_size)),fill='black')
            if (seg_mark == 2):
                draw.rectangle(((x - double_size, y - double_size), (x + double_size, y + double_size)), fill='green')    ##NOT REQUIRED.!!!!!
            
        draw.ellipse(((x - point_size, y - point_size), (x + point_size, y + point_size)), fill='red')    #This draws an ellipse in red colour marking the end of character.
        #print 'end'
    img.save(filename,"png")
    
    
    return img
    
    

def equal_distance(len):
	if(len < 10):
		return len
		
	return len/10
	
def sqrt(n):
	return n**(1/2.0)
	
def find_features(equidistant):
	sine = []
	global lap
	
	if(len(equidistant) != 0):
		for lis in equidistant[0]:
			dis = float(sqrt((lis[1][0]-lis[0][0])**2 + (lis[2][0]-lis[1][0])**2))
			if dis== 0:
				lap+=1
			#if  (float((lis[1][0]-lis[0][0])) != 0) and (float((lis[2][0]-lis[1][0])) !=0):  ## Losing the straight line features!!!
			if(dis !=0):
				#m1 = (lis[1][1]-lis[0][1])/float((lis[1][0]-lis[0][0]))
				#m2 = (lis[2][1]-lis[1][1])/float((lis[2][0]-lis[1][0]))
				m1 = (lis[1][1]-lis[0][1])/float(dis)
				m2 = (lis[2][1]-lis[1][1])/float(dis)
				sine.append([m1,m2])
		
	#print 'length of sine =' 
	#print len(sine)
	
	return sine
	
	
	
def compute_angle(equidistant):
	tangents = []
	
	if(len(equidistant) != 0):
		for lis in equidistant[0]:
			if  (float((lis[1][0]-lis[0][0])) != 0) and (float((lis[2][0]-lis[1][0])) !=0):  ## Losing the straight line features!!!
				m1 = (lis[1][1]-lis[0][1])/float((lis[1][0]-lis[0][0]))
				m2 = (lis[2][1]-lis[1][1])/float((lis[2][0]-lis[1][0]))
				tangents.append([m1,m2])
		
	return tangents

def find_equidistant(h5file):
	f = h5py.File(h5file, 'r')
	keys = f.keys()
	total=len(keys)   #Number of character samples in the file.
	print total/2
	
	hf1 = h5py.File('Features_Extracted.h5', 'w')
	#print hf1.name
	file_names = []
	
	for t in range(total/2):
		group=f.get(keys[t])
		lapse =0

		#print group.name

		num =(int)(group.attrs['Nb_Strokes'])    ##Gives the number of strokes that make up the character
	
		equidistant = []    #List which will store all the equidistant points and their corr adj points also.
		for i in range(num):
			stroke_name = 'S'+str(i)
			features=np.asarray(group.get(stroke_name))
			try:
				l = len(features)
				lapse+=1
				#print l
			except:
				continue
				
			e = equal_distance(l)
			p = 10   # considering only 10 points from entire set, all are equidistant
			
			list_equal = []   #List which will store the point and its adjacent points;
			
			j =1            #As we are using j-1 in computation;
			
			while j+1<l and p>0:
				list_equal.append([features[j-1],features[j],features[j+1]])
				j+=e
				p-=1	
							
			equidistant.append(list_equal)	#contains for all strokes making the character.	
			
		#tangents = compute_angle(equidistant)
		tangents = find_features(equidistant)
		
		if(len(tangents) >= 10):
			name = group.name
			file_names.append(name)		
		
			hf1.create_dataset(name, data=tangents)
		
	#return hf1.name           -- returns the folder name which is : '/'
	print 'No of elements = '
	print len(file_names)
	print 'lapse'
	print lapse
	print 'laps'
	print lap
	return file_names
		

	
def GMM_results(n_clusters, file_names):     #Applies GMM library to the data, and prints the labels.

	data = pd.read_csv('feature_array.csv')
	X = data.iloc[:,1:].values
	
	gmm1 = GMM(n_components = n_clusters, init_params = 'kmeans', max_iter = 5000, covariance_type = 'diag').fit(X)
	label_gmm_1 = gmm1.predict(X)
	
	"""for i in range(len(label_gmm_1)):
		print label_gmm_1[i]        """         #Use for printing
		
	label_list =[]
	
	for i in range(len(label_gmm_1)):
		label_list.append([file_names[i], label_gmm_1[i]])
			
	return label_list

	
def flatten_data(h5file):
	f = h5py.File(h5file, 'r')
	keys = f.keys()
	total=len(keys)
	
	listB = []
	
	for t in range(total):
		group=f.get(keys[t])
		features=np.asarray(group)
		features = features.flatten('F')  #'F' means it flattens column by column [[1,2],[3,4]] flattened as [1,3,2,4]
		#print features
		listB.append(features)
		
	return listB

def group_clusters(clusters_names,n):
	#print 'called group_clusters'

	hf1 = h5py.File('/home/cvpr/OHR data/Cluster_Groups.h5', 'w')
	hf2 = h5py.File('/home/cvpr/OHR data/Features_Extracted.h5', 'r')
	hf3 = h5py.File("/home/cvpr/OHR data/Test_SS.h5", 'r')
	hf4 = h5py.File('/home/cvpr/OHR data/Image_Groups_64.h5', 'w')
	
	for i in range(n):
		group_name = 'group'+str(i)
		g = 'g'+ str(i)
		
		g = hf1.create_group(group_name)
		
	for i in range(n):
		group_name = 'group'+str(i)
		g = 'g'+ str(i)
		
		g = hf4.create_group(group_name)	
		
	"""group_name = 'group'+str(1)
	arr = np.array(hf2.get(clusters_names[1][0]))
	hf1.get(group_name).create_dataset('data1',data=arr)"""
	p =1
		
	for i in range(len(clusters_names)):
		
		label = clusters_names[i][1]
		group_name = 'group'+str(label)
		arr = np.array(hf2.get(clusters_names[i][0]))
		#file_name = clusters_names[i][0]
		data_name = 'data' + '_' + str(p) + '_' + str(label)
		hf1.get(group_name).create_dataset(data_name,data=arr)
		p+=1
		 
		img, filename = test(clusters_names,i)
		img_array = array(img, filename)
		data_name_image = 'data' + '_' + str(p) + '_' + str(label)+ '.png'
		hf4.get(group_name).create_dataset(data_name_image,data=img_array)
		
		
#############################################################################################################		
def test(clusters_names, i):
	#print 'called'
	hf = h5py.File("/home/cvpr/OHR data/Test_SS.h5", 'r')
	group_name = hf.get(clusters_names[i][0])
	#print group_name.name
	#print 'blah'
	nbstrokes=int(group_name.attrs["Nb_Strokes"])       # Number of strokes making one character.
        box=group_name.attrs["Global_Box"]
        
        allstrokes=[]
        for strk in range(nbstrokes):
            stroke=np.asarray(group_name.get("S"+str(strk)))
            allstrokes.append(stroke)
        #print(allstrokes)
        filename = '/home/cvpr/OHR data/images_Test/' + group_name.name+ '.png'
        #print filename
        img = render_strokes_as_image(allstrokes,box,filename,4)         #point_size is just for making the box, a size of 4 gives neatly joined ellipses.
        #break
	hf.close()
	return [img, filename]
	
	
def array(img, filename):
	#print 'array called'
	#path = '/home/cvpr/OHR data/images_Test' + filename
	img_r = cv2.imread(filename)
	img_r = np.asarray(img_r)
	#print image
	#print image.shape
	return img_r
	

	



#def put_into_same_clusters(label_gmm_1):
	
	
#print feature_array[0].tolist()

file_names = find_equidistant("/home/cvpr/OHR data/Test_SS.h5")       ##This function creates the h5file containing the slope feature extracted from equidistant points
#file_name = "/home/cvpr/OHR data/" + hf1+ '.h5'
#feature_array = get_results_from_HDF('/home/cvpr/OHR data/Features_Extracted.h5')
#print feature_array

#minimum = find_min(feature_array)
#print minimum

"""listA = return_list_for_csv_file(feature_array)
			
df= pd.DataFrame(listA)
df.to_csv("feature_array.csv", sep = ',')"""


flattened_data = flatten_data('/home/cvpr/OHR data/Features_Extracted.h5')    ##Flatten the dataset
df= pd.DataFrame(flattened_data)
df.to_csv("feature_array.csv", sep = ',')##store in csv, ready for use

#n = input('Enter the number of clusters')	
n = 64

clusters_names = GMM_results(n, file_names)

"""for i in range(len(clusters_names)):
	print clusters_names[i];"""	
	
#print len(clusters_names)

"""Now: we have extracted features from each stroke by taking 10 consecutive equidistant points, and flattened the input data, which will be fed to the GMM_results, to get the reults. Once this is done, we have to put each data in the dataset into an HDF5 file, where one attribute shall be the cluster number and the other would be the rendering of character. This has to be manually checked to see how the performance is"""
"""Another way is to group the clusters, each group forms a folder in HDF5 file, so there shall be n folders in the file, which shall be rendered to see effects"""	


group_clusters(clusters_names,n)
		
#img = test(clusters_names, 1)		
#array(img)

		
		
