from __future__ import print_function
import numpy as np
from PIL import Image,ImageDraw
import h5py

def render_strokes_as_image(allstrokes,box,filename,point_size):
    maxx = int(box[0])    # This box is the dimensions of the box enclosing the word from which this letter came from.
    minx = int(box[1])
    maxy = int(box[2])
    miny = int(box[3])
    width = maxx - minx + 15
    height = maxy - miny + 15
    img=Image.new("RGB",[width,height],color='white')     #Syntax : PIL.Image.new(mode, size, color=0)  mode- eg. RGB, size- tuple (width, height) colour- background colour. Returns an Image object
    draw=ImageDraw.Draw(img)
    double_size=point_size*2
    for s in allstrokes:
        for p in range(len(s)):
            x=s[p][0]-minx
            y=s[p][1]-miny
            seg_mark=s[p][2]               #These can be neglected. ##IGNORE.
            draw.ellipse(((x-point_size,y-point_size),(x+point_size,y+point_size)),fill='black')
            if (seg_mark == 2):
                draw.rectangle(((x - double_size, y - double_size), (x + double_size, y + double_size)), fill='green')    ##NOT REQUIRED.!!!!!
            
        draw.ellipse(((x - point_size, y - point_size), (x + point_size, y + point_size)), fill='red')    #This draws an ellipse in red colour marking the end of character.
    img.save(filename,"png")


def convert_online_to_offline(h5file,outputdir):
    f=h5py.File(h5file,"r")
    keys=list(f.keys())
    
    print(keys)
    
    totaldirs=len(keys)
    for d in range(totaldirs/10000):
        onesample=f.get(keys[d])
        samplename = onesample.name 
        print(samplename)
        print("Reading sample ",samplename)
        
        filename=outputdir+"/"+samplename+".png"
        nbstrokes=int(onesample.attrs["Nb_Strokes"])       # Number of strokes making one character.
        box=onesample.attrs["Global_Box"]
          
        allstrokes=[]
        for strk in range(nbstrokes):
            stroke=np.asarray(onesample.get("S"+str(strk)))
            allstrokes.append(stroke)
        #print(allstrokes)
        render_strokes_as_image(allstrokes,box,filename,4)         #point_size is just for making the box, a size of 4 gives neatly joined ellipses.
        #break
    f.close()
    
    

convert_online_to_offline("/home/cvpr/OHR data/Test_SS.h5","/home/cvpr/OHR data")


## A Stroke: The sequence of x-y coordinates recorded from the start of writing till the point the user lifts the pen
## Here, a stroke can be a character or a modifier or a part of.(in Bangla). A stroke is a sequence of x-y coordinates which represents the character.
## It is so possible that a character is formed by more than one stroke, in which case we append all of them, and make a list of strokes corr to a particular character.
## box : Gives the box enclosing the word from where the letter was taken.

## The dataset is in HDF5 format.





