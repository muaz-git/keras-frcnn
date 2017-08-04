import json
import cv2
import numpy as np
import os
from xml.etree.ElementTree import Element, SubElement, Comment, tostring
from xml.etree import ElementTree
from xml.dom import minidom
import matplotlib.pyplot as plt
import scipy.ndimage
import scipy.misc

# create folder name WIDERdevkit and move folders WIDER_train, WIDER_val, wider_face_split and WIDER_test in to it.
def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ElementTree.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

def getDimensionsOfImg(subfoldername, filename, f_type):
    imagePath = ('./WIDERdevkit/WIDER_'+f_type+'/images/'+subfoldername+'/')+filename.rstrip()

    img = scipy.ndimage.imread(imagePath)
    print imagePath
    imgShape = np.shape(img)
    # scipy.misc.pilutil.imshow(img)
    return str(3), str(480), str(360)


def parseGTTxtFile(path, f_type):
    count = 0
    flag = 0
    annotation = None
    with open(path) as data_file:
        for line in data_file:
            if '.jpg' in line:
                if annotation == None: # to ignore first time
                    pass
                else:
                    # save xml file before creating new
                    # print prettify(annotation)
                    # print '\n\n'
                    pass
                annotation = Element('annotation')
                subfoldername = SubElement(annotation, 'subfoldername')
                filename = SubElement(annotation, 'filename')

                splittedLine = line.split('/')
                subfoldername.text = splittedLine[0].lstrip()
                filename.text = splittedLine[1].lstrip()


                depth, height, width = getDimensionsOfImg(subfoldername.text, filename.text, f_type)

                size = SubElement(annotation, 'size')
                depth_elem = SubElement(size, 'depth')
                depth_elem.text = depth
                height_elem = SubElement(size, 'height')
                height_elem.text = height
                width_elem = SubElement(size, 'width')
                width_elem.text = width


                # exit()
                flag = 1
                count += 1
            else:
                if flag == 1:  # just to skip the line after the path
                    flag = 0
                    continue
                else:
                    gtArray = line.split()
                    object = SubElement(annotation, 'object')

                    name = SubElement(object, 'name')
                    name.text = 'Face'

                    bndbox = SubElement(object, 'bndbox')
                    xmin = SubElement(bndbox, 'xmin')
                    xmin.text = str(gtArray[0].lstrip())

                    xmax = SubElement(bndbox, 'xmax')
                    xmax.text = str(int(gtArray[0].lstrip()) + int(gtArray[2].lstrip()))

                    ymin = SubElement(bndbox, 'ymin')
                    ymin.text = str(gtArray[1].lstrip())

                    ymax = SubElement(bndbox, 'ymax')
                    ymax.text = str(int(gtArray[1].lstrip()) + int(gtArray[3].lstrip()))

                    # blur, expression, illumination, invalid, occlusion, pose
                    blur = SubElement(object, 'blur')
                    blur.text = str(gtArray[4].lstrip())
                    expression = SubElement(object, 'expression')
                    expression.text = str(gtArray[5].lstrip())
                    illumination = SubElement(object, 'illumination')
                    illumination.text = str(gtArray[6].lstrip())
                    invalid = SubElement(object, 'invalid')
                    invalid.text = str(gtArray[7].lstrip())
                    occlusion = SubElement(object, 'occlusion')
                    occlusion.text = str(gtArray[8].lstrip())
                    pose = SubElement(object, 'pose')
                    pose.text = str(gtArray[9].lstrip())

            if count > 1:
                break

trainFilePath = './WIDERdevkit/wider_face_split/wider_face_train_bbx_gt.txt'
folderType = 'train'
parseGTTxtFile(trainFilePath, folderType)


