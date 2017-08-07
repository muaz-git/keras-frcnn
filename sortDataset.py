import cv2
import numpy as np
import os
from xml.etree.ElementTree import Element, SubElement
from xml.etree import ElementTree
from xml.dom import minidom
import xml.etree.cElementTree as ET
from shutil import move


# create folder name WIDERdevkit and move folders WIDER_train, WIDER_val, wider_face_split and WIDER_test in to it.
rootDir = './WIDERdevkit/'
annotDir = rootDir + 'Annotations/'
imageSetsDir = rootDir + 'ImageSets/Main/'
JPEGImagesDir = rootDir + 'JPEGImages/'

def createFolders():
    if not os.path.exists(annotDir):
        os.makedirs(annotDir)
    if not os.path.exists(imageSetsDir):
        os.makedirs(imageSetsDir)
    if not os.path.exists(JPEGImagesDir):
        os.makedirs(JPEGImagesDir)

def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ElementTree.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

def getDimensionsOfImg(subfoldername, filename, f_type):
    imagePath = (rootDir+'WIDER_'+f_type+'/images/'+subfoldername+'/')+filename.rstrip()

    img = cv2.imread(imagePath)
    (imgHeight, imgWidth, imgChannels) = np.shape(img)

    return str(imgChannels), str(imgHeight), str(imgWidth)

def parseObjDetails(xmlObj, ln): # parse line which contain details of face object
    gtArray = ln.split()
    object = SubElement(xmlObj, 'object')

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

def parseObjInit(ln, f_type): # initialize root of xml object with basic properties
    annotation = Element('annotation')
    subfoldername = SubElement(annotation, 'subfoldername')
    filename = SubElement(annotation, 'filename')

    splittedLine = ln.split('/')
    subfoldername.text = splittedLine[0].rstrip()
    filename.text = splittedLine[1].rstrip()

    depth, height, width = getDimensionsOfImg(subfoldername.text, filename.text, f_type)

    size = SubElement(annotation, 'size')
    depth_elem = SubElement(size, 'depth')
    depth_elem.text = depth
    height_elem = SubElement(size, 'height')
    height_elem.text = height
    width_elem = SubElement(size, 'width')
    width_elem.text = width

    return annotation

def saveEverything(xmlObj, fileType):
    fullFilename = xmlObj.find('filename').text
    subFolderName = xmlObj.find('subfoldername').text

    filename_ = os.path.splitext(os.path.basename(fullFilename))[0]

    tree = ET.ElementTree(xmlObj)
    tree.write(annotDir+filename_+'.xml')

    if fileType == 'val':
        with open(imageSetsDir + "val.txt", "a") as myfile:
            myfile.write(filename_+'\n')
    else:
        with open(imageSetsDir + "train.txt", "a") as myfile:
            myfile.write(filename_+'\n')

    move((rootDir+'WIDER_'+fileType+'/images/'+subFolderName+'/')+filename_.rstrip()+'.jpg', JPEGImagesDir+filename_+'.jpg')



def parseGTTxtFile(path, f_type):
    count = 0
    flag = 0
    annotation = None
    with open(path) as data_file:
        for line in data_file:
            if '.jpg' in line:
                if not(annotation == None): # if loop is not here for the first time
                    # save xml file before creating new
                    # print prettify(annotation)
                    # print filename_
                    saveEverything(annotation, f_type)

                annotation = parseObjInit(line, f_type)


                flag = 1
                count += 1
            else: # if line contains object information
                if flag == 1:  # just to skip the line after the path
                    flag = 0
                    continue
                else:
                    parseObjDetails(annotation, line)


                    # if count > 10:
                    #     break

        fullFilename = annotation.find('filename').text
        subFolderName = annotation.find('subfoldername').text
        filename_ = os.path.splitext(os.path.basename(fullFilename))[0]

        tree = ET.ElementTree(annotation)
        tree.write(annotDir + filename_ + '.xml')

        if f_type == 'val':
            with open(imageSetsDir + "val.txt", "a") as myfile:
                myfile.write(filename_ + '\n')
        else:
            with open(imageSetsDir + "train.txt", "a") as myfile:
                myfile.write(filename_ + '\n')

        move((rootDir + 'WIDER_' + f_type + '/images/' + subFolderName + '/') + filename_.rstrip() + '.jpg',
             JPEGImagesDir + filename_ + '.jpg')

def parseTestFile():
    testFilePath = './WIDERdevkit/wider_face_split/wider_face_test_filelist.txt'
    with open(testFilePath) as data_file:
        for line in data_file:
            splittedLine = line.split('/')
            subFolderName = splittedLine[0].rstrip()
            fullFilename = splittedLine[1].rstrip()
            filename_ = os.path.splitext(os.path.basename(fullFilename))[0]
            # print subFolderName
            # print filename_
            with open(imageSetsDir + "test.txt", "a") as myfile:
                myfile.write(filename_ + '\n')
            move(rootDir + 'WIDER_test/images/' + subFolderName + '/' + filename_.rstrip() + '.jpg',
                 JPEGImagesDir + filename_ + '.jpg')
            # exit()

# moveFiles()

createFolders()
trainFilePath = './WIDERdevkit/wider_face_split/wider_face_train_bbx_gt.txt'
# trainFilePath = './WIDERdevkit/wider_face_split/train_tmp.txt'
folderType = 'train'
print folderType
parseGTTxtFile(trainFilePath, folderType)
trainFilePath = './WIDERdevkit/wider_face_split/wider_face_val_bbx_gt.txt'
# trainFilePath = './WIDERdevkit/wider_face_split/val_temp.txt'
folderType = 'val'
print folderType
parseGTTxtFile(trainFilePath, folderType)
#
folderType = 'test'
print folderType
parseTestFile()
