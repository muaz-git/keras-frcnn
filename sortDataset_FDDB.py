import cv2
import numpy as np
import os
from xml.etree.ElementTree import Element, SubElement
from xml.etree import ElementTree
from xml.dom import minidom
import xml.etree.cElementTree as ET
from shutil import move, copy, rmtree
import json
import math

# create folder name FDDBdevkit and move folders FDDB-folds and originalPics in to it.
rootDir = './FDDBdevkit/'
annotDir = rootDir + 'Annotations/'
imageSetsDir = rootDir + 'ImageSets/Main/'
JPEGImagesDir = rootDir + 'JPEGImages/'

numberOfBBoxes = 0

def createFolders():
    if not os.path.exists(annotDir):
        os.makedirs(annotDir)
    if not os.path.exists(imageSetsDir):
        os.makedirs(imageSetsDir)
    if not os.path.exists(JPEGImagesDir):
        os.makedirs(JPEGImagesDir)


def deleteFolders():
    if os.path.exists(annotDir):
        rmtree(annotDir)
    if os.path.exists(imageSetsDir):
        rmtree(imageSetsDir)
    if os.path.exists(JPEGImagesDir):
        rmtree(JPEGImagesDir)


def getDimensionsOfImg2(fullFilePath):
    fullFilePath = rootDir + 'originalPics/' + fullFilePath
    img = cv2.imread(fullFilePath)
    (imgHeight, imgWidth, imgChannels) = np.shape(img)

    return int(imgChannels), int(imgHeight), int(imgWidth)


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


def caclulateBbox(annotObj, ellipseObj):
    a = ellipseObj['major_axis_radius']
    b = ellipseObj['minor_axis_radius']
    phi = ellipseObj['angle']
    cx = ellipseObj['center_x']
    cy = ellipseObj['center_y']
    tmp_img = np.zeros((annotObj['size']['height'], annotObj['size']['width']), dtype=np.uint8)

    cv2.ellipse(tmp_img,
                (int(cx), int(cy)), (
                    int(a),
                    int(b)),
                math.degrees(phi), 0, 360, 255, -1)
    image, contours, hierarchy = cv2.findContours(tmp_img, 1, 2)
    try:
        cnt = contours[0]
    except IndexError:
        return -1

    x, y, w, h = cv2.boundingRect(cnt)

    bbox = {}



    c = math.sqrt((b**2) * (math.sin(phi)**2) + (a**2) + (math.cos(phi)**2))
    d = math.sqrt((a**2) * (math.sin(phi)**2) + (b**2) + (math.cos(phi)**2))

    bbox['xmin'] = int(x)
    bbox['xmax'] = int(x+w)
    bbox['ymin'] = int(y)
    bbox['ymax'] = int(y+h)

    return bbox

def parseObjDetails2(annotObj, ln):
    gtArray = ln.split()
    # objObj = {}
    # objObj['name'] = 'Face'

    ellipseObj = {}
    ellipseObj['major_axis_radius'] = float(gtArray[0])
    ellipseObj['minor_axis_radius'] = float(gtArray[1])
    ellipseObj['angle'] = float(gtArray[2])
    ellipseObj['center_x'] = float(gtArray[3])
    ellipseObj['center_y'] = float(gtArray[4])

    if caclulateBbox(annotObj, ellipseObj) == -1:
        return annotObj

    bndBoxObj = caclulateBbox(annotObj, ellipseObj)

    # objObj['bndbox'] = bndBoxObj
    # objObj['ellipse'] = ellipseObj
    # annotObj['object'] = objObj

    annotObj['object']['ellipses'].append(ellipseObj)
    annotObj['object']['bboxes'].append(bndBoxObj)

    return annotObj


def parseObjInit2(ln):
    annotation = {}
    fullFilePath = ln + '.jpg'
    print 'Init ' + fullFilePath
    fileName = os.path.basename(fullFilePath)

    foldername = os.path.dirname(fullFilePath)
    annotation['foldername'] = foldername
    annotation['origFileName'] = fileName

    foldername = foldername.replace('/','_')
    annotation['filename'] = os.path.splitext(fileName)[0] + '_' + foldername

    depth, height, width = getDimensionsOfImg2(fullFilePath)

    sizeObj = {}
    sizeObj['width'] = width
    sizeObj['height'] = height
    sizeObj['depth'] = depth
    annotation['size'] = sizeObj
    annotation['object'] = {}
    annotation['object']['bboxes'] = []
    annotation['object']['ellipses'] = []
    annotation['object']['name'] = 'Face'
    return annotation


def saveEverything2(dictObj):
    srcFileName = dictObj['origFileName']
    dstFileName = dictObj['filename']

    foldername = dictObj['foldername']

    annotationFilePath = annotDir + dstFileName + '.json'

    with open(annotationFilePath, 'w') as fp:
        json.dump(dictObj, fp)

    srcFilePath = rootDir + 'originalPics/' + foldername + '/' + srcFileName
    dstFilePath = JPEGImagesDir + dstFileName + '.jpg'

    copy(srcFilePath, dstFilePath)


def detectLineType(ln):
    if len(ln) == 1 or len(ln) == 2:  # contains only number of ellipses
        return 1
    if len([c for c in ln if c == '/']) == 4:
        return 0
    return 2


def parsefoldFile(path, myFlag):

    count = 0
    annotation = None
    with open(path) as data_file:
        for line in data_file:

            line = line.rstrip()
            typ = detectLineType(line)  # 0 : imagePath, 1 : number of faces, 2 : contains ellipse detail

            if typ == 0:
                if not(annotation == None): # if loop is not here for the first time
                    # save json file before creating new
                    saveEverything2(annotation)
                    with open(imageSetsDir + myFlag+".txt", "a") as myfile:
                        myfile.write(annotation['filename'] + '\n')



                annotation = parseObjInit2(line)


                count += 1
            elif typ == 1:  # just to skip the line after the path

                continue
            elif typ == 2:
                global numberOfBBoxes
                numberOfBBoxes +=1
                parseObjDetails2(annotation, line)

        saveEverything2(annotation)
        with open(imageSetsDir + "train.txt", "a") as myfile:
            myfile.write(annotation['filename'] + '\n')


deleteFolders()
createFolders()

for i in range(1,11):
    if i < 10:
        filePath = rootDir + 'FDDB-folds/FDDB-fold-0'+str(i)+'-ellipseList.txt'

    else:
        filePath = rootDir + 'FDDB-folds/FDDB-fold-10-ellipseList.txt'

    print '***********************************************'
    print filePath
    if i <8:
        flag = 'train'
    else:
        flag = 'val'
    parsefoldFile(filePath, flag)
    # break
print 'total bboxes = ' + str(numberOfBBoxes)
