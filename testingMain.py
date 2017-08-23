import cv2
from test_FasterRCNN import test_FasterRCNN
import numpy as np
from keras_frcnn import roi_helpers
import xml.etree.ElementTree as ET
from xml.etree import ElementTree
from xml.dom import minidom
import os.path
from os import listdir
from os.path import isfile, join

prediction_color = (255, 0, 0) # bgr
gt_color = (0, 255, 0) # bgr
frcnnObj = test_FasterRCNN()

def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ElementTree.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def loadImg(i_path):
    img_rgb = cv2.imread(i_path)
    return img_rgb


def predictBBoxes(img_rgb):

    # ratio, probs, bboxes = frcnnObj.getBBoxes(img_rgb)
    return frcnnObj.getBBoxes(img_rgb)


def NMS(probs, bboxes):
    newBboxesDict = {}
    newProbsDict = {}
    for key in bboxes:
        bbox = np.array(bboxes[key])

        new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.5)
        newBboxesDict[key] = new_boxes
        newProbsDict[key] = new_probs
    return newBboxesDict, newProbsDict


def getTPAndFP(bboxes):
    pass


def parseXML(path):

    tree = ET.parse(path)
    root = tree.getroot()

    annotationObj = {}
    annotationObj['subfoldername'] = root.findall('subfoldername')[0].text
    annotationObj['filename'] = root.findall('filename')[0].text
    sizes = root.findall('size')[0]
    sizesObj = {}
    for node in sizes.getiterator():
        if not (node.tag == 'size'):
            sizesObj[node.tag] = node.text
    annotationObj['size'] = sizesObj

    objects = root.findall('object')
    objList = []
    for obj in objects:
        myObj = {}

        myObj['name'] = obj.findall('name')[0].text
        myObj['blur'] = obj.findall('blur')[0].text
        myObj['expression'] = obj.findall('expression')[0].text
        myObj['illumination'] = obj.findall('illumination')[0].text
        myObj['invalid'] = obj.findall('invalid')[0].text
        myObj['occlusion'] = obj.findall('occlusion')[0].text
        myObj['pose'] = obj.findall('pose')[0].text

        bbox = obj.findall('bndbox')[0]

        bBoxObj = {}

        for node in bbox.getiterator():
            if not(node.tag == 'bndbox'):
                bBoxObj[node.tag] = int(node.text)

                #print node.tag, node.attrib, node.text
        myObj['bndbox'] = bBoxObj

        objList.append(myObj)

        # exit()
    annotationObj['objList'] = objList
    return annotationObj


def drawPredBBoxes(img_rgb, bboxes, probsDict, ratio):
    for k in bboxes:
        # # print bboxes[k]
        for jk in range(len(bboxes[k])):
            # print jk
            (x1, y1, x2, y2) = bboxes[k][jk, :]
            (real_x1, real_y1, real_x2, real_y2) = frcnnObj.get_real_coordinates(ratio, x1, y1, x2, y2)
            cv2.rectangle(img_rgb, (real_x1, real_y1), (real_x2, real_y2),
                          #(int(frcnnObj.class_to_color[k][0]), int(frcnnObj.class_to_color[k][1]), int(frcnnObj.class_to_color[k][2])), 2)
                          prediction_color, 2)


def drawGT(img_rgb, annotationObj):
    objList = annotationObj['objList']
    for obj in objList:
        bndbox = obj['bndbox']
        # print bndbox['xmin']
        # print type(bndbox['xmin'])
        cv2.rectangle(img_rgb, (bndbox['xmin'], bndbox['ymin']), (bndbox['xmax'], bndbox['ymax']),
                      gt_color, 3)
    pass

def getListOfFiles(folderName):
    filesNameList = [f for f in listdir(folderName) if isfile(join(folderName, f))]
    return filesNameList


def mainProcess():
    annotationFolder = './WIDERdevkit/Annotations/'
    imgsFolder = './WIDERdevkit/JPEGImages/'
    filesNameList = getListOfFiles(annotationFolder)

    idx = 0
    while idx<100:
        xmlFileName = filesNameList[idx]
        (fileBaseName, fileExt) = os.path.splitext(os.path.basename(xmlFileName))
        img_path = imgsFolder + fileBaseName + '.jpg'
        annotation_path = annotationFolder + xmlFileName

        annotationObj = parseXML(annotation_path)
        img = loadImg(img_path)
        ratio, probs, bboxes = predictBBoxes(img)
        newBboxesDict, newProbsDict = NMS(probs, bboxes)
        drawPredBBoxes(img, newBboxesDict, newProbsDict, ratio)
        drawGT(img, annotationObj)

        print 'writing ' + fileBaseName + '.jpg'
        cv2.imwrite('./results_imgs_new/' + fileBaseName + '.jpg', img)
        idx +=1




if __name__ == "__main__":
    mainProcess()
    exit()
    img_path = './WIDERdevkit/JPEGImages/0_Parade_Parade_0_883.jpg'
    # annotation_path = './WIDERdevkit/Annotations/0_Parade_Parade_0_883.xml'
    annotation_path = './WIDERdevkit/Annotations/0_Parade_Parade_0_883.xml'
    #annotation_path = './WIDERdevkit/Annotations/0_Parade_Parade_0_883.xml'
    if not(os.path.isfile(annotation_path)):
        print 'pass'
        exit()
    else:
        annotationObj = parseXML(annotation_path)

    img = loadImg(img_path)

    mydict = np.load('temp.npy')

    ratio = mydict[()]['ratio']
    probs = mydict[()]['probs']
    bboxes = mydict[()]['bboxes']


    # ratio, probs, bboxes = predictBBoxes(img)
    newBboxesDict, newProbsDict = NMS(probs, bboxes)
    drawPredBBoxes(img, newBboxesDict, newProbsDict)
    drawGT(img, annotationObj)
    # mydict = {}
    # mydict['ratio'] = ratio
    # mydict['probs'] = probs
    # mydict['bboxes'] = bboxes
    #
    # np.save('temp.npy', mydict)



    img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
    cv2.imshow('img', img)
    cv2.waitKey()
    # TPs, FPs = getTPAndFP(bboxes)
