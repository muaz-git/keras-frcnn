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
import json
from pprint import pprint

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


def parseJSON(path):
    with open(path) as data_file:
        data = json.load(data_file)

    # pprint(data)
    # pprint(data['filename'])
    return data

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


def drawGTJSON(img_rgb, annotationObj):
    bboxList = annotationObj['object']['bboxes']
    for box in bboxList:
        cv2.rectangle(img_rgb, (box['xmin'], box['ymin']), (box['xmax'], box['ymax']),
                      gt_color, 3)

def drawGTXML(img_rgb, annotationObj):
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


def calculateIOU(boxA, boxB, ratio):
    # determine the (x, y)-coordinates of the intersection rectangle
    print 'boxA ', boxA
    print 'boxB ', boxB
    (real_x1, real_y1, real_x2, real_y2) = frcnnObj.get_real_coordinates(ratio, boxB[0], boxB[1], boxB[2], boxB[3])
    xA = max(boxA['xmin'],real_x1)
    yA = max(boxA['ymin'], real_y1)
    xB = min(boxA['xmax'], real_x2)
    yB = min(boxA['ymax'], real_y2)

    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA['xmax'] - boxA['xmin'] + 1) * (boxA['ymax'] - boxA['ymin'] + 1)
    boxBArea = (real_x2 - real_x1 + 1) * (real_y2 - real_y1 + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def getBestIOUwithGT(predictionBox, gtBoxList, ratio):
    bestIndex = None
    bestIOU = -1
    mylist = []
    for gtBox, idxx in zip(gtBoxList, range(len(gtBoxList))):
        iou = calculateIOU(gtBox, predictionBox, ratio)
        mylist.append(iou)
    return mylist

def evaluateBBoxes(predictionBoxesDict, bboxList, ratio):
    predictionBoxesList = predictionBoxesDict['Face']
    # bboxList = gtAnnotationObj['object']['bboxes']

    # for predictionBox, idx in zip(predictionBoxesList, range(len(predictionBoxesList))):
    for predictionBox in predictionBoxesList:
        getBestIOUwithGT(predictionBox, bboxList, ratio)
    pass


def mainProcess():
    annotationFolder = './FDDBdevkit/Annotations/'
    imgsFolder = './FDDBdevkit/JPEGImages/'
    filesNameList = getListOfFiles(annotationFolder)

    idx = 0
    while idx<1:
        xmlFileName = filesNameList[idx]
        (fileBaseName, fileExt) = os.path.splitext(os.path.basename(xmlFileName))
        img_path = imgsFolder + fileBaseName + '.jpg'
        annotation_path = annotationFolder + xmlFileName

        # annotationObj = parseXML(annotation_path)
        annotationObj = parseJSON(annotation_path)
        # exit()
        # img = loadImg(img_path)
        img = loadImg('./1.jpg')
        ratio, probs, bboxes = predictBBoxes(img)
        newBboxesDict, newProbsDict = NMS(probs, bboxes)

        # evaluateBBoxes(newBboxesDict, annotationObj['object']['bboxes'], ratio)

        drawPredBBoxes(img, newBboxesDict, newProbsDict, ratio)
        # drawGTJSON(img, annotationObj)


        # print type(newBboxesDict)
        # print '\n\n'
        # pprint(annotationObj)
        print 'writing ' + fileBaseName + '.jpg'
        cv2.imwrite('./results_imgs/' + fileBaseName + '.jpg', img)
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
    drawGTJSON(img, annotationObj)
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
