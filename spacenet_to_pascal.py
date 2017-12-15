#!/usr/bin/env python

import cv2, pdb, json, os, shutil, re
import xml.etree.cElementTree as ET

def to_pascal(spec, directory, type):
    if not os.path.exists(os.path.join('spacenet', type, 'images')):
        os.makedirs(os.path.join('spacenet', type, 'images'))
        os.makedirs(os.path.join('spacenet', type, 'annotations'))
    for anno in spec:
        if len(anno['rects']) == 0:
            continue
        shutil.copy(
            os.path.join(directory, anno['image_path']), 
            os.path.join('spacenet', type, 'images', os.path.basename(anno['image_path']))
        )

        shape = cv2.imread(os.path.join(directory, anno['image_path'])).shape

        root = ET.Element('annotation')
        ET.SubElement(root, 'folder').text = 'SpaceNet'
        ET.SubElement(root, 'filename').text = os.path.basename(anno['image_path'])
        size = ET.SubElement(root, 'size')
        ET.SubElement(size, 'height').text = str(shape[0])
        ET.SubElement(size, 'width').text = str(shape[1])
        ET.SubElement(size, 'depth').text = str(shape[2])
        ET.SubElement(root, 'segmented').text = '0'
        for r in anno['rects']:
            obj = ET.SubElement(root, 'object')
            ET.SubElement(obj, 'name').text = 'building'
            ET.SubElement(obj, 'pose').text = 'Unspecified'
            ET.SubElement(obj, 'truncated').text = '0'
            ET.SubElement(obj, 'difficult').text = '0'
            bbox = ET.SubElement(obj, 'bndbox')
            ET.SubElement(bbox, 'xmin').text = str(int(r['x1']))
            ET.SubElement(bbox, 'ymin').text = str(int(r['y1']))
            ET.SubElement(bbox, 'xmax').text = str(int(r['x2']))
            ET.SubElement(bbox, 'ymax').text = str(int(r['y2']))
        tree = ET.ElementTree(root)
        xmlfile = os.path.splitext(os.path.basename(anno['image_path']))[0] + '.xml'
        tree.write(os.path.join('spacenet', type, 'annotations', xmlfile))

def even(spec):
    sets = {}
    for anno in spec:
        t = re.search('([^\d]*)\d+.jpg', anno['image_path']).group(1)
        if t not in sets:
            sets[t] = [anno['image_path']]
        else:
            sets[t].append(anno['image_path'])

    result = []
    max_len = max([len(sets[k]) for k in sets.keys()])
    for k in sets.keys():
        count, iters = 0, 0
        while count < max_len and iters < 10:
            result.extend(sets[k][0:(max_len - count)])
            count += len(sets[k])
            iters += 1
    return result

if __name__ == '__main__':
    spec = json.load(open('../data/train_data_2017-11-14T10:08:58.650463.json')) + json.load(open('../data/rio_train.json'))
    spec = even(spec)
    to_pascal(spec, '../data', 'train')


    spec = json.load(open('../data/val_data_2017-11-14T10:08:58.650463.json'))
    to_pascal(spec, '../data', 'val')










