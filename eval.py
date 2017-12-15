#!/usr/bin/env python

from darkflow.defaults import argHandler #Import the default arguments
from darkflow.net.build import TFNet
import sys, pdb, os, json, numpy as np, cv2, pandas, time

def predict(self):
    DIR = os.path.dirname(os.path.realpath(self.FLAGS.test_boxes))
    val_data = json.load(open(self.FLAGS.test_boxes))

    all_boxes = []
    total_time = 0.0

    for i in range(0, len(val_data), self.FLAGS.batch):
        batch = val_data[i:i+self.FLAGS.batch]

        orig_images = [cv2.imread(os.path.join(DIR, a['image_path'])) for a in batch]

        inputs = [np.expand_dims(self.framework.preprocess(img.copy()), 0) for img in orig_images]
        feed_dict = {self.inp : np.concatenate(inputs, 0)}

        t1 = time.time()
        out = self.sess.run(self.out, feed_dict)
        total_time += time.time() - t1

        for j, fmap in enumerate(out):
            current_boxes = self.framework.findboxes(fmap)
            h, w, _ = orig_images[j].shape
            boxes = []
            for b in current_boxes:
                box = list(self.framework.process_box(b, h, w, -1))
                if box[5] == 1:
                    box.append(i + j)
                    boxes.append(box)
            all_boxes.extend(boxes)
        print('Done with [%d/%d]' % (i + self.FLAGS.batch, len(val_data)))
    print('Total time: %.4f seconds, per image: %.4f' % (total_time, total_time / len(val_data)))
    df = pandas.DataFrame(all_boxes)
    df.columns = ['x1', 'x2', 'y1', 'y2', 'label', 'label_id', 'score', 'image_id']
    df.to_csv('predictions.csv', index=False)


if __name__ == '__main__':
    FLAGS = argHandler()
    FLAGS.define('test_boxes', '', 'Path to json file defining validation data')
    FLAGS.setDefaults()
    FLAGS.parseArgs(sys.argv)

    # make sure all necessary dirs exist
    def _get_dir(dirs):
        for d in dirs:
            this = os.path.abspath(os.path.join(os.path.curdir, d))
            if not os.path.exists(this): os.makedirs(this)
    _get_dir([FLAGS.imgdir, FLAGS.binary, FLAGS.backup, os.path.join(FLAGS.imgdir,'out'), FLAGS.summary])

    # fix FLAGS.load to appropriate type
    try: FLAGS.load = int(FLAGS.load)
    except: pass

    tfnet = TFNet(FLAGS)
    predict(tfnet)
