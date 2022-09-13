import json

from util.eval_map import ANETdetection
import pandas as pd
import numpy as np


def gen_detection_multicore():
    # get video list
    thumos_test_anno = pd.read_csv("outputs/results_eval.csv")
    video_list = thumos_test_anno.video.unique()
    thu_label_id = np.sort(thumos_test_anno.type_idx.unique())[1:] - 1  # get thumos class id
    thu_video_id = np.array([int(i[-4:]) - 1 for i in video_list])  # -1 is to match python index

    # load video level classification
    cls_data = np.load("./data/uNet_test.npy")
    cls_data = cls_data[thu_video_id, :][:, thu_label_id]  # order by video list, output 213x20

    # detection_result
    thumos_gt = pd.read_csv("./data/thumos_annotations/thumos14_test_groundtruth.csv")
    global result
    result = {
        video:
            {
                'fps': thumos_gt.loc[thumos_gt['video-name'] == video]['frame-rate'].values[0],
                'num_frames': thumos_gt.loc[thumos_gt['video-name'] == video]['video-frames'].values[0]
            }
        for video in video_list
    }

    parallel = Parallel(n_jobs=15, prefer="processes")
    detection = parallel(delayed(_gen_detection_video)(video_name, video_cls, thu_label_id, opt)
                         for video_name, video_cls in zip(video_list, cls_data))
    detection_dict = {}
    [detection_dict.update(d) for d in detection]
    output_dict = {"version": "THUMOS14", "results": detection_dict, "external_data": {}}

    with open('outputs/detection_result.json', "w") as out:
        json.dump(output_dict, out)


if __name__ == '__main__':
    # TODO 待完成！
    tious = [0.3, 0.4, 0.5, 0.6, 0.7]
    eval_obj = ANETdetection(ground_truth_filename='./data/thomos_gt.json', prediction_filename='',
                             tiou_thresholds=tious,
                             subset='test', verbose=True)
