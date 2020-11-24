import _init_path
import os
import numpy as np
import pickle
import torch
print(os.getcwd())
import lib.utils.roipool3d.roipool3d_utils as roipool3d_utils
from lib.datasets.jrdb_handle import JRDBHandle
import argparse
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', type=str, default='./gt_database')
parser.add_argument('--class_name', type=str, default='Car')
parser.add_argument('--split', type=str, default='train')
args = parser.parse_args()


class GTDatabaseGenerator(JRDBHandle):
    def __init__(self, cfg, split='train'):
        super().__init__(split, cfg)
        self.gt_database = None
        self.classes = ('Background', 'Pedestrian')
        self.__handle = JRDBHandle(split, cfg=cfg)

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError

    def filtrate_objects(self, obj_list):
        valid_obj_list = []
        for obj in obj_list:
            if obj.cls_type not in self.classes:
                continue
            valid_obj_list.append(obj)

        return valid_obj_list

    def generate_gt_database(self):
        gt_database = []
        for idx, data in enumerate(self.__handle):
            sample_id = int(data["sample_id"])
            # print('process gt sample (id=%06d)' % sample_id)

            # pts_lidar = self.get_lidar(sample_id)
            # calib = self.get_calib(sample_id)

            # pts_rect = calib.lidar_to_rect(pts_lidar[:, 0:3])
            pts_rect = data['points']
            pts_intensity = np.ones((len(pts_rect), 1))

            # obj_list = self.filtrate_objects(self.get_label(sample_id))

            # gt_boxes3d = np.zeros((obj_list.__len__(), 7), dtype=np.float32)
            # for k, obj in enumerate(data["boxes"]):
            #     gt_boxes3d[k, 0:3], gt_boxes3d[k, 3], gt_boxes3d[k, 4], gt_boxes3d[k, 5], gt_boxes3d[k, 6] \
            #         = obj[:3], obj[5], obj[4], obj[3], obj[6]

            gt_boxes3d = data["boxes"][:, [0, 1, 2, 5, 4, 3, 6]]


            if gt_boxes3d.__len__() == 0:
                print('No gt object')
                continue

            boxes_pts_mask_list = roipool3d_utils.pts_in_boxes3d_cpu(torch.from_numpy(pts_rect), torch.from_numpy(gt_boxes3d))

            for k in range(boxes_pts_mask_list.__len__()):
                pt_mask_flag = (boxes_pts_mask_list[k].numpy() == 1)
                cur_pts = pts_rect[pt_mask_flag].astype(np.float32)
                cur_pts_intensity = pts_intensity[pt_mask_flag].astype(np.float32)
                sample_dict = {'sample_id': sample_id,
                               'cls_type': "Pedestrian",
                               'gt_box3d': gt_boxes3d[k],
                               'points': cur_pts,
                               'intensity': cur_pts_intensity,
                               'obj': data["boxes"][k]}
                gt_database.append(sample_dict)

        save_file_name = os.path.join(args.save_dir, '%s_gt_database_3level_%s.pkl' % (args.split, self.classes[-1]))
        with open(save_file_name, 'wb') as f:
            pickle.dump(gt_database, f)

        self.gt_database = gt_database
        print('Save refine training sample info file to %s' % save_file_name)


if __name__ == '__main__':
    cfg_file = "cfgs/jrdb_cfg.yaml"

    with open(cfg_file, "r") as f:
        cfg = yaml.safe_load(f)

    dataset = GTDatabaseGenerator(cfg["dataset"], split=args.split)
    os.makedirs(args.save_dir, exist_ok=True)

    dataset.generate_gt_database()
    print("database created")

    # gt_database = pickle.load(open('gt_database/train_gt_database.pkl', 'rb'))
    # print(gt_database.__len__())
    # import pdb
    # pdb.set_trace()
