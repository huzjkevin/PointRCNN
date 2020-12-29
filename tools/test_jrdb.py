import os
import random
import shutil
import time
import numpy as np
from mayavi import mlab
from lib.datasets.jrdb_handle import JRDBHandle
# import dr_spaam.utils.utils as u
import lib.utils.jrdb_transforms as jt

_XY_LIM = (-7, 7)
# _XY_LIM = (-30, 30)
_Z_LIM = (-1, 2)
_INTERACTIVE = False
_SAVE_DIR = "/home/jia/tmp_imgs/test_jrdb_handle_mayavi"

def rphi_to_xy(r, phi):
    return r * np.cos(phi), r * np.sin(phi)


def rphi_to_xy_torch(r, phi):
    return r * torch.cos(phi), r * torch.sin(phi)



def _test_loading_speed():
    data_handle = JRDBHandle(
        split="train",
        cfg={"data_dir": "./data/JRDB", "num_scans": 10, "scan_stride": 1},
    )
    total_frame = 100
    inds = random.sample(range(len(data_handle)), total_frame)
    t0 = time.time()
    for idx in inds:
        _ = data_handle[idx]
    t1 = time.time()
    print(f"Loaded {total_frame} frames in {t1 - t0} seconds.")
def _plot_sequence():
    jrdb_handle = JRDBHandle(
        split="train",
        cfg={"data_dir": "./data/JRDB", "num_scans": 10, "scan_stride": 1},
    )
    color_pool = np.random.uniform(size=(100, 3))
    if os.path.exists(_SAVE_DIR):
        shutil.rmtree(_SAVE_DIR)
    os.makedirs(_SAVE_DIR)
    for i, data_dict in enumerate(jrdb_handle):
        # lidar
        pc_xyz_upper = jt.transform_pts_upper_velodyne_to_base(
            data_dict["pc_data"]["upper_velodyne"]
        )
        pc_xyz_lower = jt.transform_pts_lower_velodyne_to_base(
            data_dict["pc_data"]["lower_velodyne"]
        )
        # labels
        boxes, label_ids = [], []
        for ann in data_dict["pc_anns"]:
            jrdb_handle.box_is_on_ground(ann)
            boxes.append(jt.box_from_jrdb(ann["box"]))
            label_ids.append(int(ann["label_id"].split(":")[-1]) % len(color_pool))
        # laser
        laser_r = data_dict["laser_data"][-1]
        laser_phi = data_dict["laser_grid"]
        laser_z = data_dict["laser_z"]
        laser_x, laser_y = rphi_to_xy(laser_r, laser_phi)
        pc_xyz_laser = jt.transform_pts_laser_to_base(
            np.stack((laser_x, laser_y, laser_z), axis=0)
        )
        mlab.points3d(pc_xyz_lower[0], pc_xyz_lower[1],
                      pc_xyz_lower[2], scale_factor=0.05, color=(0.0, 1.0, 0.0))
        mlab.points3d(pc_xyz_upper[0], pc_xyz_upper[1],
                      pc_xyz_upper[2], scale_factor=0.05, color=(0.0, 0.0, 1.0))
        mlab.points3d(pc_xyz_laser[0], pc_xyz_laser[1],
                      pc_xyz_laser[2], scale_factor=0.05, color=(1.0, 0.0, 0.0))
        for p_id, box in zip(label_ids, boxes):
            line_segments = box.to_line_segments()
            for lines in line_segments:
                mlab.plot3d(lines[0], lines[1], lines[2], tube_radius=None,
                            line_width=5, color=tuple(color_pool[p_id % 100]))
        mlab.show()
if __name__ == "__main__":
    # _test_loading_speed()
    _plot_sequence()