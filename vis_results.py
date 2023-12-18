from pathlib import Path
from PIL import Image
from tqdm import tqdm

import cv2
import numpy as np
from scipy import io as sio


def get_bounding_box(img):
    """Get bounding box coordinate information."""
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # due to python indexing, need to add 1 to max
    # else accessing will be 1px in the box, not out
    rmax += 1
    cmax += 1
    return [rmin, rmax, cmin, cmax]


def get_contour(pred):
    inst_id_list = np.unique(pred)[1:]  # exclude background
    inst_contour_list = []
    for inst_id in inst_id_list:
        inst_map = pred == inst_id
        rmin, rmax, cmin, cmax = get_bounding_box(inst_map)
        inst_bbox = np.array([[rmin, cmin], [rmax, cmax]])
        inst_map = inst_map[
                   inst_bbox[0][0]: inst_bbox[1][0], inst_bbox[0][1]: inst_bbox[1][1]
                   ]
        inst_map = inst_map.astype(np.uint8)
        inst_moment = cv2.moments(inst_map)
        inst_contour = cv2.findContours(
            inst_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        # * opencv protocol format may break
        inst_contour = np.squeeze(inst_contour[0][0].astype("int32"))
        # < 3 points dont make a contour, so skip, likely artifact too
        # as the contours obtained via approximation => too small or sthg
        if inst_contour.shape[0] < 3:
            continue
        if len(inst_contour.shape) != 2:
            continue
        inst_centroid = [
            (inst_moment["m10"] / inst_moment["m00"]),
            (inst_moment["m01"] / inst_moment["m00"]),
        ]
        inst_centroid = np.array(inst_centroid)
        inst_contour[:, 0] += inst_bbox[0][1]  # X
        inst_contour[:, 1] += inst_bbox[0][0]  # Y
        inst_centroid[0] += inst_bbox[0][1]  # X
        inst_centroid[1] += inst_bbox[0][0]  # Y
        inst_contour_list.append(inst_contour)

    return inst_contour_list


image_files = sorted(list(Path("data/PanNuke/images/Fold 3").glob("*.png")))[:300]
for d in ['pannuke2pannuke', 'consep2pannuke']:
    save_dir = Path(f"vis-{d}")
    (save_dir / "hv").mkdir(exist_ok=True, parents=True)
    (save_dir / "star").mkdir(exist_ok=True, parents=True)
    (save_dir / "our").mkdir(exist_ok=True, parents=True)
    (save_dir / "gt").mkdir(exist_ok=True, parents=True)
    # train on pannuke
    for idx, image_file in tqdm(enumerate(image_files), total=len(image_files)):
        hv_pred = Path(f"results/hv/{d}") / (image_file.stem + ".mat")
        hv_pred = sio.loadmat(hv_pred)['inst_map']
        star_pred = Path(f"results/stardist/{d}") / (image_file.stem + ".npy")
        star_pred = np.load(star_pred)
        our_pred = Path(f"results/segformer/{d}") / (image_file.stem + ".npy")
        our_pred = np.load(our_pred)
        gt = Path("data/PanNuke/labels/Fold 3") / (image_file.stem + ".npy")
        gt = np.load(gt)

        hv_overlay = np.array(Image.open(image_file))
        star_overlay = np.copy(hv_overlay)
        our_overlay = np.copy(hv_overlay)
        gt_overlay = np.copy(hv_overlay)
        # overlay = np.zeros(input_image.shape, dtype=np.uint8)
        # inst_rng_colors = random_colors(len(inst_dict))
        # inst_rng_colors = np.array(inst_rng_colors) * 255
        # inst_rng_colors = inst_rng_colors.astype(np.uint8)

        hv_contour = get_contour(hv_pred)
        star_contour = get_contour(star_pred)
        our_contour = get_contour(our_pred)
        gt_contour = get_contour(gt)
        # inst_colour = (0, 0, 0)  # 黑色
        cv2.drawContours(hv_overlay, hv_contour, -1, (0, 200, 0), 2)
        cv2.drawContours(star_overlay, star_contour, -1, (0, 200, 0), 2)
        cv2.drawContours(our_overlay, our_contour, -1, (0, 200, 0), 2)
        cv2.drawContours(gt_overlay, gt_contour, -1, (0, 200, 0), 2)

        Image.fromarray(hv_overlay).save(save_dir / "hv" / image_file.name)
        Image.fromarray(star_overlay).save(save_dir / "star" / image_file.name)
        Image.fromarray(our_overlay).save(save_dir / "our" / image_file.name)
        Image.fromarray(gt_overlay).save(save_dir / "gt" / image_file.name)
