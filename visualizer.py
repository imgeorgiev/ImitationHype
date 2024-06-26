from mimicplay.scripts.aloha_process.simarUtils import (
    cam_frame_to_cam_pixels,
    draw_dot_on_frame,
    ee_pose_to_cam_frame,
    AlohaFK,
)
import torchvision
import numpy as np
import torch
import os

CURR_EXTRINSICS = np.array(
    [
        [-0.0096297, -0.70631061, 0.70783656, 0.0884877],
        [0.99961502, 0.01162055, 0.02519467, -0.30150412],
        [-0.02602072, 0.70780667, 0.70592679, -0.06821658],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

WIDE_LENS_ROBOT_LEFT_K = np.array(
    [
        [133.25430222 * 2, 0.0, 160.27941013 * 2, 0],
        [0.0, 133.2502574 * 2, 122.05743188 * 2, 0],
        [0.0, 0.0, 1.0, 0],
    ]
)


def visualize_policy(model, data, video_dir, device, epoch=None):
    """
    Args:
        model: torch model with forward()
        data: tensor dict with images, qpos, act, og_images
            og_images must by uint8; double check!
        video_dir: directory to save videos
        device: torch device
        epoch: int used for saving video
    """

    # Internal realsense numbers
    intrinsics = WIDE_LENS_ROBOT_LEFT_K

    T = data["act"].shape[0]
    video = torch.zeros((T, 480, 640, 3))

    aloha_fk = AlohaFK()

    for t in range(T):

        # predict trajectory first
        with torch.no_grad():
            data = data.to(device)
            img = data["images"][t][None]
            qpos = data["qpos"][t][None]
            a_hat, _ = model(img, qpos)

        H = a_hat.shape[1]

        im = data["og_images"][t].cpu().numpy()
        pred_values = a_hat[0].cpu().numpy()
        actions = data["act"][t : t + H].cpu().numpy()

        left_act = actions[:, :6]
        left_act_hat = pred_values[:, :6]
        left = not np.all(left_act == 0)
        right_act = actions[:, 7:-1]
        right_act_hat = pred_values[:, 7:-1]
        right = not np.all(right_act == 0)

        to_draw = []
        if left:
            to_draw.append(aloha_fk.fk(left_act))
            to_draw.append(aloha_fk.fk(left_act_hat))
        if right:
            to_draw.append(aloha_fk.fk(right_act))
            to_draw.append(aloha_fk.fk(right_act_hat))

        assert len(to_draw) > 0, "Nothing to draw"

        for i, each in enumerate(to_draw):
            each = ee_pose_to_cam_frame(each, CURR_EXTRINSICS)
            each = cam_frame_to_cam_pixels(each, intrinsics)
            pal = "Greens" if i % 2 == 0 else "Purples"
            im = draw_dot_on_frame(im, each, show=False, palette=pal)

        video[t] = torch.from_numpy(im)

    filename = os.path.join(video_dir, f"epoch_{epoch}.mp4")
    torchvision.io.write_video(filename, video, fps=30)
    return filename
