import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob
import h5py
from tensordict import TensorDict
from torchvision.transforms.v2 import Compose
from hydra.utils import instantiate
from typing import Optional, List


class TrajectoryDataset(Dataset):
    def __init__(
        self,
        directory: str,
        traj_len: int,
        downsample: Optional[int] = None,
        use_full_trajectory: bool = False,
        transforms: List = [],
        name: Optional[str] = None,
    ):
        """
        Args:
            file_list (list of str): List of file paths to the trajectory files.
            trajectory_length (int): The length of the trajectory to sample.
            downsample (int): Downsample the dataset by this factor.
            use_full_trajectory (bool): If True samples a sequence from the full trajectory and zero pads the end if it goes out of scope.
                If False doesn't sample from the end of an episode
            transforms (list of torchivision transforms): List of transforms to apply to the images.
            name (str): Name of the dataset.
        """
        assert traj_len > 0, "Trajectory length must be greater than 0"
        assert downsample is None or downsample > 0, "Downsample must be greater than 0"
        assert len(glob.glob(directory + "/*.hdf5")) > 0, "No files found in directory"

        self.name = name
        self.traj_len = traj_len
        self.use_full_trajectory = use_full_trajectory
        self.downsample = downsample
        self.file_list = glob.glob(directory + "/*.hdf5")
        self.qpos_norm, self.act_norm, self.filemap, self.data_len = (
            self.get_norm_stats(self.file_list)
        )
        print(
            f"Found {len(self.file_list)} files in {directory} with a total of {self.data_len} samples."
        )
        print(self.qpos_norm)
        print(self.act_norm)

        # remap filemap if not using full trajectories
        if not use_full_trajectory:
            new_filemap = {}
            new_data_len = 0
            for (start_idx, end_idx), f in self.filemap.items():
                new_start_idx = new_data_len
                new_end_idx = new_data_len + (end_idx - start_idx) - self.traj_len
                new_filemap[(new_start_idx, new_end_idx)] = f
                new_data_len += (end_idx - start_idx) - self.traj_len
            print(
                f"Cutting off episode ends, reduced dataset size from {self.data_len} to {new_data_len} samples."
            )
            self.filemap = new_filemap
            self.data_len = new_data_len

        # remap filemap to downsample
        if downsample and downsample != 1:
            new_filemap = {}
            new_data_len = 0
            for (start_idx, end_idx), f in self.filemap.items():
                new_start_idx = new_data_len
                new_end_idx = new_data_len + (end_idx - start_idx) // downsample
                new_filemap[(new_start_idx, new_end_idx)] = f
                new_data_len += (end_idx - start_idx) // downsample
            print(
                f"Downsampled dataset from {self.data_len} to {new_data_len} samples."
            )
            self.filemap = new_filemap
            self.data_len = new_data_len

        # Add transformations to dataset
        transforms_list = [instantiate(tf) for tf in transforms]
        self.transform = Compose(transforms_list)

        self.qpos_dim = int(np.prod(self.qpos_norm.mean.shape))
        self.act_dim = int(np.prod(self.act_norm.mean.shape))

    def __len__(self):
        return self.data_len

    def find_file(self, idx):
        # find which file to load; TODO can be more efficient
        filename = None
        for (start_idx, end_idx), f in self.filemap.items():
            if idx >= start_idx and idx < end_idx:
                filename = f
                ts = idx - start_idx

        assert (
            filename is not None
        ), f"couldn't find {idx} in dataset of length {len(self)}"
        return filename, ts

    def __getitem__(self, idx):

        # find which file to open and which timestep to sample
        filename, ts = self.find_file(idx)

        # Load the trajectory data from the file
        with h5py.File(filename, "r") as f:
            action = f["action"][:]
            obs = f["observations"]
            effort = obs["effort"][ts]
            qpos = obs["qpos"][ts]
            qvel = obs["qvel"][ts]
            images = []
            for key in obs["images"].keys():
                images.append(obs["images"][key][ts])

        # stack images and transform to a usable format
        images = torch.tensor(np.stack(images, axis=0), dtype=torch.float)
        images = torch.permute(images, (0, 3, 1, 2))
        images /= 255.0
        images = self.transform(images)

        # create qpos, just need to convert to tesnor
        qpos = torch.tensor(qpos, dtype=torch.float)

        # create actions from a trajectory slice
        # if the slice extends past the real trajectory, just 0 pad
        act = torch.zeros(self.traj_len, self.act_dim, dtype=torch.float)
        action = torch.tensor(action[ts : ts + self.traj_len], dtype=torch.float)
        act[: action.shape[0]] = action

        # normalize data for easier learning
        qpos = self.qpos_norm(qpos)
        act = self.act_norm(act)

        td = TensorDict(
            {"images": images, "qpos": qpos.squeeze(), "act": act.squeeze()}
        )
        return td

    def get_full_seq(self, idx):
        """Gets a full sequencen with original video input for evaluation purposes

        NOTE: heavy code duplication; need to figure out how to minimize"""

        # find which file to open and which timestep to sample
        filename, ts = self.find_file(idx)

        # Load the trajectory data from the file
        with h5py.File(filename, "r") as f:
            act = f["action"][:]
            obs = f["observations"]
            effort = obs["effort"][:]
            qpos = obs["qpos"][:]
            qvel = obs["qvel"][:]
            images = []
            for key in obs["images"].keys():
                images.append(obs["images"][key][:])
            og_images = obs["images"]["cam_high"][:]

        # stack images and transform to a usable format
        images = torch.tensor(np.stack(images, axis=0), dtype=torch.float)
        images = torch.permute(images, (1, 0, 4, 2, 3))
        images /= 255.0
        images = self.transform(images)
        qpos = torch.tensor(qpos, dtype=torch.float)
        act = torch.tensor(act, dtype=torch.float)

        qpos = self.qpos_norm(qpos)
        act = self.act_norm(act)

        td = TensorDict(
            {
                "images": images,
                "qpos": qpos.squeeze(),
                "act": act.squeeze(),
                "og_images": torch.tensor(np.stack(og_images, axis=0)),
            }
        )
        return td

    def get_norm_stats(self, dataset_files):
        all_qpos_data = []
        all_action_data = []
        filemap = {}
        start_idx = 0
        for f in dataset_files:
            with h5py.File(f, "r") as root:
                qpos = root["/observations/qpos"][()]
                qvel = root["/observations/qvel"][()]
                action = root["/action"][()]
            all_qpos_data.append(torch.from_numpy(qpos))
            all_action_data.append(torch.from_numpy(action))
            end_idx = start_idx + len(all_qpos_data[-1])
            filemap[(start_idx, end_idx)] = f
            start_idx += len(all_qpos_data[-1])

        all_qpos_data = torch.stack(all_qpos_data)
        all_action_data = torch.stack(all_action_data)

        qpos_norm = Normalizer.from_data(all_qpos_data)
        act_norm = Normalizer.from_data(all_action_data)

        return qpos_norm, act_norm, filemap, start_idx


class Normalizer:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = torch.clip(std, 1e-2, np.inf)

    @classmethod
    def from_data(cls, data):
        mean = data.mean(dim=[0, 1], keepdim=True)
        std = data.std(dim=[0, 1], keepdim=True)
        return cls(mean, std)

    def __call__(self, x):
        return (x - self.mean) / self.std

    def __repr__(self):
        return f"Normalizer(mean={self.mean}, std={self.std})"


if __name__ == "__main__":

    from IPython.core import ultratb
    import sys

    # For debugging
    sys.excepthook = ultratb.FormattedTB(
        mode="Plain", color_scheme="Neutral", call_pdb=1
    )

    datapath = "/coc/flash7/datasets/egoplay/_OBOO_ROBOT/oboov2_robot_apr16/rawAloha"
    trajectory_length = 100  # Desired length of the trajectory segment

    dataset = TrajectoryDataset(datapath, trajectory_length)

    # Example DataLoader
    dataloader = DataLoader(
        dataset, batch_size=16, shuffle=True, collate_fn=lambda x: torch.stack(x)
    )

    # Iterate over the DataLoader
    for batch in dataloader:
        print(batch)
        break
