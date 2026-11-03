import logging
import random
import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from tmrl.core.memory import BaseMemory, check_samples_crc
from tmrl.core.torch.memory import TorchMemory
from tmrl.custom.torch.poincare_memory import (
    REPLAY_SAMPLER_DEFAULTS,
    greedy_diverse_subset,
    mean_off_diagonal,
    pairwise_euclidean_distance,
    pairwise_poincare_distance,
    poincare_expmap0,
)


# LOCAL BUFFER COMPRESSION ==============================


def get_local_buffer_sample_lidar(prev_act, obs, rew, terminated, truncated, info):
    """
    Input:
        prev_act: action computed from a previous observation and applied to yield obs in the transition (but not influencing the unaugmented observation in real-time envs)
        obs, rew, terminated, truncated, info: outcome of the transition
    this function creates the object that will actually be stored in local buffers for networking
    this is to compress the sample before sending it over the Internet/local network
    buffers of such samples will be given as input to the append() method of the memory
    the user must define both this function and the append() method of the memory
    CAUTION: prev_act is the action that comes BEFORE obs (i.e. prev_obs, prev_act(prev_obs), obs(prev_act))
    """
    obs_mod = (obs[0], obs[1][-19:])  # speed and most recent LIDAR only
    rew_mod = np.float32(rew)
    terminated_mod = terminated
    truncated_mod = truncated
    return prev_act, obs_mod, rew_mod, terminated_mod, truncated_mod, info


def get_local_buffer_sample_lidar_progress(prev_act, obs, rew, terminated, truncated, info):
    """
    Input:
        prev_act: action computed from a previous observation and applied to yield obs in the transition (but not influencing the unaugmented observation in real-time envs)
        obs, rew, terminated, truncated, info: outcome of the transition
    this function creates the object that will actually be stored in local buffers for networking
    this is to compress the sample before sending it over the Internet/local network
    buffers of such samples will be given as input to the append() method of the memory
    the user must define both this function and the append() method of the memory
    CAUTION: prev_act is the action that comes BEFORE obs (i.e. prev_obs, prev_act(prev_obs), obs(prev_act))
    """
    obs_mod = (obs[0], obs[1], obs[2][-19:])  # speed and most recent LIDAR only
    rew_mod = np.float32(rew)
    terminated_mod = terminated
    truncated_mod = truncated
    return prev_act, obs_mod, rew_mod, terminated_mod, truncated_mod, info


def get_local_buffer_sample_tm20_imgs(prev_act, obs, rew, terminated, truncated, info):
    """
    Sample compressor for MemoryTMFull
    Input:
        prev_act: action computed from a previous observation and applied to yield obs in the transition
        obs, rew, terminated, truncated, info: outcome of the transition
    this function creates the object that will actually be stored in local buffers for networking
    this is to compress the sample before sending it over the Internet/local network
    buffers of such samples will be given as input to the append() method of the memory
    the user must define both this function and the append() method of the memory
    CAUTION: prev_act is the action that comes BEFORE obs (i.e. prev_obs, prev_act(prev_obs), obs(prev_act))
    """
    prev_act_mod = prev_act
    obs_mod = (obs[0], obs[1], obs[2], (obs[3][-1] * 256.0).astype(np.uint8))
    rew_mod = rew
    terminated_mod = terminated
    truncated_mod = truncated
    info_mod = info
    return prev_act_mod, obs_mod, rew_mod, terminated_mod, truncated_mod, info_mod


# FUNCTIONS ====================================================


def last_true_in_list(li):
    for i in reversed(range(len(li))):
        if li[i]:
            return i
    return None


def replace_hist_before_eoe(hist, eoe_idx_in_hist):
    """
    Pads the history hist before the End Of Episode (EOE) index.

    Previous entries in hist are padded with copies of the first element occurring after EOE.
    """
    last_idx = len(hist) - 1
    assert eoe_idx_in_hist <= last_idx, f"replace_hist_before_eoe: eoe_idx_in_hist:{eoe_idx_in_hist}, last_idx:{last_idx}"
    if 0 <= eoe_idx_in_hist < last_idx:
        for i in reversed(range(len(hist))):  # clone the sample that is AFTER the last EOE (i.e., at reset)
            if i <= eoe_idx_in_hist:
                hist[i] = hist[i + 1]


# SUPPORTED CUSTOM MEMORIES ============================================================================================


class GenericTorchMemory(TorchMemory):
    """
    Generic Torch memory implementation.

    GenericTorchMemory can handle most nested structures, but does not implement any memory or computational optimization.
    """
    def __init__(self,
                 memory_size=1e6,
                 batch_size=1,
                 dataset_path="",
                 sample_preprocessor: callable = None,
                 crc_debug=False,
                 device="cpu"):
        super().__init__(memory_size=memory_size,
                         batch_size=batch_size,
                         dataset_path=dataset_path,
                         sample_preprocessor=sample_preprocessor,
                         crc_debug=crc_debug,
                         device=device)

    def append_buffer(self, buffer):

        # parse:
        d0 = [b[0] for b in buffer.memory]  # actions
        d1 = [b[1] for b in buffer.memory]  # observations
        d2 = [b[2] for b in buffer.memory]  # rewards
        d3 = [b[3] for b in buffer.memory]  # terminated
        d4 = [b[4] for b in buffer.memory]  # truncated
        d5 = [b[5] for b in buffer.memory]  # info
        d6 = [b[3] or b[4] for b in buffer.memory]  # done

        # append:
        if self.__len__() > 0:
            self.data[0] += d0
            self.data[1] += d1
            self.data[2] += d2
            self.data[3] += d3
            self.data[4] += d4
            self.data[5] += d5
            self.data[6] += d6
        else:
            self.data.append(d0)
            self.data.append(d1)
            self.data.append(d2)
            self.data.append(d3)
            self.data.append(d4)
            self.data.append(d5)
            self.data.append(d6)

        # trim
        to_trim = int(self.__len__() - self.memory_size)
        if to_trim > 0:
            self.data[0] = self.data[0][to_trim:]
            self.data[1] = self.data[1][to_trim:]
            self.data[2] = self.data[2][to_trim:]
            self.data[3] = self.data[3][to_trim:]
            self.data[4] = self.data[4][to_trim:]
            self.data[5] = self.data[5][to_trim:]
            self.data[6] = self.data[6][to_trim:]

    def __len__(self):
        if len(self.data) == 0:
            return 0
        res = len(self.data[0]) - 1
        if res < 0:
            return 0
        else:
            return res

    def get_transition(self, item):

        # This is a hack to avoid invalid transitions from terminal to initial
        # TODO: find a way to only index valid transitions instead
        while self.data[6][item]:
            item = random.randint(a=0, b=self.__len__() - 1)

        idx_last = item
        idx_now = item + 1

        last_obs = self.data[1][idx_last]
        new_act = self.data[0][idx_now]
        rew = self.data[2][idx_now]
        new_obs = self.data[1][idx_now]
        terminated = self.data[3][idx_now]
        truncated = self.data[4][idx_now]
        info = self.data[5][idx_now]

        return last_obs, new_act, rew, new_obs, terminated, truncated, info


class ArrayTorchMemory(BaseMemory):
    """
    Optimized generic Torch memory implementation for simple numpy arrays.

    ArrayTorchMemory only handles numpy arrays (no nested structures).
    If your observations or actions are made of nested structures, you need to flatten them into homogeneous numpy arrays when using ArrayTorchMemory.
    """
    def __init__(self,
                 memory_size=1e6,
                 batch_size=1,
                 dataset_path="",
                 sample_preprocessor: callable = None,
                 crc_debug=False,
                 device="cpu",
                 replace=False,
                 shuffle=False):
        super().__init__(memory_size=memory_size,
                         batch_size=batch_size,
                         dataset_path=dataset_path,
                         sample_preprocessor=sample_preprocessor,
                         crc_debug=crc_debug,
                         device=device)
        self.replace = replace
        self.shuffle = shuffle
        self.data = []
        self.rng = np.random.default_rng()

    def __len__(self):
        if len(self.data) == 0:
            return 0
        res = len(self.data[0]) - 1
        if res < 0:
            return 0
        else:
            return res

    def append_buffer(self, buffer):

        elt = buffer.memory[0]
        assert isinstance(elt[0], np.ndarray), f"Actions must be numpy arrays. Found {type(elt[0])}"
        assert isinstance(elt[1], np.ndarray), f"Observations must be numpy arrays. Found {type(elt[1])}"

        # parse:
        d0 = np.stack([b[0] for b in buffer.memory], dtype=np.float32)  # actions
        d1 = np.stack([b[1] for b in buffer.memory], dtype=np.float32)  # observations
        d2 = np.stack([b[2] for b in buffer.memory], dtype=np.float32)  # rewards
        d3 = np.stack([b[3] for b in buffer.memory])  # terminated
        d4 = np.stack([b[4] for b in buffer.memory])  # truncated
        d5 = [b[5] for b in buffer.memory]  # info dicts
        d6 = np.stack([b[3] or b[4] for b in buffer.memory])  # done

        # append:
        if self.__len__() > 0:
            self.data[0] = np.concatenate((self.data[0], d0))
            self.data[1] = np.concatenate((self.data[1], d1))
            self.data[2] = np.concatenate((self.data[2], d2))
            self.data[3] = np.concatenate((self.data[3], d3))
            self.data[4] = np.concatenate((self.data[4], d4))
            self.data[5] += d5  # info dicts
            self.data[6] = np.concatenate((self.data[6], d6))
        else:
            self.data.append(d0)
            self.data.append(d1)
            self.data.append(d2)
            self.data.append(d3)
            self.data.append(d4)
            self.data.append(d5)
            self.data.append(d6)

        # trim
        to_trim = int(self.__len__() - self.memory_size)
        if to_trim > 0:
            self.data[0] = self.data[0][to_trim:]
            self.data[1] = self.data[1][to_trim:]
            self.data[2] = self.data[2][to_trim:]
            self.data[3] = self.data[3][to_trim:]
            self.data[4] = self.data[4][to_trim:]
            self.data[5] = self.data[5][to_trim:]
            self.data[6] = self.data[6][to_trim:]

    def sample(self):

        max_idx = len(self) - 1

        # sample indices in replay buffer:
        indices = self.rng.choice(a=max_idx, size=self.batch_size, replace=self.replace if max_idx > self.batch_size else True, shuffle=self.shuffle)
        dones = self.data[6][indices]

        # resample indices that refer to invalid transitions from terminal to initial states:
        # TODO: find a way to only index valid transitions instead
        while np.any(dones):
            to_resample = np.where(dones)[0]
            indices[to_resample] = self.rng.choice(a=max_idx, size=len(to_resample), replace=self.replace if max_idx > len(to_resample) else True, shuffle=self.shuffle)
            dones[to_resample] = self.data[6][indices[to_resample]]

        idx_last = indices
        idx_now = indices + 1

        # numpy batched tensors:
        last_obs_batch = self.data[1][idx_last]
        new_act_batch = self.data[0][idx_now]
        rew_batch = self.data[2][idx_now]
        new_obs_batch = self.data[1][idx_now]
        terminated_batch = self.data[3][idx_now]
        truncated_batch = self.data[4][idx_now]

        # CRC-debug numpy batched tensors:
        if self.crc_debug:
            for i in range(len(idx_now)):
                prev_obs = last_obs_batch[i]
                new_act = new_act_batch[i]
                rew = rew_batch[i]
                new_obs = new_obs_batch[i]
                terminated = terminated_batch[i]
                truncated = truncated_batch[i]
                info = self.data[5][idx_now[i]]
                po, a, o, r, d, t = info['crc_sample']
                debug_ts, debug_ts_res = info['crc_sample_ts']
                check_samples_crc(po, a, o, r, d, t, prev_obs, new_act, new_obs, rew, terminated, truncated, debug_ts, debug_ts_res, epsilon=1e-4)

        if self.sample_preprocessor is not None:
            raise RuntimeError("Sample preprocessor support not implemented here.")

        # convert numpy tensors to torch tensors:
        last_obs_batch = torch.tensor(last_obs_batch, dtype=torch.float32).to(self.device)
        new_act_batch = torch.tensor(new_act_batch, dtype=torch.float32).to(self.device)
        rew_batch = torch.tensor(rew_batch, dtype=torch.float32).to(self.device)
        new_obs_batch = torch.tensor(new_obs_batch, dtype=torch.float32).to(self.device)
        terminated_batch = torch.tensor(terminated_batch, dtype=torch.float32).to(self.device)
        truncated_batch = torch.tensor(truncated_batch, dtype=torch.float32).to(self.device)

        return last_obs_batch, new_act_batch, rew_batch, new_obs_batch, terminated_batch, truncated_batch


class MemoryTM(TorchMemory):
    def __init__(self,
                 memory_size=None,
                 batch_size=None,
                 dataset_path="",
                 imgs_obs=4,
                 act_buf_len=1,
                 sample_preprocessor: callable = None,
                 crc_debug=False,
                 device="cpu"):
        self.imgs_obs = imgs_obs
        self.act_buf_len = act_buf_len
        self.min_samples = max(self.imgs_obs, self.act_buf_len)
        self.start_imgs_offset = max(0, self.min_samples - self.imgs_obs)
        self.start_acts_offset = max(0, self.min_samples - self.act_buf_len)
        super().__init__(memory_size=memory_size,
                         batch_size=batch_size,
                         dataset_path=dataset_path,
                         sample_preprocessor=sample_preprocessor,
                         crc_debug=crc_debug,
                         device=device)

    def append_buffer(self, buffer):
        raise NotImplementedError

    def __len__(self):
        if len(self.data) == 0:
            return 0
        res = len(self.data[0]) - self.min_samples - 1
        if res < 0:
            return 0
        else:
            return res

    def get_transition(self, item):
        raise NotImplementedError


class MemoryTMLidar(MemoryTM):
    def get_transition(self, item):
        """
        CAUTION: item is the first index of the 4 images in the images history of the OLD observation
        CAUTION: in the buffer, a sample is (act, obs(act)) and NOT (obs, act(obs))
            i.e. in a sample, the observation is what step returned after being fed act (and preprocessed)
            therefore, in the RTRL setting, act is appended to obs
        So we load 5 images from here...
        Don't forget the info dict for CRC debugging
        """
        if self.data[4][item + self.min_samples - 1]:
            if item == 0:  # if first item of the buffer
                item += 1
            elif item == self.__len__() - 1:  # if last item of the buffer
                item -= 1
            elif random.random() < 0.5:  # otherwise, sample randomly
                item += 1
            else:
                item -= 1

        idx_last = item + self.min_samples - 1
        idx_now = item + self.min_samples

        acts = self.load_acts(item)
        last_act_buf = acts[:-1]
        new_act_buf = acts[1:]

        imgs = self.load_imgs(item)
        imgs_last_obs = imgs[:-1]
        imgs_new_obs = imgs[1:]

        # if a reset transition has influenced the observation, special care must be taken
        last_eoes = self.data[4][idx_now - self.min_samples:idx_now]  # self.min_samples values
        last_eoe_idx = last_true_in_list(last_eoes)  # last occurrence of True

        assert last_eoe_idx is None or last_eoes[last_eoe_idx], f"last_eoe_idx:{last_eoe_idx}"

        if last_eoe_idx is not None:
            replace_hist_before_eoe(hist=new_act_buf, eoe_idx_in_hist=last_eoe_idx - self.start_acts_offset - 1)
            replace_hist_before_eoe(hist=last_act_buf, eoe_idx_in_hist=last_eoe_idx - self.start_acts_offset)
            replace_hist_before_eoe(hist=imgs_new_obs, eoe_idx_in_hist=last_eoe_idx - self.start_imgs_offset - 1)
            replace_hist_before_eoe(hist=imgs_last_obs, eoe_idx_in_hist=last_eoe_idx - self.start_imgs_offset)

        imgs_new_obs = np.ndarray.flatten(imgs_new_obs)
        imgs_last_obs = np.ndarray.flatten(imgs_last_obs)

        last_obs = (self.data[2][idx_last], imgs_last_obs, *last_act_buf)
        new_act = self.data[1][idx_now]
        rew = np.float32(self.data[5][idx_now])
        new_obs = (self.data[2][idx_now], imgs_new_obs, *new_act_buf)
        terminated = self.data[7][idx_now]
        truncated = self.data[8][idx_now]
        info = self.data[6][idx_now]
        return last_obs, new_act, rew, new_obs, terminated, truncated, info

    def load_imgs(self, item):
        res = self.data[3][(item + self.start_imgs_offset):(item + self.start_imgs_offset + self.imgs_obs + 1)]
        return np.stack(res)

    def load_acts(self, item):
        res = self.data[1][(item + self.start_acts_offset):(item + self.start_acts_offset + self.act_buf_len + 1)]
        return res

    def append_buffer(self, buffer):
        """
        buffer is a list of samples (act, obs, rew, terminated, truncated, info)
        don't forget to keep the info dictionary in the sample for CRC debugging
        """

        first_data_idx = self.data[0][-1] + 1 if self.__len__() > 0 else 0

        d0 = [first_data_idx + i for i, _ in enumerate(buffer.memory)]  # indexes
        d1 = [b[0] for b in buffer.memory]  # actions
        d2 = [b[1][0] for b in buffer.memory]  # speeds
        d3 = [b[1][1] for b in buffer.memory]  # lidar
        d4 = [b[3] or b[4] for b in buffer.memory]  # eoes (terminated or truncated)
        d5 = [b[2] for b in buffer.memory]  # rewards
        d6 = [b[5] for b in buffer.memory]  # infos
        d7 = [b[3] for b in buffer.memory]  # terminated
        d8 = [b[4] for b in buffer.memory]  # truncated

        if self.__len__() > 0:
            self.data[0] += d0
            self.data[1] += d1
            self.data[2] += d2
            self.data[3] += d3
            self.data[4] += d4
            self.data[5] += d5
            self.data[6] += d6
            self.data[7] += d7
            self.data[8] += d8
        else:
            self.data.append(d0)
            self.data.append(d1)
            self.data.append(d2)
            self.data.append(d3)
            self.data.append(d4)
            self.data.append(d5)
            self.data.append(d6)
            self.data.append(d7)
            self.data.append(d8)

        to_trim = self.__len__() - self.memory_size
        if to_trim > 0:
            self.data[0] = self.data[0][to_trim:]
            self.data[1] = self.data[1][to_trim:]
            self.data[2] = self.data[2][to_trim:]
            self.data[3] = self.data[3][to_trim:]
            self.data[4] = self.data[4][to_trim:]
            self.data[5] = self.data[5][to_trim:]
            self.data[6] = self.data[6][to_trim:]
            self.data[7] = self.data[7][to_trim:]
            self.data[8] = self.data[8][to_trim:]

        return self


class MemoryTMLidarProgress(MemoryTM):
    def get_transition(self, item):
        """
        CAUTION: item is the first index of the 4 images in the images history of the OLD observation
        CAUTION: in the buffer, a sample is (act, obs(act)) and NOT (obs, act(obs))
            i.e. in a sample, the observation is what step returned after being fed act (and preprocessed)
            therefore, in the RTRL setting, act is appended to obs
        So we load 5 images from here...
        Don't forget the info dict for CRC debugging
        """
        if self.data[4][item + self.min_samples - 1]:
            if item == 0:  # if first item of the buffer
                item += 1
            elif item == self.__len__() - 1:  # if last item of the buffer
                item -= 1
            elif random.random() < 0.5:  # otherwise, sample randomly
                item += 1
            else:
                item -= 1

        idx_last = item + self.min_samples - 1
        idx_now = item + self.min_samples

        acts = self.load_acts(item)
        last_act_buf = acts[:-1]
        new_act_buf = acts[1:]

        imgs = self.load_imgs(item)
        imgs_last_obs = imgs[:-1]
        imgs_new_obs = imgs[1:]

        # if a reset transition has influenced the observation, special care must be taken
        last_eoes = self.data[4][idx_now - self.min_samples:idx_now]  # self.min_samples values
        last_eoe_idx = last_true_in_list(last_eoes)  # last occurrence of True

        assert last_eoe_idx is None or last_eoes[last_eoe_idx], f"last_eoe_idx:{last_eoe_idx}"

        if last_eoe_idx is not None:
            replace_hist_before_eoe(hist=new_act_buf, eoe_idx_in_hist=last_eoe_idx - self.start_acts_offset - 1)
            replace_hist_before_eoe(hist=last_act_buf, eoe_idx_in_hist=last_eoe_idx - self.start_acts_offset)
            replace_hist_before_eoe(hist=imgs_new_obs, eoe_idx_in_hist=last_eoe_idx - self.start_imgs_offset - 1)
            replace_hist_before_eoe(hist=imgs_last_obs, eoe_idx_in_hist=last_eoe_idx - self.start_imgs_offset)

        imgs_new_obs = np.ndarray.flatten(imgs_new_obs)
        imgs_last_obs = np.ndarray.flatten(imgs_last_obs)

        last_obs = (self.data[2][idx_last], self.data[7][idx_last], imgs_last_obs, *last_act_buf)
        new_act = self.data[1][idx_now]
        rew = np.float32(self.data[5][idx_now])
        new_obs = (self.data[2][idx_now], self.data[7][idx_now], imgs_new_obs, *new_act_buf)
        terminated = self.data[8][idx_now]
        truncated = self.data[9][idx_now]
        info = self.data[6][idx_now]
        return last_obs, new_act, rew, new_obs, terminated, truncated, info

    def load_imgs(self, item):
        res = self.data[3][(item + self.start_imgs_offset):(item + self.start_imgs_offset + self.imgs_obs + 1)]
        return np.stack(res)

    def load_acts(self, item):
        res = self.data[1][(item + self.start_acts_offset):(item + self.start_acts_offset + self.act_buf_len + 1)]
        return res

    def append_buffer(self, buffer):
        """
        buffer is a list of samples (act, obs, rew, truncated, terminated, info)
        don't forget to keep the info dictionary in the sample for CRC debugging
        """

        first_data_idx = self.data[0][-1] + 1 if self.__len__() > 0 else 0

        d0 = [first_data_idx + i for i, _ in enumerate(buffer.memory)]  # indexes
        d1 = [b[0] for b in buffer.memory]  # actions
        d2 = [b[1][0] for b in buffer.memory]  # speeds
        d3 = [b[1][2] for b in buffer.memory]  # lidar
        d4 = [b[3] or b[4] for b in buffer.memory]  # eoes
        d5 = [b[2] for b in buffer.memory]  # rewards
        d6 = [b[5] for b in buffer.memory]  # infos
        d7 = [b[1][1] for b in buffer.memory]  # progress
        d8 = [b[3] for b in buffer.memory]  # terminated
        d9 = [b[4] for b in buffer.memory]  # truncated

        if self.__len__() > 0:
            self.data[0] += d0
            self.data[1] += d1
            self.data[2] += d2
            self.data[3] += d3
            self.data[4] += d4
            self.data[5] += d5
            self.data[6] += d6
            self.data[7] += d7
            self.data[8] += d8
            self.data[9] += d9
        else:
            self.data.append(d0)
            self.data.append(d1)
            self.data.append(d2)
            self.data.append(d3)
            self.data.append(d4)
            self.data.append(d5)
            self.data.append(d6)
            self.data.append(d7)
            self.data.append(d8)
            self.data.append(d9)

        to_trim = self.__len__() - self.memory_size
        if to_trim > 0:
            self.data[0] = self.data[0][to_trim:]
            self.data[1] = self.data[1][to_trim:]
            self.data[2] = self.data[2][to_trim:]
            self.data[3] = self.data[3][to_trim:]
            self.data[4] = self.data[4][to_trim:]
            self.data[5] = self.data[5][to_trim:]
            self.data[6] = self.data[6][to_trim:]
            self.data[7] = self.data[7][to_trim:]
            self.data[8] = self.data[8][to_trim:]
            self.data[9] = self.data[9][to_trim:]

        return self


class MemoryTMFull(MemoryTM):
    def get_transition(self, item):
        """
        CAUTION: item is the first index of the 4 images in the images history of the OLD observation
        CAUTION: in the buffer, a sample is (act, obs(act)) and NOT (obs, act(obs))
            i.e. in a sample, the observation is what step returned after being fed act (and preprocessed)
            therefore, in the RTRL setting, act is appended to obs
        So we load 5 images from here...
        Don't forget the info dict for CRC debugging
        """
        if self.data[4][item + self.min_samples - 1]:
            if item == 0:  # if first item of the buffer
                item += 1
            elif item == self.__len__() - 1:  # if last item of the buffer
                item -= 1
            elif random.random() < 0.5:  # otherwise, sample randomly
                item += 1
            else:
                item -= 1

        idx_last = item + self.min_samples - 1
        idx_now = item + self.min_samples

        acts = self.load_acts(item)
        last_act_buf = acts[:-1]
        new_act_buf = acts[1:]

        imgs = self.load_imgs(item)
        imgs_last_obs = imgs[:-1]
        imgs_new_obs = imgs[1:]

        # if a reset transition has influenced the observation, special care must be taken
        last_eoes = self.data[4][idx_now - self.min_samples:idx_now]  # self.min_samples values
        last_eoe_idx = last_true_in_list(last_eoes)  # last occurrence of True

        assert last_eoe_idx is None or last_eoes[last_eoe_idx], f"last_eoe_idx:{last_eoe_idx}"

        if last_eoe_idx is not None:
            replace_hist_before_eoe(hist=new_act_buf, eoe_idx_in_hist=last_eoe_idx - self.start_acts_offset - 1)
            replace_hist_before_eoe(hist=last_act_buf, eoe_idx_in_hist=last_eoe_idx - self.start_acts_offset)
            replace_hist_before_eoe(hist=imgs_new_obs, eoe_idx_in_hist=last_eoe_idx - self.start_imgs_offset - 1)
            replace_hist_before_eoe(hist=imgs_last_obs, eoe_idx_in_hist=last_eoe_idx - self.start_imgs_offset)

        last_obs = (self.data[2][idx_last], self.data[7][idx_last], self.data[8][idx_last], imgs_last_obs, *last_act_buf)
        new_act = self.data[1][idx_now]
        rew = np.float32(self.data[5][idx_now])
        new_obs = (self.data[2][idx_now], self.data[7][idx_now], self.data[8][idx_now], imgs_new_obs, *new_act_buf)
        terminated = self.data[9][idx_now]
        truncated = self.data[10][idx_now]
        info = self.data[6][idx_now]
        return last_obs, new_act, rew, new_obs, terminated, truncated, info

    def load_imgs(self, item):
        res = self.data[3][(item + self.start_imgs_offset):(item + self.start_imgs_offset + self.imgs_obs + 1)]
        return np.stack(res).astype(np.float32) / 256.0

    def load_acts(self, item):
        res = self.data[1][(item + self.start_acts_offset):(item + self.start_acts_offset + self.act_buf_len + 1)]
        return res

    def append_buffer(self, buffer):
        """
        buffer is a list of samples ( act, obs, rew, terminated, truncated, info)
        don't forget to keep the info dictionary in the sample for CRC debugging
        """

        first_data_idx = self.data[0][-1] + 1 if self.__len__() > 0 else 0

        d0 = [first_data_idx + i for i, _ in enumerate(buffer.memory)]  # indexes
        d1 = [b[0] for b in buffer.memory]  # actions
        d2 = [b[1][0] for b in buffer.memory]  # speeds
        d3 = [b[1][3] for b in buffer.memory]  # images
        d4 = [b[3] or b[4] for b in buffer.memory]  # eoes
        d5 = [b[2] for b in buffer.memory]  # rewards
        d6 = [b[5] for b in buffer.memory]  # infos
        d7 = [b[1][1] for b in buffer.memory]  # gears
        d8 = [b[1][2] for b in buffer.memory]  # rpms
        d9 = [b[3] for b in buffer.memory]  # terminated
        d10 = [b[4] for b in buffer.memory]  # truncated

        if self.__len__() > 0:
            self.data[0] += d0
            self.data[1] += d1
            self.data[2] += d2
            self.data[3] += d3
            self.data[4] += d4
            self.data[5] += d5
            self.data[6] += d6
            self.data[7] += d7
            self.data[8] += d8
            self.data[9] += d9
            self.data[10] += d10
        else:
            self.data.append(d0)
            self.data.append(d1)
            self.data.append(d2)
            self.data.append(d3)
            self.data.append(d4)
            self.data.append(d5)
            self.data.append(d6)
            self.data.append(d7)
            self.data.append(d8)
            self.data.append(d9)
            self.data.append(d10)

        to_trim = self.__len__() - self.memory_size
        if to_trim > 0:
            self.data[0] = self.data[0][to_trim:]
            self.data[1] = self.data[1][to_trim:]
            self.data[2] = self.data[2][to_trim:]
            self.data[3] = self.data[3][to_trim:]
            self.data[4] = self.data[4][to_trim:]
            self.data[5] = self.data[5][to_trim:]
            self.data[6] = self.data[6][to_trim:]
            self.data[7] = self.data[7][to_trim:]
            self.data[8] = self.data[8][to_trim:]
            self.data[9] = self.data[9][to_trim:]
            self.data[10] = self.data[10][to_trim:]

        return self


def fix_batch_history_around_eoes_numpy(batch_history, eoes):
    assert eoes.shape[0] == batch_history.shape[0]  # batch dimension
    assert eoes.shape[1] == batch_history.shape[1]  # history dimension
    has_true = eoes[:, :-1].any(axis=1)  # the last eoe in the history is not relevant
    if np.any(has_true):
        to_resample = np.where(has_true)[0]
        eoes_to_resample = eoes[to_resample][:, :-1]
        rev_eoes_to_resample = np.flip(eoes_to_resample, axis=1)
        last_true_idx = batch_history.shape[1] - 2 - rev_eoes_to_resample.argmax(axis=1)
        resampled = batch_history[to_resample]
        for i in range(len(to_resample)):
            idx = last_true_idx[i]
            val_to_copy = resampled[i, idx+1]
            resampled[i, :idx+1] = val_to_copy
        batch_history[to_resample] = resampled


def fix_batch_history_around_eoes_torch(batch_history, eoes):
    assert eoes.shape[0] == batch_history.shape[0]  # batch dimension
    assert eoes.shape[1] == batch_history.shape[1]  # history dimension
    has_true = eoes[:, :-1].any(axis=1)  # the last eoe in the history is not relevant
    if torch.any(has_true):
        to_resample = torch.where(has_true)[0]
        eoes_to_resample = eoes[to_resample][:, :-1]
        rev_eoes_to_resample = torch.flip(eoes_to_resample, dims=[1])
        last_true_idx = batch_history.shape[1] - 2 - rev_eoes_to_resample.argmax(axis=1)
        resampled = batch_history[to_resample]
        for i in range(len(to_resample)):
            idx = last_true_idx[i]
            val_to_copy = resampled[i, idx+1]
            resampled[i, :idx+1] = val_to_copy
        batch_history[to_resample] = resampled


# sampling: 0.055, training: 0.068
class NumpyArrayTorchMemoryTMFull(BaseMemory):
    """
    data is stored as concatenated numpy arrays
    """
    def __init__(self,
                 memory_size=None,
                 batch_size=None,
                 dataset_path="",
                 imgs_obs=4,
                 act_buf_len=1,
                 sample_preprocessor: callable = None,
                 crc_debug=False,
                 device="cpu",
                 replace=False,
                 shuffle=False):
        self.replace = replace
        self.shuffle = shuffle
        self.data = []
        self.rng = np.random.default_rng()
        self.imgs_obs = imgs_obs
        self.act_buf_len = act_buf_len
        self.min_samples = max(self.imgs_obs, self.act_buf_len)
        self.start_imgs_offset = max(0, self.min_samples - self.imgs_obs)
        self.start_acts_offset = max(0, self.min_samples - self.act_buf_len)

        # benchmarks
        self.nb_iterations = 0
        self.index_time = 0.0
        self.load_acts_time = 0.0
        self.load_imgs_time = 0.0
        self.move_to_device_time = 0.0
        self.assemble_time = 0.0

        # to remove
        self.debug1_time = 0.0
        self.debug2_time = 0.0
        self.debug3_time = 0.0
        self.debug4_time = 0.0
        self.tensorize_time = 0.0

        super().__init__(memory_size=memory_size,
                         batch_size=batch_size,
                         dataset_path=dataset_path,
                         sample_preprocessor=sample_preprocessor,
                         crc_debug=crc_debug,
                         device=device)

    def __len__(self):
        if len(self.data) == 0:
            return 0
        res = len(self.data[0]) - self.min_samples - 1
        if res < 0:
            return 0
        else:
            return res

    def append_buffer(self, buffer):
        """
        buffer is a list of samples ( act, obs, rew, terminated, truncated, info)
        don't forget to keep the info dictionary in the sample for CRC debugging
        """

        first_data_idx = self.data[0][-1] + 1 if self.__len__() > 0 else 0

        d0 = np.stack([first_data_idx + i for i, _ in enumerate(buffer.memory)])  # indexes
        d1 = np.stack([b[0] for b in buffer.memory])  # actions
        d2 = np.stack([b[1][0] for b in buffer.memory])  # speeds
        d3 = np.stack([b[1][3] for b in buffer.memory])  # images
        d4 = np.stack([b[3] or b[4] for b in buffer.memory])  # eoes
        d5 = np.stack([b[2] for b in buffer.memory])  # rewards
        d6 = [b[5] for b in buffer.memory]  # infos
        d7 = np.stack([b[1][1] for b in buffer.memory])  # gears
        d8 = np.stack([b[1][2] for b in buffer.memory])  # rpms
        d9 = np.stack([b[3] for b in buffer.memory])  # terminated
        d10 = np.stack([b[4] for b in buffer.memory])  # truncated

        if self.__len__() > 0:
            self.data[0] = np.concatenate((self.data[0], d0))
            self.data[1] = np.concatenate((self.data[1], d1))
            self.data[2] = np.concatenate((self.data[2], d2))
            self.data[3] = np.concatenate((self.data[3], d3))
            self.data[4] = np.concatenate((self.data[4], d4))
            self.data[5] = np.concatenate((self.data[5], d5))
            self.data[6] += d6  # infos
            self.data[7] = np.concatenate((self.data[7], d7))
            self.data[8] = np.concatenate((self.data[8], d8))
            self.data[9] = np.concatenate((self.data[9], d9))
            self.data[10] = np.concatenate((self.data[10], d10))
        else:
            self.data.append(d0)
            self.data.append(d1)
            self.data.append(d2)
            self.data.append(d3)
            self.data.append(d4)
            self.data.append(d5)
            self.data.append(d6)
            self.data.append(d7)
            self.data.append(d8)
            self.data.append(d9)
            self.data.append(d10)

        to_trim = self.__len__() - self.memory_size
        if to_trim > 0:
            self.data[0] = self.data[0][to_trim:]
            self.data[1] = self.data[1][to_trim:]
            self.data[2] = self.data[2][to_trim:]
            self.data[3] = self.data[3][to_trim:]
            self.data[4] = self.data[4][to_trim:]
            self.data[5] = self.data[5][to_trim:]
            self.data[6] = self.data[6][to_trim:]
            self.data[7] = self.data[7][to_trim:]
            self.data[8] = self.data[8][to_trim:]
            self.data[9] = self.data[9][to_trim:]
            self.data[10] = self.data[10][to_trim:]

        return self

    def load_batch_acts(self, indices):
        offset_indices = indices + self.start_acts_offset
        full_indices = offset_indices[:, np.newaxis] + np.arange(self.act_buf_len + 1)
        res = self.data[1][full_indices]
        return res

    def load_batch_imgs(self, indices):
        offset_indices = indices + self.start_imgs_offset
        full_indices = offset_indices[:, np.newaxis] + np.arange(self.imgs_obs + 1)
        res = self.data[3][full_indices]
        return np.stack(res).astype(np.float32) / 256.0

    def load_batch_eoes(self, indices):
        full_indices = indices[:, np.newaxis] + np.arange(self.min_samples + 1)
        res = self.data[4][full_indices]
        return res

    def sample_batch_indices(self):  # FIXME
        max_idx = len(self) - 1  # this takes self.min_samples into account

        # sample indices in replay buffer:
        indices = self.rng.choice(a=max_idx, size=self.batch_size, replace=self.replace if max_idx > self.batch_size else True, shuffle=self.shuffle)
        dones = self.data[4][indices + self.min_samples - 1]  # idx_last

        # resample indices that refer to invalid transitions from terminal to initial states:
        # TODO: find a way to only index valid transitions instead
        while np.any(dones):
            to_resample = np.where(dones)[0]
            indices[to_resample] = self.rng.choice(a=max_idx, size=len(to_resample), replace=self.replace if max_idx > len(to_resample) else True, shuffle=self.shuffle)
            dones[to_resample] = self.data[4][indices[to_resample] + self.min_samples - 1]

        return indices

    def sample(self):
        t_0 = time.perf_counter()

        indices = self.sample_batch_indices()

        idx_last = indices + self.min_samples - 1
        idx_now = indices + self.min_samples

        t_1 = time.perf_counter()

        acts = self.load_batch_acts(indices)

        t_1_1 = time.perf_counter()

        last_act_buf = acts[:, :-1]
        new_act_buf = acts[:, 1:]

        t_2 = time.perf_counter()

        imgs = self.load_batch_imgs(indices)

        t_2_1 = time.perf_counter()

        imgs_last_obs = imgs[:, :-1]
        imgs_new_obs = imgs[:, 1:]

        t_3 = time.perf_counter()

        # if a reset transition has influenced the observation, special care must be taken

        eoes = self.load_batch_eoes(indices)

        t_3_0_1 = time.perf_counter()

        eoes_prev_acts = eoes[:, -self.act_buf_len-1:-1]
        eoes_new_acts = eoes[:, -self.act_buf_len:]
        eoes_prev_imgs = eoes[:, -self.imgs_obs - 1:-1]
        eoes_new_imgs = eoes[:, -self.imgs_obs:]

        t_3_0_2 = time.perf_counter()

        # NB: buffers are changed in-place
        fix_batch_history_around_eoes_numpy(last_act_buf, eoes_prev_acts)
        fix_batch_history_around_eoes_numpy(new_act_buf, eoes_new_acts)
        fix_batch_history_around_eoes_numpy(imgs_last_obs, eoes_prev_imgs)
        fix_batch_history_around_eoes_numpy(imgs_new_obs, eoes_new_imgs)

        t_3_0_3 = time.perf_counter()

        t1 = torch.tensor(self.data[2][idx_last], dtype=torch.float32)  # speed prev
        t2 = torch.tensor(self.data[7][idx_last], dtype=torch.float32)  # gear prev
        t3 = torch.tensor(self.data[8][idx_last], dtype=torch.float32)  # rpm prev
        t4 = torch.tensor(imgs_last_obs, dtype=torch.float32)  # imgs_buf prev
        t5 = torch.tensor(last_act_buf.swapaxes(0, 1), dtype=torch.float32)  # act_buf prev
        t6 = torch.tensor(self.data[1][idx_now], dtype=torch.float32)  # act
        t7 = torch.tensor(self.data[5][idx_now], dtype=torch.float32)  # rew
        t8 = torch.tensor(self.data[2][idx_now], dtype=torch.float32)  # speed
        t9 = torch.tensor(self.data[7][idx_now], dtype=torch.float32)  # gear
        t10 = torch.tensor(self.data[8][idx_now], dtype=torch.float32)  # rpm
        t11 = torch.tensor(imgs_new_obs, dtype=torch.float32)  # img_buf
        t12 = torch.tensor(new_act_buf.swapaxes(0, 1), dtype=torch.float32)  # act_buf
        t13 = torch.tensor(self.data[9][idx_now], dtype=torch.float32)  # terminated
        t14 = torch.tensor(self.data[10][idx_now], dtype=torch.float32)  # truncated

        t_3_1 = time.perf_counter()

        t1 = t1.to(self.device)
        t2 = t2.to(self.device)
        t3 = t3.to(self.device)
        t4 = t4.to(self.device)
        t5 = t5.to(self.device)
        t6 = t6.to(self.device)
        t7 = t7.to(self.device)
        t8 = t8.to(self.device)
        t9 = t9.to(self.device)
        t10 = t10.to(self.device)
        t11 = t11.to(self.device)
        t12 = t12.to(self.device)
        t13 = t13.to(self.device)
        t14 = t14.to(self.device)

        t_4 = time.perf_counter()

        last_obs_batch = (t1, t2, t3, t4, *t5)
        new_act_batch = t6
        rew_batch = t7
        new_obs_batch = (t8, t9, t10, t11, *t12)
        terminated_batch = t13
        truncated_batch = t14

        t_5 = time.perf_counter()

        self.nb_iterations += 1
        self.index_time += t_1 - t_0
        self.load_acts_time += t_1_1 - t_1
        self.load_imgs_time += t_2_1 - t_2
        self.move_to_device_time += t_4 - t_3_1
        self.assemble_time += t_5 - t_4

        self.debug1_time += t_3_0_1 - t_3
        self.debug2_time += t_3_0_2 - t_3_0_1
        self.debug3_time += t_3_0_3 - t_3_0_2
        self.debug4_time += t_5 - t_0

        self.tensorize_time += t_3_1 - t_3_0_3

        return last_obs_batch, new_act_batch, rew_batch, new_obs_batch, terminated_batch, truncated_batch

    def get_benchmarks(self):

        if self.nb_iterations == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        else:
            index_time = self.index_time / self.nb_iterations
            load_acts_time = self.load_acts_time / self.nb_iterations
            load_imgs_time = self.load_imgs_time / self.nb_iterations
            move_to_device_time = self.move_to_device_time / self.nb_iterations
            assemble_time = self.assemble_time / self.nb_iterations

            debug1_time = self.debug1_time / self.nb_iterations
            debug2_time = self.debug2_time / self.nb_iterations
            debug3_time = self.debug3_time / self.nb_iterations
            debug4_time = self.debug4_time / self.nb_iterations
            tensorize_time = self.tensorize_time / self.nb_iterations

            self.nb_iterations = 0
            self.index_time = 0.0
            self.load_acts_time = 0.0
            self.load_imgs_time = 0.0
            self.move_to_device_time = 0.0
            self.assemble_time = 0.0

            self.debug1_time = 0.0
            self.debug2_time = 0.0
            self.debug3_time = 0.0
            self.debug4_time = 0.0
            self.tensorize_time = 0.0

            return index_time, load_acts_time, load_imgs_time, move_to_device_time, assemble_time, debug1_time, debug2_time, debug3_time, debug4_time, tensorize_time

    def get_benchmarks_names(self):
        return "index", "load_acts", "load_imgs", "move_to_device", "assemble", "debug1", "debug2", "debug3", "debug4", "tensorize"


# sampling: 0.0248, training: 0.069
class ArrayTorchMemoryTMFull(BaseMemory):
    """
    data stored as concatenated torch tensors
    """
    def __init__(self,
                 memory_size=None,
                 batch_size=None,
                 dataset_path="",
                 imgs_obs=4,
                 act_buf_len=1,
                 sample_preprocessor: callable = None,
                 crc_debug=False,
                 device="cpu",
                 replace=False,
                 shuffle=False):
        self.replace = replace
        self.shuffle = shuffle
        self.data = []
        self.rng = np.random.default_rng()
        self.imgs_obs = imgs_obs
        self.act_buf_len = act_buf_len
        self.min_samples = max(self.imgs_obs, self.act_buf_len)
        self.start_imgs_offset = max(0, self.min_samples - self.imgs_obs)
        self.start_acts_offset = max(0, self.min_samples - self.act_buf_len)

        # benchmarks
        self.nb_iterations = 0
        self.index_time = 0.0
        self.load_acts_time = 0.0
        self.load_imgs_time = 0.0
        self.fix_histories_time = 0.0
        self.pin_memory_time = 0.0
        self.move_to_device_time = 0.0
        self.assemble_time = 0.0

        super().__init__(memory_size=memory_size,
                         batch_size=batch_size,
                         dataset_path=dataset_path,
                         sample_preprocessor=sample_preprocessor,
                         crc_debug=crc_debug,
                         device=device)

    def __len__(self):
        if len(self.data) == 0:
            return 0
        res = len(self.data[0]) - self.min_samples - 1
        if res < 0:
            return 0
        else:
            return res

    def append_buffer(self, buffer):
        """
        buffer is a list of samples ( act, obs, rew, terminated, truncated, info)
        don't forget to keep the info dictionary in the sample for CRC debugging
        """

        with torch.no_grad():

            first_data_idx = self.data[0][-1] + 1 if self.__len__() > 0 else 0

            d0 = np.stack([first_data_idx + i for i, _ in enumerate(buffer.memory)])  # indexes
            d1 = torch.stack([torch.tensor(b[0], dtype=torch.float32) for b in buffer.memory])  # actions
            d2 = torch.stack([torch.tensor(b[1][0], dtype=torch.float32) for b in buffer.memory])  # speeds
            d3 = torch.stack([torch.tensor(b[1][3], dtype=torch.float32) for b in buffer.memory])  # images
            d4 = np.stack([b[3] or b[4] for b in buffer.memory])  # eoe
            d5 = torch.stack([torch.tensor(b[2], dtype=torch.float32) for b in buffer.memory])  # rewards
            d6 = [b[5] for b in buffer.memory]  # infos
            d7 = torch.stack([torch.tensor(b[1][1], dtype=torch.float32) for b in buffer.memory])  # gears
            d8 = torch.stack([torch.tensor(b[1][2], dtype=torch.float32) for b in buffer.memory])  # rpms
            d9 = torch.stack([torch.tensor(b[3], dtype=torch.float32) for b in buffer.memory])  # terminated
            d10 = torch.stack([torch.tensor(b[4], dtype=torch.float32) for b in buffer.memory])  # truncated

            if self.__len__() > 0:
                self.data[0] = np.concatenate((self.data[0], d0))
                self.data[1] = torch.concatenate((self.data[1], d1))
                self.data[2] = torch.concatenate((self.data[2], d2))
                self.data[3] = torch.concatenate((self.data[3], d3))
                self.data[4] = np.concatenate((self.data[4], d4))  # eoe
                self.data[5] = torch.concatenate((self.data[5], d5))
                self.data[6] += d6  # infos
                self.data[7] = torch.concatenate((self.data[7], d7))
                self.data[8] = torch.concatenate((self.data[8], d8))
                self.data[9] = torch.concatenate((self.data[9], d9))
                self.data[10] = torch.concatenate((self.data[10], d10))
            else:
                self.data.append(d0)
                self.data.append(d1)
                self.data.append(d2)
                self.data.append(d3)
                self.data.append(d4)
                self.data.append(d5)
                self.data.append(d6)
                self.data.append(d7)
                self.data.append(d8)
                self.data.append(d9)
                self.data.append(d10)

            to_trim = self.__len__() - self.memory_size
            if to_trim > 0:
                self.data[0] = self.data[0][to_trim:]
                self.data[1] = self.data[1][to_trim:]
                self.data[2] = self.data[2][to_trim:]
                self.data[3] = self.data[3][to_trim:]
                self.data[4] = self.data[4][to_trim:]
                self.data[5] = self.data[5][to_trim:]
                self.data[6] = self.data[6][to_trim:]
                self.data[7] = self.data[7][to_trim:]
                self.data[8] = self.data[8][to_trim:]
                self.data[9] = self.data[9][to_trim:]
                self.data[10] = self.data[10][to_trim:]

            return self

    def load_batch_acts(self, indices):
        offset_indices = indices + self.start_acts_offset
        full_indices = offset_indices[:, np.newaxis] + np.arange(self.act_buf_len + 1)
        res = self.data[1][full_indices]
        return res

    def load_batch_imgs(self, indices):
        offset_indices = indices + self.start_imgs_offset
        full_indices = offset_indices[:, np.newaxis] + np.arange(self.imgs_obs + 1)
        res = self.data[3][full_indices] / 256.0
        return res

    def sample_batch_indices(self):  # FIXME
        max_idx = len(self) - 1  # this takes self.min_samples into account

        # sample indices in replay buffer:
        indices = self.rng.choice(a=max_idx, size=self.batch_size, replace=self.replace if max_idx > self.batch_size else True, shuffle=self.shuffle)
        dones = self.data[4][indices + self.min_samples - 1]  # idx_last

        # resample indices that refer to invalid transitions from terminal to initial states:
        # TODO: find a way to only index valid transitions instead
        while np.any(dones):
            to_resample = np.where(dones)[0]
            indices[to_resample] = self.rng.choice(a=max_idx, size=len(to_resample), replace=self.replace if max_idx > len(to_resample) else True, shuffle=self.shuffle)
            dones[to_resample] = self.data[4][indices[to_resample] + self.min_samples - 1]

        return indices

    def load_batch_eoes(self, indices):
        full_indices = indices[:, np.newaxis] + np.arange(self.min_samples + 1)
        res = self.data[4][full_indices]
        return res

    def sample(self):

        with torch.no_grad():

            t_0 = time.perf_counter()

            indices = self.sample_batch_indices()

            idx_last = indices + self.min_samples - 1
            idx_now = indices + self.min_samples

            t_1 = time.perf_counter()

            acts = self.load_batch_acts(indices)

            last_act_buf = acts[:, :-1]
            new_act_buf = acts[:, 1:]

            t_2 = time.perf_counter()

            imgs = self.load_batch_imgs(indices)

            imgs_last_obs = imgs[:, :-1]
            imgs_new_obs = imgs[:, 1:]

            t_3 = time.perf_counter()

            # if a reset transition has influenced the observation, special care must be taken
            eoes = self.load_batch_eoes(indices)

            eoes_prev_acts = eoes[:, -self.act_buf_len - 1:-1]
            eoes_new_acts = eoes[:, -self.act_buf_len:]
            eoes_prev_imgs = eoes[:, -self.imgs_obs - 1:-1]
            eoes_new_imgs = eoes[:, -self.imgs_obs:]

            fix_batch_history_around_eoes_numpy(last_act_buf, eoes_prev_acts)
            fix_batch_history_around_eoes_numpy(new_act_buf, eoes_new_acts)
            fix_batch_history_around_eoes_numpy(imgs_last_obs, eoes_prev_imgs)
            fix_batch_history_around_eoes_numpy(imgs_new_obs, eoes_new_imgs)

            t_4 = time.perf_counter()

            last_act_buf = last_act_buf.pin_memory()
            new_act_buf = new_act_buf.pin_memory()
            imgs_last_obs = imgs_last_obs.pin_memory()
            imgs_new_obs = imgs_new_obs.pin_memory()

            t_5 = time.perf_counter()

            # CRC:  # TODO
            # info = self.data[6][idx_now]

            t1 = self.data[2][idx_last]
            t2 = self.data[7][idx_last]
            t3 = self.data[8][idx_last]
            t4 = imgs_last_obs
            t5 = last_act_buf.swapaxes(0, 1)
            t6 = self.data[1][idx_now]
            t7 = self.data[5][idx_now]
            t8 = self.data[2][idx_now]
            t9 = self.data[7][idx_now]
            t10 = self.data[8][idx_now]
            t11 = imgs_new_obs
            t12 = new_act_buf.swapaxes(0, 1)
            t13 = self.data[9][idx_now]
            t14 = self.data[10][idx_now]

            t_6 = time.perf_counter()

            t1 = t1.to(self.device, non_blocking=True)
            t2 = t2.to(self.device, non_blocking=True)
            t3 = t3.to(self.device, non_blocking=True)
            t4 = t4.to(self.device, non_blocking=True)
            t5 = t5.to(self.device, non_blocking=True)
            t6 = t6.to(self.device, non_blocking=True)
            t7 = t7.to(self.device, non_blocking=True)
            t8 = t8.to(self.device, non_blocking=True)
            t9 = t9.to(self.device, non_blocking=True)
            t10 = t10.to(self.device, non_blocking=True)
            t11 = t11.to(self.device, non_blocking=True)
            t12 = t12.to(self.device, non_blocking=True)
            t13 = t13.to(self.device, non_blocking=True)
            t14 = t14.to(self.device, non_blocking=True)

            t_7 = time.perf_counter()

            last_obs_batch = (t1, t2, t3, t4, *t5)
            new_act_batch = t6
            rew_batch = t7
            new_obs_batch = (t8, t9, t10, t11, *t12)
            terminated_batch = t13
            truncated_batch = t14

            t_8 = time.perf_counter()

            # CRC-debug numpy batched tensors:
            if self.crc_debug:
                for i in range(len(idx_now)):
                    prev_obs = tuple(x[i].cpu().detach().numpy() for x in last_obs_batch)
                    new_act = new_act_batch[i].cpu().detach().numpy()
                    rew = rew_batch[i].cpu().detach().numpy()
                    new_obs = tuple(x[i].cpu().detach().numpy() for x in new_obs_batch)
                    terminated = terminated_batch[i].cpu().detach().numpy()
                    truncated = truncated_batch[i].cpu().detach().numpy()
                    info = self.data[6][idx_now[i]]
                    po, a, o, r, d, t = info['crc_sample']
                    debug_ts, debug_ts_res = info['crc_sample_ts']
                    check_samples_crc(po, a, o, r, d, t, prev_obs, new_act, new_obs, rew, terminated, truncated, debug_ts, debug_ts_res, epsilon=1e-4)

            if self.sample_preprocessor is not None:
                raise RuntimeError("Sample preprocessor support not implemented here.")

            self.nb_iterations += 1
            self.index_time += t_1 - t_0
            self.load_acts_time += t_2 - t_1
            self.load_imgs_time += t_3 - t_2
            self.fix_histories_time += t_4 - t_3
            self.pin_memory_time += t_5 - t_4
            self.move_to_device_time += t_7 - t_6
            self.assemble_time += t_8 - t_7

            return last_obs_batch, new_act_batch, rew_batch, new_obs_batch, terminated_batch, truncated_batch

    def get_benchmarks(self):

        if self.nb_iterations == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        else:
            index_time = self.index_time / self.nb_iterations
            load_acts_time = self.load_acts_time / self.nb_iterations
            load_imgs_time = self.load_imgs_time / self.nb_iterations
            fix_histories_time = self.fix_histories_time / self.nb_iterations
            pin_memory_time = self.pin_memory_time / self.nb_iterations
            move_to_device_time = self.move_to_device_time / self.nb_iterations
            assemble_time = self.assemble_time / self.nb_iterations

            self.nb_iterations = 0
            self.index_time = 0.0
            self.load_acts_time = 0.0
            self.load_imgs_time = 0.0
            self.fix_histories_time = 0.0
            self.pin_memory_time = 0.0
            self.move_to_device_time = 0.0
            self.assemble_time = 0.0

            return index_time, load_acts_time, load_imgs_time, fix_histories_time, pin_memory_time, move_to_device_time, assemble_time

    def get_benchmarks_names(self):
        return "index", "load_acts", "load_imgs", "fix_histories", "pin_memory", "move_to_device", "assemble"


class ArrayTorchMemoryTMFullSequence(ArrayTorchMemoryTMFull):
    """Episode-safe contiguous sequence sampler for recurrent world models.

    The underlying storage remains identical to :class:`ArrayTorchMemoryTMFull`,
    so existing replay checkpoints can be reused when switching algorithms.
    Returned tensors have leading dimensions ``[batch, time, ...]`` and an
    additional ``is_first`` mask marks the truncated recurrent start of each
    sampled sequence.
    """

    _REPLAY_MODE_CODES = {"uniform": 0.0, "euclidean": 1.0, "poincare": 2.0}

    def __init__(
        self,
        sequence_length=16,
        replay_sampler="uniform",
        replay_candidate_count=128,
        replay_uniform_fraction=0.25,
        replay_embedding_dim=16,
        replay_curvature=1.0,
        replay_tangent_scale=0.75,
        replay_max_radius=0.95,
        replay_seed=0,
        **kwargs,
    ):
        self.sequence_length = int(sequence_length)
        if self.sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        super().__init__(**kwargs)

        self._replay_projection = None
        self._replay_projection_shape = None
        self._replay_sampler_calls = 0
        self._replay_distance_total = 0.0
        self._replay_radius_total = 0.0
        self._replay_candidate_total = 0.0
        self.configure_replay_sampler(
            replay_sampler=replay_sampler,
            replay_candidate_count=replay_candidate_count,
            replay_uniform_fraction=replay_uniform_fraction,
            replay_embedding_dim=replay_embedding_dim,
            replay_curvature=replay_curvature,
            replay_tangent_scale=replay_tangent_scale,
            replay_max_radius=replay_max_radius,
            replay_seed=replay_seed,
        )

    @staticmethod
    def _normalize_replay_sampler(mode):
        mode = str(mode).strip().lower()
        if mode in {"hyperbolic", "poincaré"}:
            mode = "poincare"
        if mode not in {"uniform", "euclidean", "poincare"}:
            raise ValueError(
                "replay_sampler must be 'uniform', 'euclidean', or 'poincare'"
            )
        return mode

    def configure_replay_sampler(
        self,
        replay_sampler="uniform",
        replay_candidate_count=128,
        replay_uniform_fraction=0.25,
        replay_embedding_dim=16,
        replay_curvature=1.0,
        replay_tangent_scale=0.75,
        replay_max_radius=0.95,
        replay_seed=0,
    ):
        """Configure the detached sequence-selection index.

        Returns ``True`` when the effective configuration changed.  This is
        intentionally separate from replay storage so checkpoint migration can
        apply new sampling settings without rewriting stored transitions.
        """
        mode = self._normalize_replay_sampler(replay_sampler)
        candidate_count = int(replay_candidate_count)
        uniform_fraction = float(replay_uniform_fraction)
        embedding_dim = int(replay_embedding_dim)
        curvature = float(replay_curvature)
        tangent_scale = float(replay_tangent_scale)
        max_radius = float(replay_max_radius)
        seed = int(replay_seed)

        if candidate_count < 1:
            raise ValueError("replay_candidate_count must be positive")
        if not np.isfinite(uniform_fraction) or not 0.0 <= uniform_fraction <= 1.0:
            raise ValueError("replay_uniform_fraction must be between 0 and 1")
        if embedding_dim < 2:
            raise ValueError("replay_embedding_dim must be at least 2")
        if not np.isfinite(curvature) or curvature <= 0.0:
            raise ValueError("replay_curvature must be finite and positive")
        if not np.isfinite(tangent_scale) or tangent_scale <= 0.0:
            raise ValueError("replay_tangent_scale must be finite and positive")
        if not np.isfinite(max_radius) or not 0.0 < max_radius < 1.0:
            raise ValueError("replay_max_radius must be between 0 and 1")

        configuration = (
            mode,
            candidate_count,
            uniform_fraction,
            embedding_dim,
            curvature,
            tangent_scale,
            max_radius,
            seed,
        )
        previous = tuple(
            getattr(self, name, None)
            for name in (
                "replay_sampler",
                "replay_candidate_count",
                "replay_uniform_fraction",
                "replay_embedding_dim",
                "replay_curvature",
                "replay_tangent_scale",
                "replay_max_radius",
                "replay_seed",
            )
        )
        projection_changed = (
            getattr(self, "replay_embedding_dim", None) != embedding_dim
            or getattr(self, "replay_seed", None) != seed
        )

        (
            self.replay_sampler,
            self.replay_candidate_count,
            self.replay_uniform_fraction,
            self.replay_embedding_dim,
            self.replay_curvature,
            self.replay_tangent_scale,
            self.replay_max_radius,
            self.replay_seed,
        ) = configuration

        if projection_changed or not hasattr(self, "_replay_projection"):
            self._replay_projection = None
            self._replay_projection_shape = None
        for name, initial in (
            ("_replay_sampler_calls", 0),
            ("_replay_distance_total", 0.0),
            ("_replay_radius_total", 0.0),
            ("_replay_candidate_total", 0.0),
        ):
            if not hasattr(self, name):
                setattr(self, name, initial)

        changed = previous != configuration
        if changed:
            logging.info(
                "Dreamer replay sampler configured: mode=%s, candidates=%d, "
                "uniform_fraction=%.2f, embed_dim=%d, curvature=%.3f.",
                mode,
                candidate_count,
                uniform_fraction,
                embedding_dim,
                curvature,
            )
        return changed

    def _ensure_replay_sampler_state(self):
        """Supply defaults when loading a checkpoint created before this index."""
        missing_configuration = any(
            not hasattr(self, name)
            for name in (
                "replay_sampler",
                "replay_candidate_count",
                "replay_uniform_fraction",
                "replay_embedding_dim",
                "replay_curvature",
                "replay_tangent_scale",
                "replay_max_radius",
                "replay_seed",
            )
        )
        if missing_configuration:
            configuration = {
                name: getattr(self, name, default)
                for name, default in REPLAY_SAMPLER_DEFAULTS.items()
            }
            self.configure_replay_sampler(**configuration)
            return

        if not hasattr(self, "_replay_projection"):
            self._replay_projection = None
            self._replay_projection_shape = None
        for name, initial in (
            ("_replay_sampler_calls", 0),
            ("_replay_distance_total", 0.0),
            ("_replay_radius_total", 0.0),
            ("_replay_candidate_total", 0.0),
        ):
            if not hasattr(self, name):
                setattr(self, name, initial)

    def _sample_episode_safe_starts(self, count):
        max_start_exclusive = len(self) - self.sequence_length + 1
        if max_start_exclusive <= 0:
            raise RuntimeError(
                f"Replay needs at least {self.sequence_length + self.min_samples} "
                "samples before sequence training."
            )

        count = int(count)
        replace = max_start_exclusive < count or self.replace
        starts = self.rng.choice(
            a=max_start_exclusive,
            size=count,
            replace=replace,
            shuffle=self.shuffle,
        )
        offsets = self.min_samples - 1 + np.arange(self.sequence_length)

        # eoe at idx_last means this transition would jump from a terminal
        # sample to the next episode. A terminal idx_now is allowed only at the
        # final transition because it appears as idx_last on the following step.
        invalid = self.data[4][starts[:, np.newaxis] + offsets].any(axis=1)
        attempts = 0
        while np.any(invalid):
            count_invalid = int(invalid.sum())
            starts[invalid] = self.rng.choice(
                a=max_start_exclusive,
                size=count_invalid,
                replace=True,
                shuffle=self.shuffle,
            )
            invalid = self.data[4][starts[:, np.newaxis] + offsets].any(axis=1)
            attempts += 1
            if attempts >= 1000:
                raise RuntimeError(
                    "Could not find enough episode-safe replay sequences; "
                    "reduce DREAMER.SEQUENCE_LENGTH or collect longer episodes."
                )
        return np.asarray(starts, dtype=np.int64)

    @staticmethod
    def _series_summary(series):
        series = series.reshape(series.shape[0], series.shape[1], -1).mean(dim=-1)
        return torch.stack(
            (
                series.mean(dim=1),
                series.std(dim=1, unbiased=False),
                series[:, 0],
                series[:, -1],
                series[:, -1] - series[:, 0],
                series.amin(dim=1),
                series.amax(dim=1),
            ),
            dim=1,
        )

    def _sequence_descriptors(self, starts):
        """Build cheap, detached descriptors for candidate replay sequences."""
        starts = np.asarray(starts, dtype=np.int64)
        sequence_indices = starts[:, np.newaxis] + self.min_samples + np.arange(
            self.sequence_length
        )
        sequence_indices = torch.as_tensor(sequence_indices, dtype=torch.long)
        candidate_count = len(starts)

        actions = self.data[1][sequence_indices].detach().float().cpu()
        actions = actions.reshape(candidate_count, self.sequence_length, -1)
        action_summary = torch.cat(
            (
                actions.mean(dim=1),
                actions.std(dim=1, unbiased=False),
                actions[:, 0],
                actions[:, -1],
                actions[:, -1] - actions[:, 0],
            ),
            dim=1,
        )
        speed = self.data[2][sequence_indices].detach().float().cpu() / 300.0
        gear = self.data[7][sequence_indices].detach().float().cpu() / 6.0
        rpm = self.data[8][sequence_indices].detach().float().cpu() / 10000.0
        reward = self.data[5][sequence_indices].detach().float().cpu()
        reward = torch.sign(reward) * torch.log1p(reward.abs())

        center_indices = sequence_indices[:, self.sequence_length // 2]
        images = self.data[3][center_indices].detach().float().cpu() / 256.0
        if images.ndim == 2:
            images = images.reshape(candidate_count, 1, 1, -1)
        elif images.ndim == 3:
            images = images.unsqueeze(1)
        else:
            images = images.reshape(
                candidate_count, -1, images.shape[-2], images.shape[-1]
            ).mean(dim=1, keepdim=True)
        coarse_image = F.adaptive_avg_pool2d(images, (2, 2)).flatten(start_dim=1)
        image_std = images.flatten(start_dim=1).std(dim=1, unbiased=False).unsqueeze(1)

        descriptor = torch.cat(
            (
                action_summary,
                self._series_summary(speed),
                self._series_summary(gear),
                self._series_summary(rpm),
                self._series_summary(reward),
                coarse_image,
                image_std,
            ),
            dim=1,
        )
        return descriptor.numpy().astype(np.float32, copy=False)

    def _project_replay_descriptors(self, descriptors):
        descriptors = np.asarray(descriptors, dtype=np.float32)
        center = descriptors.mean(axis=0, keepdims=True)
        scale = descriptors.std(axis=0, keepdims=True)
        standardized = np.clip(
            (descriptors - center) / np.maximum(scale, 1e-5), -6.0, 6.0
        )

        projection_shape = (standardized.shape[1], self.replay_embedding_dim)
        if (
            self._replay_projection is None
            or self._replay_projection_shape != projection_shape
        ):
            projection_seed = (
                self.replay_seed
                + 1009 * projection_shape[0]
                + 9176 * projection_shape[1]
            ) % (2**32)
            projection_rng = np.random.default_rng(projection_seed)
            self._replay_projection = projection_rng.normal(
                0.0,
                1.0 / np.sqrt(projection_shape[0]),
                size=projection_shape,
            ).astype(np.float32)
            self._replay_projection_shape = projection_shape

        tangent = standardized @ self._replay_projection
        tangent *= self.replay_tangent_scale / np.sqrt(self.replay_embedding_dim)
        return tangent.astype(np.float32, copy=False)

    def _record_replay_metrics(self, candidate_count, distance, radius):
        self._replay_sampler_calls += 1
        self._replay_candidate_total += float(candidate_count)
        self._replay_distance_total += float(distance)
        self._replay_radius_total += float(radius)

    def _sample_diverse_sequence_starts(self):
        candidates = self._sample_episode_safe_starts(
            max(self.batch_size, self.replay_candidate_count)
        )
        _, first_positions = np.unique(candidates, return_index=True)
        unique_starts = candidates[np.sort(first_positions)]
        descriptors = self._sequence_descriptors(unique_starts)
        tangent = self._project_replay_descriptors(descriptors)

        if self.replay_sampler == "poincare":
            points = poincare_expmap0(
                tangent,
                curvature=self.replay_curvature,
                max_radius=self.replay_max_radius,
            )
            distances = pairwise_poincare_distance(
                points, curvature=self.replay_curvature
            )
            radii = np.sqrt(self.replay_curvature) * np.linalg.norm(points, axis=1)
        else:
            points = tangent
            distances = pairwise_euclidean_distance(points)
            radii = np.linalg.norm(points, axis=1)

        selection_count = min(self.batch_size, len(unique_starts))
        anchor_count = max(
            1,
            int(round(selection_count * self.replay_uniform_fraction)),
        )
        selected = greedy_diverse_subset(
            distances,
            selection_count,
            self.rng,
            initial_count=anchor_count,
        )
        starts = unique_starts[selected]
        if len(starts) < self.batch_size:
            starts = np.concatenate(
                (
                    starts,
                    self.rng.choice(
                        candidates,
                        size=self.batch_size - len(starts),
                        replace=True,
                    ),
                )
            )

        selected_distances = distances[np.ix_(selected, selected)]
        self._record_replay_metrics(
            candidate_count=len(unique_starts),
            distance=mean_off_diagonal(selected_distances),
            radius=float(radii[selected].mean()) if len(selected) else 0.0,
        )
        return np.asarray(starts, dtype=np.int64)

    def sample_sequence_indices(self):
        self._ensure_replay_sampler_state()
        if self.replay_sampler == "uniform":
            starts = self._sample_episode_safe_starts(self.batch_size)
            self._record_replay_metrics(
                candidate_count=len(starts), distance=0.0, radius=0.0
            )
            return starts
        return self._sample_diverse_sequence_starts()

    def get_benchmarks(self):
        self._ensure_replay_sampler_state()
        base_benchmarks = tuple(super().get_benchmarks())
        # The parent has a legacy five-value empty branch despite seven names.
        if len(base_benchmarks) < 7:
            base_benchmarks += (0.0,) * (7 - len(base_benchmarks))

        calls = self._replay_sampler_calls
        if calls:
            candidate_count = self._replay_candidate_total / calls
            mean_distance = self._replay_distance_total / calls
            mean_radius = self._replay_radius_total / calls
        else:
            candidate_count = mean_distance = mean_radius = 0.0
        self._replay_sampler_calls = 0
        self._replay_candidate_total = 0.0
        self._replay_distance_total = 0.0
        self._replay_radius_total = 0.0
        return (
            *base_benchmarks,
            self._REPLAY_MODE_CODES[self.replay_sampler],
            candidate_count,
            mean_distance,
            mean_radius,
        )

    def get_benchmarks_names(self):
        return (
            *super().get_benchmarks_names(),
            "replay_sampler_code",
            "replay_candidate_count",
            "replay_mean_distance",
            "replay_mean_radius",
        )

    def sample(self):
        if self.crc_debug:
            raise RuntimeError("CRC debug is not implemented for sequence sampling")
        if self.sample_preprocessor is not None:
            raise RuntimeError("Sample preprocessor support is not implemented for sequences")

        with torch.no_grad():
            t0 = time.perf_counter()
            starts = self.sample_sequence_indices()
            batch_size = len(starts)
            time_offsets = np.arange(self.sequence_length)
            indices = (starts[:, np.newaxis] + time_offsets).reshape(-1)
            idx_last = indices + self.min_samples - 1
            idx_now = indices + self.min_samples
            t1 = time.perf_counter()

            acts = self.load_batch_acts(indices)
            last_act_buf = acts[:, :-1]
            new_act_buf = acts[:, 1:]
            t2 = time.perf_counter()

            imgs = self.load_batch_imgs(indices)
            imgs_last_obs = imgs[:, :-1]
            imgs_new_obs = imgs[:, 1:]
            t3 = time.perf_counter()

            eoes = self.load_batch_eoes(indices)
            eoes_prev_acts = eoes[:, -self.act_buf_len - 1:-1]
            eoes_new_acts = eoes[:, -self.act_buf_len:]
            eoes_prev_imgs = eoes[:, -self.imgs_obs - 1:-1]
            eoes_new_imgs = eoes[:, -self.imgs_obs:]
            fix_batch_history_around_eoes_numpy(last_act_buf, eoes_prev_acts)
            fix_batch_history_around_eoes_numpy(new_act_buf, eoes_new_acts)
            fix_batch_history_around_eoes_numpy(imgs_last_obs, eoes_prev_imgs)
            fix_batch_history_around_eoes_numpy(imgs_new_obs, eoes_new_imgs)
            t4 = time.perf_counter()

            if str(self.device).startswith("cuda"):
                last_act_buf = last_act_buf.pin_memory()
                new_act_buf = new_act_buf.pin_memory()
                imgs_last_obs = imgs_last_obs.pin_memory()
                imgs_new_obs = imgs_new_obs.pin_memory()
            t5 = time.perf_counter()

            fields = (
                self.data[2][idx_last],
                self.data[7][idx_last],
                self.data[8][idx_last],
                imgs_last_obs,
                self.data[1][idx_now],
                self.data[5][idx_now],
                self.data[2][idx_now],
                self.data[7][idx_now],
                self.data[8][idx_now],
                imgs_new_obs,
                self.data[9][idx_now],
                self.data[10][idx_now],
            )
            fields = tuple(
                field.to(self.device, non_blocking=True) for field in fields
            )
            last_act_buf = last_act_buf.to(self.device, non_blocking=True)
            new_act_buf = new_act_buf.to(self.device, non_blocking=True)
            t6 = time.perf_counter()

            def sequence(tensor):
                return tensor.reshape(
                    batch_size, self.sequence_length, *tensor.shape[1:]
                )

            (
                speed,
                gear,
                rpm,
                images,
                action,
                reward,
                next_speed,
                next_gear,
                next_rpm,
                next_images,
                terminated,
                truncated,
            ) = (sequence(field) for field in fields)
            last_actions = tuple(
                sequence(last_act_buf[:, j]) for j in range(self.act_buf_len)
            )
            next_actions = tuple(
                sequence(new_act_buf[:, j]) for j in range(self.act_buf_len)
            )
            obs = (speed, gear, rpm, images, *last_actions)
            next_obs = (
                next_speed,
                next_gear,
                next_rpm,
                next_images,
                *next_actions,
            )
            is_first = torch.zeros(
                batch_size,
                self.sequence_length,
                device=self.device,
                dtype=torch.bool,
            )
            is_first[:, 0] = True
            t7 = time.perf_counter()

            self.nb_iterations += 1
            self.index_time += t1 - t0
            self.load_acts_time += t2 - t1
            self.load_imgs_time += t3 - t2
            self.fix_histories_time += t4 - t3
            self.pin_memory_time += t5 - t4
            self.move_to_device_time += t6 - t5
            self.assemble_time += t7 - t6

            return (
                obs,
                action,
                reward,
                next_obs,
                terminated,
                truncated,
                is_first,
            )


# JIT torch memories ===================================================================================================

TORCH_DATA_TYPE_TM20_FULL = List[torch.Tensor]

def load_batch_acts(indices:torch.Tensor, start_acts_offset:int, act_buf_len:int, torch_data:TORCH_DATA_TYPE_TM20_FULL) -> torch.Tensor:
    offset_indices = indices + start_acts_offset
    full_indices = offset_indices.unsqueeze(1) + torch.arange(act_buf_len + 1)
    res = torch_data[0][full_indices]
    return res.pin_memory()

def load_batch_imgs(indices:torch.Tensor, start_imgs_offset:int, imgs_obs:int, torch_data:TORCH_DATA_TYPE_TM20_FULL) -> torch.Tensor:
    offset_indices = indices + start_imgs_offset
    # full_indices = offset_indices[:, np.newaxis] + np.arange(imgs_obs + 1)
    full_indices = offset_indices.unsqueeze(1) + torch.arange(imgs_obs + 1)
    res = torch_data[2][full_indices] / 256.0
    return res.pin_memory()  # TODO: optimize

def get_tensors_tm20_full(indices:torch.Tensor,
                          min_samples:int,
                          start_acts_offset:int,
                          act_buf_len:int,
                          start_imgs_offset:int,
                          imgs_obs:int,
                          torch_data:TORCH_DATA_TYPE_TM20_FULL,
                          device:str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # t_0 = time.perf_counter()

    idx_last = indices + min_samples - 1
    idx_now = indices + min_samples

    # t_1 = time.perf_counter()

    acts = load_batch_acts(indices, start_acts_offset, act_buf_len, torch_data)

    # t_1_1 = time.perf_counter()

    last_act_buf = acts[:, :-1]
    new_act_buf = acts[:, 1:]

    # t_2 = time.perf_counter()

    imgs = load_batch_imgs(indices, start_imgs_offset, imgs_obs, torch_data)

    # t_2_1 = time.perf_counter()

    imgs_last_obs = imgs[:, :-1]
    imgs_new_obs = imgs[:, 1:]

    # t_3 = time.perf_counter()

    # TODO:
    # if a reset transition has influenced the observation, special care must be taken
    # last_eoes = self.data[4][idx_now - self.min_samples:idx_now]  # self.min_samples values
    # last_eoe_idx = last_true_in_list(last_eoes)  # last occurrence of True
    #
    # assert last_eoe_idx is None or last_eoes[last_eoe_idx], f"last_eoe_idx:{last_eoe_idx}"
    #
    # if last_eoe_idx is not None:
    #     replace_hist_before_eoe(hist=new_act_buf, eoe_idx_in_hist=last_eoe_idx - self.start_acts_offset - 1)
    #     replace_hist_before_eoe(hist=last_act_buf, eoe_idx_in_hist=last_eoe_idx - self.start_acts_offset)
    #     replace_hist_before_eoe(hist=imgs_new_obs, eoe_idx_in_hist=last_eoe_idx - self.start_imgs_offset - 1)
    #     replace_hist_before_eoe(hist=imgs_last_obs, eoe_idx_in_hist=last_eoe_idx - self.start_imgs_offset)

    # CRC:  # TODO
    # info = self.data[6][idx_now]

    t1 = torch_data[1][idx_last]  # speed prev
    t2 = torch_data[4][idx_last]  # gear prev
    t3 = torch_data[5][idx_last]  # rpm prev
    t4 = imgs_last_obs  # imgs_buf prev
    t5 = last_act_buf.swapaxes(0, 1)  # act_buf prev
    t6 = torch_data[0][idx_now]  # act
    t7 = torch_data[3][idx_now]  # rew
    t8 = torch_data[1][idx_now]  # speed
    t9 = torch_data[4][idx_now]  # gear
    t10 = torch_data[5][idx_now]  # rpm
    t11 = imgs_new_obs  # img_buf
    t12 = new_act_buf.swapaxes(0, 1)  # act_buf
    t13 = torch_data[6][idx_now]  # terminated
    t14 = torch_data[7][idx_now]  # truncated

    # t_3_1 = time.perf_counter()

    t1 = t1.to(device, non_blocking=True)
    t2 = t2.to(device, non_blocking=True)
    t3 = t3.to(device, non_blocking=True)
    t4 = t4.to(device, non_blocking=True)
    t5 = t5.to(device, non_blocking=True)
    t6 = t6.to(device, non_blocking=True)
    t7 = t7.to(device, non_blocking=True)
    t8 = t8.to(device, non_blocking=True)
    t9 = t9.to(device, non_blocking=True)
    t10 = t10.to(device, non_blocking=True)
    t11 = t11.to(device, non_blocking=True)
    t12 = t12.to(device, non_blocking=True)
    t13 = t13.to(device, non_blocking=True)
    t14 = t14.to(device, non_blocking=True)

    # t_4 = time.perf_counter()

    # t_5 = time.perf_counter()

    return t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14


# sampling: 0.0248, training: 0.069
class NewArrayTorchMemoryTMFull(BaseMemory):
    def __init__(self,
                 memory_size:int = 1e5,
                 batch_size:int = 1e5,
                 dataset_path:str = "",
                 imgs_obs:int = 4,
                 act_buf_len:int = 1,
                 sample_preprocessor: callable = None,
                 crc_debug:bool = False,
                 device:str = "cpu",
                 replace:bool = False,
                 shuffle:bool = False):
        self.replace: bool = replace
        self.shuffle: bool = shuffle
        self.torch_data: List[torch.Tensor] = []
        self.aux_data: List = []
        self.rng = np.random.default_rng()
        self.imgs_obs: int = imgs_obs
        self.act_buf_len: int = act_buf_len
        self.min_samples: int = max(self.imgs_obs, self.act_buf_len)
        self.start_imgs_offset: int = max(0, self.min_samples - self.imgs_obs)
        self.start_acts_offset: int = max(0, self.min_samples - self.act_buf_len)

        # benchmarks
        self.nb_iterations: int = 0
        self.index_time: float = 0.0
        self.load_acts_time: float = 0.0
        self.load_imgs_time: float = 0.0
        self.move_to_device_time: float = 0.0
        self.assemble_time: float = 0.0

        # to remove
        self.debug1_time: float = 0.0
        self.debug2_time: float = 0.0
        self.tensorize_time: float = 0.0

        BaseMemory.__init__(self,
                            memory_size=memory_size,
                            batch_size=batch_size,
                            dataset_path=dataset_path,
                            sample_preprocessor=sample_preprocessor,
                            crc_debug=crc_debug,
                            device=device)
        # torch.jit.ScriptModule.__init__(self)

        self.jit_get_tensors = torch.jit.script(get_tensors_tm20_full)
        # self.jit_get_tensors = get_tensors_tm20_full

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove non-picklable attributes for pickling
        del state['jit_get_tensors']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Recreate non_picklable attribute
        self.jit_get_tensors = torch.jit.script(get_tensors_tm20_full)

    def __len__(self):
        if len(self.aux_data) == 0:
            return 0
        res = len(self.aux_data[0]) - self.min_samples - 1
        if res < 0:
            return 0
        else:
            return res

    def append_buffer(self, buffer):
        """
        buffer is a list of samples ( act, obs, rew, terminated, truncated, info)
        don't forget to keep the info dictionary in the sample for CRC debugging
        """

        with torch.no_grad():

            first_data_idx = self.aux_data[0][-1] + 1 if self.__len__() > 0 else 0

            da0 = np.stack([first_data_idx + i for i, _ in enumerate(buffer.memory)])  # indexes
            da1 = [b[5] for b in buffer.memory]  # infos
            da2 = np.stack([b[3] or b[4] for b in buffer.memory])  # eoe

            dt0 = torch.stack([torch.tensor(b[0], dtype=torch.float32) for b in buffer.memory])  # actions
            dt1 = torch.stack([torch.tensor(b[1][0], dtype=torch.float32) for b in buffer.memory])  # speeds
            dt2 = torch.stack([torch.tensor(b[1][3], dtype=torch.float32) for b in buffer.memory])  # images
            dt3 = torch.stack([torch.tensor(b[2], dtype=torch.float32) for b in buffer.memory])  # rewards
            dt4 = torch.stack([torch.tensor(b[1][1], dtype=torch.float32) for b in buffer.memory])  # gears
            dt5 = torch.stack([torch.tensor(b[1][2], dtype=torch.float32) for b in buffer.memory])  # rpms
            dt6 = torch.stack([torch.tensor(b[3], dtype=torch.float32) for b in buffer.memory])  # terminated
            dt7 = torch.stack([torch.tensor(b[4], dtype=torch.float32) for b in buffer.memory])  # truncated

            if self.__len__() > 0:
                self.aux_data[0] = np.concatenate((self.aux_data[0], da0))  # index
                self.aux_data[1] += da1  # infos
                self.aux_data[2] = np.concatenate((self.aux_data[2], da2))  # eoe

                self.torch_data[0] = torch.concatenate((self.torch_data[0], dt0))
                self.torch_data[1] = torch.concatenate((self.torch_data[1], dt1))
                self.torch_data[2] = torch.concatenate((self.torch_data[2], dt2))
                self.torch_data[3] = torch.concatenate((self.torch_data[3], dt3))
                self.torch_data[4] = torch.concatenate((self.torch_data[4], dt4))
                self.torch_data[5] = torch.concatenate((self.torch_data[5], dt5))
                self.torch_data[6] = torch.concatenate((self.torch_data[6], dt6))
                self.torch_data[7] = torch.concatenate((self.torch_data[7], dt7))
            else:
                self.aux_data.append(da0)
                self.aux_data.append(da1)
                self.aux_data.append(da2)

                self.torch_data.append(dt0)
                self.torch_data.append(dt1)
                self.torch_data.append(dt2)
                self.torch_data.append(dt3)
                self.torch_data.append(dt4)
                self.torch_data.append(dt5)
                self.torch_data.append(dt6)
                self.torch_data.append(dt7)

            to_trim = self.__len__() - self.memory_size
            if to_trim > 0:
                self.aux_data[0] = self.aux_data[0][to_trim:]
                self.aux_data[1] = self.aux_data[1][to_trim:]
                self.aux_data[2] = self.aux_data[2][to_trim:]

                self.torch_data[0] = self.torch_data[0][to_trim:]
                self.torch_data[1] = self.torch_data[1][to_trim:]
                self.torch_data[2] = self.torch_data[2][to_trim:]
                self.torch_data[3] = self.torch_data[3][to_trim:]
                self.torch_data[4] = self.torch_data[4][to_trim:]
                self.torch_data[5] = self.torch_data[5][to_trim:]
                self.torch_data[6] = self.torch_data[6][to_trim:]
                self.torch_data[7] = self.torch_data[7][to_trim:]

            return self

    def sample_batch_indices(self) -> np.ndarray:
        max_idx = len(self) - 1  # this takes self.min_samples into account

        # sample indices in replay buffer:
        indices = self.rng.choice(a=max_idx, size=self.batch_size, replace=self.replace if max_idx > self.batch_size else True, shuffle=self.shuffle)
        dones = self.aux_data[2][indices]

        # resample indices that refer to invalid transitions from terminal to initial states:
        # TODO: find a way to only index valid transitions instead
        while np.any(dones):
            to_resample = np.where(dones)[0]
            indices[to_resample] = self.rng.choice(a=max_idx, size=len(to_resample), replace=self.replace if max_idx > len(to_resample) else True, shuffle=self.shuffle)
            dones[to_resample] = self.aux_data[2][indices[to_resample]]

        return indices

    def sample(self):
        indices = torch.tensor(self.sample_batch_indices(), requires_grad=False)
        t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14 = self.jit_get_tensors(indices=indices,
                                                                                           min_samples=self.min_samples,
                                                                                           start_acts_offset=self.start_acts_offset,
                                                                                           act_buf_len=self.act_buf_len,
                                                                                           start_imgs_offset=self.start_imgs_offset,
                                                                                           imgs_obs=self.imgs_obs,
                                                                                           torch_data=self.torch_data,
                                                                                           device=self.device)
        last_obs_batch = (t1, t2, t3, t4, *t5)
        new_act_batch = t6
        rew_batch = t7
        new_obs_batch = (t8, t9, t10, t11, *t12)
        terminated_batch = t13
        truncated_batch = t14

        return last_obs_batch, new_act_batch, rew_batch, new_obs_batch, terminated_batch, truncated_batch

    def get_benchmarks(self):

        if self.nb_iterations == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        else:
            index_time = self.index_time / self.nb_iterations
            load_acts_time = self.load_acts_time / self.nb_iterations
            load_imgs_time = self.load_imgs_time / self.nb_iterations
            move_to_device_time = self.move_to_device_time / self.nb_iterations
            assemble_time = self.assemble_time / self.nb_iterations

            debug1_time = self.debug1_time / self.nb_iterations
            debug2_time = self.debug2_time / self.nb_iterations
            tensorize_time = self.tensorize_time / self.nb_iterations

            self.nb_iterations = 0
            self.index_time = 0.0
            self.load_acts_time = 0.0
            self.load_imgs_time = 0.0
            self.move_to_device_time = 0.0
            self.assemble_time = 0.0

            self.debug1_time = 0.0
            self.debug2_time = 0.0
            self.tensorize_time = 0.0

            return index_time, load_acts_time, load_imgs_time, move_to_device_time, assemble_time, debug1_time, debug2_time, tensorize_time

    def get_benchmarks_names(self):
        return "index", "load_acts", "load_imgs", "move_to_device", "assemble", "debug1", "debug2", "tensorize"
