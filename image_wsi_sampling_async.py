import ctypes
import gc
import time
import os

import torch.multiprocessing
import numpy as np
import cv2
import tqdm

import image_wsi_sampling
import config

def subprocess_run(image_loading_pipe_recv, subdata_name: str, image_width: int, buffer_max_size: int,
                   shared_image_cat: torch.Tensor, shared_ground_truth: torch.Tensor, shared_ground_truth_mask: torch.Tensor,
                   image_access_lock: torch.multiprocessing.Lock, image_available_lock: torch.multiprocessing.Lock,
                   image_required_flag: torch.multiprocessing.Value, running: torch.multiprocessing.Value,
                   sampling_type:str="random_image"):
    try:
        print("Subprocess starting...")
        sampler = image_wsi_sampling.get_image_sampler(subdata_name, image_width, device="cpu")

        buffered_outputs = []
        pending_images = []
        print("Subprocess running...")

        run_time = 0
        while running.value:
            if image_loading_pipe_recv.poll():
                sample_object = image_loading_pipe_recv.recv()
                pending_images.append(sample_object)

            if ((buffer_max_size == -1) or (len(buffered_outputs) < buffer_max_size)) and (len(pending_images) > 0):
                sample_object = pending_images.pop(0)
                if sampling_type == "random_image":
                    result = sampler.obtain_random_image_from_tile(sample_object) # here its wsi_tile_id
                    buffered_outputs.append(result)
                elif sampling_type == "batch_random_image":
                    tile_ids, augmentation, random_location = sample_object # args for obtain_random_sample_batch
                    length = len(tile_ids)
                    result = sampler.obtain_random_sample_batch(tile_id=tile_ids, augmentation=augmentation, random_location=random_location)
                    buffered_outputs.append((result, length))
                elif sampling_type == "batch_random_image_mixup":
                    tile_id1, tile_id2, mixup_alpha, augmentation = sample_object  # args for obtain_random_sample_batch
                    length = len(tile_id1)
                    result = sampler.obtain_random_sample_with_mixup_batch(tile_id1=tile_id1, tile_id2=tile_id2, mixup_alpha=mixup_alpha, augmentation=augmentation)
                    buffered_outputs.append((result, length))


            if image_required_flag.value and len(buffered_outputs) > 0:
                image_access_lock.acquire(block=True)
                last_result = buffered_outputs.pop(0)

                if sampling_type == "random_image":
                    image_cat, ground_truth, ground_truth_mask = last_result
                    shared_image_cat.copy_(image_cat)
                    shared_ground_truth.copy_(ground_truth)
                    shared_ground_truth_mask.copy_(ground_truth_mask)

                    del image_cat, ground_truth, ground_truth_mask
                    gc.collect()
                else:
                    result, length = last_result
                    image_cat_batch, image_ground_truth_batch, image_ground_truth_mask_batch = result
                    shared_image_cat[:length, ...].copy_(image_cat_batch)
                    shared_ground_truth[:length, ...].copy_(image_ground_truth_batch)
                    shared_ground_truth_mask[:length, ...].copy_(image_ground_truth_mask_batch)

                    del result, length, image_cat_batch, image_ground_truth_batch, image_ground_truth_mask_batch
                    gc.collect()
                image_required_flag.value = False
                image_access_lock.release()
                image_available_lock.release()

            time.sleep(0.01)

            run_time += 1
            if run_time % 100 == 0:
                gc.collect()
    except KeyboardInterrupt:
        print("Interrupted...")

    print("Subprocess terminating...")

def subprocess_run_deep(image_loading_pipe_recv, subdata_name: str, image_width: int, buffer_max_size: int,
                    shared_image_cat: torch.Tensor, shared_ground_truth: torch.Tensor, shared_ground_truth_mask: torch.Tensor,
                    shared_ground_truth_deep: list[torch.Tensor], shared_ground_truth_mask_deep: list[torch.Tensor], deep_supervision_outputs: int,
                   image_access_lock: torch.multiprocessing.Lock, image_available_lock: torch.multiprocessing.Lock,
                   image_required_flag: torch.multiprocessing.Value, running: torch.multiprocessing.Value,
                   sampling_type:str="batch_random_image"):
    try:
        print("Subprocess starting...")
        sampler = image_wsi_sampling.get_image_sampler(subdata_name, image_width, device="cpu")

        buffered_outputs = []
        pending_images = []
        print("Subprocess running...")

        run_time = 0
        while running.value:
            if image_loading_pipe_recv.poll():
                sample_object = image_loading_pipe_recv.recv()
                pending_images.append(sample_object)

            if ((buffer_max_size == -1) or (len(buffered_outputs) < buffer_max_size)) and (len(pending_images) > 0):
                sample_object = pending_images.pop(0)
                if sampling_type == "batch_random_image":
                    tile_ids, augmentation, random_location = sample_object  # args for obtain_random_sample_batch
                    length = len(tile_ids)
                    result = sampler.obtain_random_sample_batch(tile_id=tile_ids, augmentation=augmentation,
                                                                random_location=random_location, deep_supervision_downsamples=deep_supervision_outputs)
                    buffered_outputs.append((result, length))
                elif sampling_type == "batch_random_image_mixup":
                    tile_id1, tile_id2, mixup_alpha, augmentation = sample_object  # args for obtain_random_sample_batch
                    length = len(tile_id1)
                    result = sampler.obtain_random_sample_with_mixup_batch(tile_id1=tile_id1, tile_id2=tile_id2,
                                                                           mixup_alpha=mixup_alpha,
                                                                           augmentation=augmentation, deep_supervision_downsamples=deep_supervision_outputs)
                    buffered_outputs.append((result, length))

            if image_required_flag.value and len(buffered_outputs) > 0:
                image_access_lock.acquire(block=True)

                last_result = buffered_outputs.pop(0)

                result, length = last_result
                image_cat_batch, image_ground_truth_batch, image_ground_truth_mask_batch, image_ground_truth_deep, image_ground_truth_mask_deep = result
                shared_image_cat[:length, ...].copy_(image_cat_batch)
                shared_ground_truth[:length, ...].copy_(image_ground_truth_batch)
                shared_ground_truth_mask[:length, ...].copy_(image_ground_truth_mask_batch)

                for l in range(deep_supervision_outputs):
                    shared_ground_truth_deep[l][:length, ...].copy_(image_ground_truth_deep[l])
                    shared_ground_truth_mask_deep[l][:length, ...].copy_(image_ground_truth_mask_deep[l])

                del result, length, image_cat_batch, image_ground_truth_batch, image_ground_truth_mask_batch, image_ground_truth_deep[:], image_ground_truth_mask_deep[:]
                gc.collect()

                image_required_flag.value = False
                image_access_lock.release()
                image_available_lock.release()

            time.sleep(0.01)

            run_time += 1
            if run_time % 100 == 0:
                gc.collect()
    except KeyboardInterrupt:
        print("Interrupted...")

    print("Subprocess terminating...")

class MultipleImageSamplerAsync:

    def __init__(self, subdata_name:str, image_width: int, batch_size: int=1, sampling_type:str="random_image", deep_supervision_outputs=0, buffer_max_size=-1):
        """
        The sampling types are either "random_image", "batch_random_image", "batch_random_image_mixup".

        :param subdata_name: The name of the subdata to sample from. This is to initialize the image sampler.
        :param image_width: The width of the image.
        :param batch_size: The batch size to use for the batch sampling types.
        :param sampling_type: The type of sampling to use. See above.
        :param deep_supervision_outputs: The number of deep supervision outputs to use.
        :param buffer_max_size: The maximum number of images to buffer. If -1, then the buffer size is unlimited.
        """
        assert sampling_type in ["random_image", "batch_random_image", "batch_random_image_mixup"], "Sampling type must be either random_image, batch_random_image or batch_random_image_mixup"
        assert deep_supervision_outputs == 0 or sampling_type != "random_image", "Deep supervision outputs can only be used with batch sampling types"
        self.batch_size = batch_size
        self.sampling_type = sampling_type
        self.deep_supervision_outputs = deep_supervision_outputs

        image_loading_pipe_recv, self.image_loading_pipe_send = torch.multiprocessing.Pipe(duplex=False)
        self.image_available_lock = torch.multiprocessing.Lock()
        self.image_required_flag = torch.multiprocessing.Value(ctypes.c_bool, True)
        self.image_access_lock = torch.multiprocessing.Lock()
        self.image_available_lock.acquire(block=True)

        if sampling_type == "random_image":
            self.shared_image_cat = torch.zeros((4, image_width, image_width), device=config.device, dtype=torch.float32)
            self.shared_ground_truth = torch.zeros((image_width, image_width), device=config.device, dtype=torch.float32)
            self.shared_ground_truth_mask = torch.zeros((image_width, image_width), device=config.device, dtype=torch.float32)
        elif sampling_type == "batch_random_image":
            self.shared_image_cat = torch.zeros((batch_size, 4, image_width, image_width), device=config.device, dtype=torch.float32)
            self.shared_ground_truth = torch.zeros((batch_size, image_width, image_width), device=config.device, dtype=torch.long)
            self.shared_ground_truth_mask = torch.zeros((batch_size, image_width, image_width), device=config.device, dtype=torch.float32)

            if deep_supervision_outputs > 0:
                self.shared_ground_truth_deep = []
                self.shared_ground_truth_mask_deep = []

                for k in range(deep_supervision_outputs):
                    scale_factor = 2 ** (k + 1)
                    self.shared_ground_truth_deep.append(torch.zeros((batch_size, image_width // scale_factor, image_width // scale_factor), device=config.device, dtype=torch.long))
                    self.shared_ground_truth_mask_deep.append(torch.zeros((batch_size, image_width // scale_factor, image_width // scale_factor), device=config.device, dtype=torch.float32))
        elif sampling_type == "batch_random_image_mixup":
            self.shared_image_cat = torch.zeros((batch_size, 4, image_width, image_width), device=config.device, dtype=torch.float32)
            self.shared_ground_truth = torch.zeros((batch_size, 3, image_width, image_width), device=config.device, dtype=torch.float32)
            self.shared_ground_truth_mask = torch.zeros((batch_size, image_width, image_width), device=config.device, dtype=torch.float32)

            if deep_supervision_outputs > 0:
                self.shared_ground_truth_deep = []
                self.shared_ground_truth_mask_deep = []

                for k in range(deep_supervision_outputs):
                    scale_factor = 2 ** (k + 1)
                    self.shared_ground_truth_deep.append(torch.zeros((batch_size, 3, image_width // scale_factor, image_width // scale_factor), device=config.device, dtype=torch.float32))
                    self.shared_ground_truth_mask_deep.append(torch.zeros((batch_size, image_width // scale_factor, image_width // scale_factor), device=config.device, dtype=torch.float32))

        self.running = torch.multiprocessing.Value(ctypes.c_bool, True)

        if deep_supervision_outputs > 0:
            self.process = torch.multiprocessing.Process(target=subprocess_run_deep,
                                                         args=[image_loading_pipe_recv, subdata_name, image_width, buffer_max_size,
                                                               self.shared_image_cat, self.shared_ground_truth, self.shared_ground_truth_mask,
                                                               self.shared_ground_truth_deep, self.shared_ground_truth_mask_deep, deep_supervision_outputs,
                                                               self.image_access_lock, self.image_available_lock,
                                                               self.image_required_flag, self.running, sampling_type])
        else:
            self.process = torch.multiprocessing.Process(target=subprocess_run, args=[image_loading_pipe_recv, subdata_name, image_width, buffer_max_size,
                                                                                           self.shared_image_cat, self.shared_ground_truth, self.shared_ground_truth_mask,
                                                                                           self.image_access_lock, self.image_available_lock,
                                                                                           self.image_required_flag, self.running, sampling_type])
        self.process.start()

    def terminate(self):
        self.running.value = False

    def request_load_image(self, wsi_ids):
        assert self.sampling_type == "random_image", "Image loading can only be requested when using random_image sampling type."
        if type(wsi_ids) is str:
            self.image_loading_pipe_send.send(wsi_ids)
        elif type(wsi_ids) is list:
            for wsi_id in wsi_ids:
                self.image_loading_pipe_send.send(wsi_id)
        else:
            raise ValueError("Invalid type for wsi_ids, expected list or string.")

    def request_load_sample(self, tile_ids: list[str], augmentation: bool, random_location: bool):
        assert self.sampling_type == "batch_random_image", "Image loading can only be requested when using batch_random_image sampling type."
        self.image_loading_pipe_send.send((tile_ids, augmentation, random_location))

    def request_load_sample_mixup(self, tile_id1: list[str], tile_id2: list[str], mixup_alpha: float, augmentation: bool):
        assert self.sampling_type == "batch_random_image_mixup", "Image loading can only be requested when using batch_random_image_mixup sampling type."
        self.image_loading_pipe_send.send((tile_id1, tile_id2, mixup_alpha, augmentation))

    def get_image(self, device):
        """Get the currently loaded image. WARNING - this is a blocking call.
        If no images are in the pipe, this might create a deadlock."""
        assert self.sampling_type == "random_image", "Image loading can only be requested when using random_image sampling type."
        self.image_available_lock.acquire(block=True)

        self.image_access_lock.acquire(block=True)
        image = self.shared_image_cat.to(device, copy=True)
        ground_truth = self.shared_ground_truth.to(device, copy=True)
        ground_truth_mask = self.shared_ground_truth_mask.to(device, copy=True)

        self.image_required_flag.value = True
        self.image_access_lock.release()

        return image, ground_truth, ground_truth_mask

    def get_samples(self, device, length:int, image_tensor: torch.Tensor=None, ground_truth_tensor: torch.Tensor=None,
                    ground_truth_mask_tensor: torch.Tensor=None, ground_truth_deep_tensors: list[torch.Tensor]=None,
                    ground_truth_mask_deep_tensors: list[torch.Tensor]=None):
        """Get the loaded samples for the batch. WARNING - this is a blocking call."""
        assert self.sampling_type != "random_image", "Sampling can only be used not with the random_image sampling type."
        self.image_available_lock.acquire(block=True)

        self.image_access_lock.acquire(block=True)

        if image_tensor is None:
            # we allocate and create a copy and then return it
            image = self.shared_image_cat[:length, ...].to(device, copy=True)
            ground_truth = self.shared_ground_truth[:length, ...].to(device, copy=True)
            ground_truth_mask = self.shared_ground_truth_mask[:length, ...].to(device, copy=True)
            if self.deep_supervision_outputs > 0:
                ground_truth_deep = []
                ground_truth_mask_deep = []
                for k in range(self.deep_supervision_outputs):
                    ground_truth_deep.append(self.shared_ground_truth_deep[k][:length, ...].to(device, copy=True))
                    ground_truth_mask_deep.append(self.shared_ground_truth_mask_deep[k][:length, ...].to(device, copy=True))
        else:
            # directly copy into the tensors
            image_tensor.copy_(self.shared_image_cat)
            ground_truth_tensor.copy_(self.shared_ground_truth)
            ground_truth_mask_tensor.copy_(self.shared_ground_truth_mask)
            if self.deep_supervision_outputs > 0:
                for k in range(self.deep_supervision_outputs):
                    ground_truth_deep_tensors[k].copy_(self.shared_ground_truth_deep[k])
                    ground_truth_mask_deep_tensors[k].copy_(self.shared_ground_truth_mask_deep[k])

        self.image_required_flag.value = True
        self.image_access_lock.release()

        if image_tensor is None:
            if self.deep_supervision_outputs > 0:
                return image, ground_truth, ground_truth_mask, ground_truth_deep, ground_truth_mask_deep
            return image, ground_truth, ground_truth_mask

def get_image_sampler(subdata_name: str, image_width=1024, batch_size: int=1, sampling_type:str="random_image", deep_supervision_outputs=0, buffer_max_size=-1) -> MultipleImageSamplerAsync:
    return MultipleImageSamplerAsync(subdata_name, image_width, batch_size, sampling_type, deep_supervision_outputs, buffer_max_size)

def generate_image_example(sampler: MultipleImageSamplerAsync, tile: str, num: int) -> float:
    ctime = time.time()
    image_comb, ground_truth, ground_truth_mask = sampler.get_image(device=config.device)
    ctime = time.time() - ctime

    image = image_comb[:3, ...].detach().cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
    region_mask = image_comb[3, ...].detach().cpu().numpy().astype(np.uint8) * 255
    region_mask = np.repeat(np.expand_dims(region_mask, axis=-1), axis=-1, repeats=3)

    ground_truth = ground_truth.detach().cpu().numpy()
    ground_truth_mask = ground_truth_mask.detach().cpu().numpy().astype(np.uint8) * 255
    ground_truth_mask = np.repeat(np.expand_dims(ground_truth_mask, axis=-1), axis=-1, repeats=3)
    classes_image = np.zeros_like(image, dtype=np.uint8)

    # convert ground_truth to hsv
    hue_mask = (255 * ground_truth.astype(np.float32) / 3).astype(np.uint8)
    saturation_mask = ((ground_truth > 0).astype(np.float32) * 255).astype(np.uint8)
    value_mask = saturation_mask

    classes_image[:, :, 0] = hue_mask
    classes_image[:, :, 1] = saturation_mask
    classes_image[:, :, 2] = value_mask

    classes_image = cv2.cvtColor(classes_image, cv2.COLOR_HSV2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    image = cv2.addWeighted(image, 0.5, region_mask, 0.5, 0)
    classes_image = cv2.addWeighted(classes_image, 0.5, ground_truth_mask, 0.5, 0)

    # save images
    if not os.path.isdir(os.path.join(image_wsi_sampling.folder, "examples")):
        os.mkdir(os.path.join(image_wsi_sampling.folder, "examples"))
    cv2.imwrite(os.path.join(image_wsi_sampling.folder, "examples", "{}_{}_image.png".format(tile, num)), image)
    cv2.imwrite(os.path.join(image_wsi_sampling.folder, "examples", "{}_{}_classes.png".format(tile, num)), classes_image)

    return ctime

if __name__ == "__main__":
    #generate_masks_from_subdata("dataset1_split1")
    #generate_masks_from_subdata("dataset1_split2")
    #generate_masks_from_subdata("dataset1_regional_split1")
    #generate_masks_from_subdata("dataset1_regional_split2")
    torch.multiprocessing.set_start_method("spawn")
    sampler = get_image_sampler("dataset1_regional_split1")

    tiles = ["5ac25a1e40dd", "39b8aafd630b", "8e90e6189c6b", "f45a29109ff5"]

    for tile in tiles:
        sampler.request_load_image([tile] * 10)

    all_time_elapsed = []
    for tile in tiles:
        print("Sampling from tile {}".format(tile))
        for i in tqdm.tqdm(range(10)):
            time_elapsed = generate_image_example(sampler, tile, i)
            time.sleep(0.4)
            all_time_elapsed.append(time_elapsed)

    all_time_elapsed = np.array(all_time_elapsed)
    print("Average time elapsed: {} seconds".format(np.mean(all_time_elapsed)))
    print("Median time elapsed: {} seconds".format(np.median(all_time_elapsed)))
    print("Min time elapsed: {} seconds".format(np.min(all_time_elapsed)))
    print("Max time elapsed: {} seconds".format(np.max(all_time_elapsed)))
    print("First time elapsed: {} seconds".format(all_time_elapsed[0]))
    print("Last time elapsed: {} seconds".format(all_time_elapsed[-1]))

    sampler.terminate()