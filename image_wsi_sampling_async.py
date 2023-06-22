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

torch.multiprocessing.set_start_method("fork")

class MultipleImageSamplerAsync:

    def __init__(self, subdata_name:str, image_width: int):
        image_loading_pipe_recv, self.image_loading_pipe_send = torch.multiprocessing.Pipe(duplex=False)
        self.image_available_lock = torch.multiprocessing.Lock()
        self.image_required_flag = torch.multiprocessing.Value(ctypes.c_bool, True)
        self.image_access_lock = torch.multiprocessing.Lock()
        self.image_available_lock.acquire(block=True)


        self.shared_image_cat = torch.zeros((4, image_width, image_width), device="cpu", dtype=torch.float32)
        self.shared_ground_truth = torch.zeros((image_width, image_width), device="cpu", dtype=torch.float32)
        self.shared_ground_truth_mask = torch.zeros((image_width, image_width), device="cpu", dtype=torch.float32)

        self.shared_image_cat.share_memory_()
        self.shared_ground_truth.share_memory_()
        self.shared_ground_truth_mask.share_memory_()

        self.running = torch.multiprocessing.Value(ctypes.c_bool, True)

        self.process = torch.multiprocessing.Process(target=self.subprocess_run, args=[image_loading_pipe_recv, subdata_name, image_width])
        self.process.start()

    def terminate(self):
        self.running.value = False

    def request_load_image(self, wsi_ids):
        if type(wsi_ids) is str:
            self.image_loading_pipe_send.send(wsi_ids)
        elif type(wsi_ids) is list:
            for wsi_id in wsi_ids:
                self.image_loading_pipe_send.send(wsi_id)
        else:
            raise ValueError("Invalid type for wsi_ids, expected list or string.")

    def get_image(self, device):
        """Get the currently loaded image. WARNING - this is a blocking call.
        If no images are in the pipe, this might create a deadlock."""
        self.image_available_lock.acquire(block=True)

        self.image_access_lock.acquire(block=True)
        image = self.shared_image_cat.to(device, copy=True)
        ground_truth = self.shared_ground_truth.to(device, copy=True)
        ground_truth_mask = self.shared_ground_truth_mask.to(device, copy=True)
        self.image_required_flag.value = True
        self.image_access_lock.release()

        return image, ground_truth, ground_truth_mask


    def subprocess_run(self, image_loading_pipe_recv, subdata_name: str, image_width: int):
        print("Subprocess starting...")
        sampler = image_wsi_sampling.get_image_sampler(subdata_name, image_width, device="cpu", use_async={})

        buffer_image_cat = None
        buffer_ground_truth = None
        buffer_ground_truth_mask = None
        buffer_image_available = False
        pending_images = []
        print("Subprocess running...")

        run_time = 0
        while self.running.value:
            if image_loading_pipe_recv.poll():
                wsi_id = image_loading_pipe_recv.recv()
                pending_images.append(wsi_id)

            if (not buffer_image_available) and (len(pending_images) > 0):
                    wsi_id = pending_images.pop(0)
                    buffer_image_cat, buffer_ground_truth, buffer_ground_truth_mask = sampler.obtain_random_image_from_tile(wsi_id, device="cpu")
                    buffer_image_available = True

            if self.image_required_flag.value and buffer_image_available:
                self.image_access_lock.acquire(block=True)
                self.shared_image_cat.copy_(buffer_image_cat)
                self.shared_ground_truth.copy_(buffer_ground_truth)
                self.shared_ground_truth_mask.copy_(buffer_ground_truth_mask)
                self.image_required_flag.value = False
                self.image_access_lock.release()
                self.image_available_lock.release()
                buffer_image_available = False


            run_time += 1
            if run_time % 100 == 0:
                gc.collect()

        print("Subprocess terminating...")

def get_image_sampler(subdata_name: str, image_width=1024) -> MultipleImageSamplerAsync:
    return MultipleImageSamplerAsync(subdata_name, image_width)

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

    sampler = get_image_sampler("dataset1_regional_split1")

    tiles = ["5ac25a1e40dd", "39b8aafd630b", "8e90e6189c6b", "f45a29109ff5"]

    for tile in tiles:
        sampler.request_load_image([tile] * 10)

    all_time_elapsed = []
    for tile in tiles:
        print("Sampling from tile {}".format(tile))
        for i in tqdm.tqdm(range(10)):
            time_elapsed = generate_image_example(sampler, tile, i)
            all_time_elapsed.append(time_elapsed)

    all_time_elapsed = np.array(all_time_elapsed)
    print("Average time elapsed: {} seconds".format(np.mean(all_time_elapsed)))
    print("Median time elapsed: {} seconds".format(np.median(all_time_elapsed)))
    print("Min time elapsed: {} seconds".format(np.min(all_time_elapsed)))
    print("Max time elapsed: {} seconds".format(np.max(all_time_elapsed)))
    print("First time elapsed: {} seconds".format(all_time_elapsed[0]))
    print("Last time elapsed: {} seconds".format(all_time_elapsed[-1]))