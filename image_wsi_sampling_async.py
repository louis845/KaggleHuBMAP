import ctypes
import time

import torch.multiprocessing

import image_wsi_sampling

torch.multiprocessing.set_start_method("fork")

class MultipleImageSamplerAsync:

    def __init__(self, sampler: image_wsi_sampling.MultipleImageSampler):
        self.sampler = sampler
        self.image_width = sampler.image_samplers[next(iter(sampler.image_samplers))].image_width

        image_loading_pipe_recv, self.image_loading_pipe_send = torch.multiprocessing.Pipe(duplex=False)
        self.shared_image_lock = torch.multiprocessing.Lock()

        self.shared_image_cat = torch.Tensor(4, self.image_width, self.image_width, device="cpu", dtype=torch.float32)
        self.shared_ground_truth = torch.Tensor(self.image_width, self.image_width, device="cpu", dtype=torch.float32)
        self.shared_ground_truth_mask = torch.Tensor(self.image_width, self.image_width, device="cpu", dtype=torch.float32)
        self.image_available = torch.multiprocessing.Value(ctypes.c_bool, False)

        self.shared_image_cat.share_memory_()
        self.shared_ground_truth.share_memory_()
        self.shared_ground_truth_mask.share_memory_()

        self.buffer_image_cat = None
        self.buffer_ground_truth = None
        self.buffer_ground_truth_mask = None
        self.buffer_image_available = torch.multiprocessing.Value(ctypes.c_bool, False)

        self.running = torch.multiprocessing.Value(ctypes.c_bool, True)

        self.process = torch.multiprocessing.Process(target=self.subprocess_run, args=[image_loading_pipe_recv])
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

    def get_image(self):
        """Get the currently loaded image. WARNING - this is a blocking call.
        If no images are in the pipe, this might create a deadlock."""
        self.shared_image_lock.acquire(block=True)


    def subprocess_run(self, image_loading_pipe_recv):
        while self.running.value:
            if