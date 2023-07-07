import torch
import time
import os

class CudaMemoryLogger:
    def __init__(self, folder):
        # open file in folder/cuda_mem_log.txt
        self.file = open(os.path.join(folder, "cuda_mem_log.txt"), "w")

    def log(self, msg):
        self.file.write("========================================\n")
        self.file.write("Time: " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + "\n")
        self.file.write(msg + "\n")
        self.file.write(torch.cuda.memory_summary())
        self.file.flush()

    def close(self):
        self.file.flush()
        self.file.close()

    def __del__(self):
        self.close()

def obtain_memory_logger(folder):
    assert os.path.isdir(folder), "Folder must be an existing directory"
    return CudaMemoryLogger(folder)
