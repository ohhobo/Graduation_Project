import torch
import pynvml



def get_gpu_memory(handle):
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    free = meminfo.free/1024/1024/1000
    return free


if __name__ == "__main__":
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    print('初始显存：%.4f G'%get_gpu_memory(handle))
