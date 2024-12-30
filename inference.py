import os
import torch
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

from typing import Iterable, List
from ppq import *
from ppq.api import *
# 如果您将 load_calibration_dataset 写在单独的文件，请在此 import
# from your_file import load_calibration_dataset



class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem
    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)
    def __repr__(self):
        return self.__str__()


def alloc_buf_N(engine, data: np.ndarray):
    """
    Allocates all host/device in/out buffers required for an engine.
    data: (batch, channel, height, width) in numpy format
    """
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    data_type = []

    for binding in engine:
        if engine.binding_is_input(binding):
            # input
            size = data.size  # batch*channel*height*width
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            data_type.append(dtype)

            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            # output
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            host_mem = cuda.pagelocked_empty(size, data_type[0])
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


def do_inference_v2(context, inputs, bindings, outputs, stream, data: np.ndarray):
    """
    Inputs and outputs are expected to be lists of HostDeviceMem objects.
    data: shape (batch, channel, height, width)
    """
    # 将 data 拷贝到 inputs[0].host
    np.copyto(inputs[0].host, data.ravel())  # flatten后拷贝

    # 将 inputs 拷贝到 GPU
    for inp in inputs:
        cuda.memcpy_htod_async(inp.device, inp.host, stream)

    # 设置动态维度（如果是固定维度，也可以省略这步）
    context.set_binding_shape(0, data.shape)

    # Run inference
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

    # Transfer predictions back from the GPU
    for out in outputs:
        cuda.memcpy_dtoh_async(out.host, out.device, stream)

    stream.synchronize()

    # Return only the host outputs
    return [out.host for out in outputs]


def load_engine(engine_path: str):
    print(f"\033[1;32mUsing Engine: {engine_path}\033[0m")
    TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
    with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


if __name__ == '__main__':
    # ---------------------------------------------------------------------
    # 1. 配置基本参数
    # ---------------------------------------------------------------------
    WORKING_DIRECTORY       = 'working'     # 您的工作目录，其中必须包含 data/ 用于存放图片数据
    INPUT_LAYOUT            = 'chw'         # 输入数据格式，可为 chw 或 hwc
    NETWORK_INPUTSHAPE      = [1, 3, 224, 224] 
    CALIBRATION_BATCHSIZE   = 1             # 此处设置批大小
    ENGINE_PATH             = "/home/wangsiyuan/ppq/working/Quantized.engine"

    # ---------------------------------------------------------------------
    # 2. 使用 load_calibration_dataset 加载数据
    #    这一步相当于一个 DataLoader，返回一个列表或可迭代对象 (batches)
    # ---------------------------------------------------------------------
    print("Loading calibration (inference) dataset ...")
    dataloader = load_calibration_dataset(
        directory    = WORKING_DIRECTORY,
        input_shape  = NETWORK_INPUTSHAPE,   # 对于 .bin/.raw 文件，此形状会用来 reshape
        batchsize    = CALIBRATION_BATCHSIZE,
        input_format = INPUT_LAYOUT
    )

    # ---------------------------------------------------------------------
    # 3. 加载 TensorRT Engine 并创建 ExecutionContext
    # ---------------------------------------------------------------------
    engine  = load_engine(ENGINE_PATH)
    context = engine.create_execution_context()

    # ---------------------------------------------------------------------
    # 4. 遍历每个批次，执行推理
    # ---------------------------------------------------------------------
    print("\nStart inference on all batches ...")
    for batch_idx, batch_tensor in enumerate(dataloader):
        # batch_tensor: (batch, channel, height, width) in PyTorch Tensor
        # 转为 NumPy 数组进行推理
        data_np = batch_tensor.cpu().numpy()   # shape: (N, C, H, W)

        # 您的 engine 通常要求 batch size <= engine.max_batch_size
        # 如果 dataloader 的 batch size > engine.max_batch_size，需要额外处理
        # 这里假设两者相等或者 max_batch_size >= batchsize

        # 分配缓存区
        inputs_alloc_buf, outputs_alloc_buf, bindings_alloc_buf, stream_alloc_buf = alloc_buf_N(engine, data_np)

        # 执行推理
        trt_feature = do_inference_v2(
            context, 
            inputs_alloc_buf, 
            bindings_alloc_buf, 
            outputs_alloc_buf,
            stream_alloc_buf, 
            data_np
        )

        # trt_feature 是一个 list，每个元素是对应输出张量 flatten 后的 numpy array
        # 如果引擎只有一个输出，可直接用 trt_feature[0]
        print(f"[Batch {batch_idx}] Output shapes:")
        for out_idx, out_arr in enumerate(trt_feature):
            print(f"  - Output {out_idx} shape: {out_arr.shape}")
        print()

    print("All inference done!")
