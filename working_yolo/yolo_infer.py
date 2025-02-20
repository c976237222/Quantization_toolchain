import pycuda.driver as cuda
import pycuda.autoinit
import torch
import tensorrt as trt
from ultralytics import YOLO  # 确保你有 ultralytics 的 YOLO 库
import numpy as np

class YOLOTensorRT:
    def __init__(self, engine_path):
        """
        初始化 TensorRT YOLO 模型，并绑定 CUDA 上下文
        """
        self.engine_path = engine_path
        self.logger = trt.Logger(trt.Logger.ERROR)

        # ✅ 创建 CUDA 上下文，防止 "invalid resource handle"
        self.cfx = cuda.Device(0).make_context()

        # ✅ 加载 TensorRT 引擎
        with open(self.engine_path, 'rb') as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        # ✅ 创建执行上下文
        self.context = self.engine.create_execution_context()

    def infer(self, input_tensor):
        """
        执行推理，确保 CUDA 上下文管理正确
        """
        self.cfx.push()  # ✅ 绑定 CUDA 上下文
        
        try:
            # ✅ 确保输入数据是 numpy
            input_data = input_tensor.cpu().numpy().astype(np.float32)

            # ✅ 分配 TensorRT 缓冲区
            inputs, outputs, bindings, stream = self.allocate_buffers()

            # ✅ 复制输入数据到 TensorRT
            inputs[0].host = input_data

            # ✅ 执行推理
            trt_output = self.do_inference(bindings, inputs, outputs, stream)

            # ✅ 处理输出结果
            result_tensor = torch.tensor(trt_output[0], dtype=torch.float32, device="cuda")

        finally:
            self.cfx.pop()  # ✅ 释放 CUDA 上下文

        return result_tensor

    def allocate_buffers(self):
        """
        分配 TensorRT 所需的缓冲区
        """
        inputs, outputs, bindings, stream = [], [], [], cuda.Stream()

        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = np.zeros(size, dtype=dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):
                inputs.append({"host": host_mem, "device": device_mem})
            else:
                outputs.append({"host": host_mem, "device": device_mem})

        return inputs, outputs, bindings, stream

    def do_inference(self, bindings, inputs, outputs, stream):
        """
        运行 TensorRT 推理
        """
        [cuda.memcpy_htod_async(inp["device"], inp["host"], stream) for inp in inputs]
        self.context.execute_v2(bindings)
        [cuda.memcpy_dtoh_async(out["host"], out["device"], stream) for out in outputs]
        stream.synchronize()
        return [out["host"] for out in outputs]

    def __del__(self):
        """
        释放 TensorRT 资源
        """
        self.cfx.pop()
        del self.context
        del self.engine
        print("YOLO TensorRT 模型已释放")

