from ppq import *                                       
from ppq.api import *
import os
from working_yolo.yolo_task import *
from ppq.utils.TensorRTUtil import build_engine
import time
import tensorrt as trt
import sys
sys.path.append('/home/wangsiyuan/ppq/ppq/samples/TensorRT')
import trt_infer
import onnx
       
# modify configuration below:
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
WORKING_DIRECTORY = '/home/wangsiyuan/ppq/working_yolo'                             # choose your working directory
TARGET_PLATFORM   = TargetPlatform.TRT_INT8          # choose your target platform
MODEL_TYPE        = NetworkFramework.ONNX                 # or NetworkFramework.CAFFE
INPUT_LAYOUT          = 'chw'                             # input data layout, chw or hwc
CALIBRATION_BATCHSIZE = 64                              # batchsize of calibration dataset
NETWORK_INPUTSHAPE    = [CALIBRATION_BATCHSIZE, 3, 640, 640]                  # input shape of your network
EXECUTING_DEVICE      = 'cuda'                            # 'cuda' or 'cpu'.
REQUIRE_ANALYSE       = False
TRAINING_YOUR_NETWORK = True                              # 是否需要 Finetuning 一下你的网络
need_accuracy = False
need_performance = True
need_quantized = False
need_compile_model = False
label_path = os.path.join(WORKING_DIRECTORY, 'val.txt')
graph = None
output_onnx = os.path.join(WORKING_DIRECTORY, 'yolo_fp32.onnx')
output_config = os.path.join(WORKING_DIRECTORY, 'yolo_fp32_cfg.json')
output_engine = os.path.join(WORKING_DIRECTORY, 'yolo_int8.engine')
QS = QuantizationSettingFactory.default_setting()

if TRAINING_YOUR_NETWORK: #还有多种微调方式
    QS.lsq_optimization = True                                      # 启动网络再训练过程，降低量化误差
    QS.lsq_optimization_setting.steps = 500                         # 再训练步数，影响训练时间，500 步大概几分钟
    QS.lsq_optimization_setting.collecting_device = 'cuda'          # 缓存数据放在那，cuda 就是放在gpu，如果显存超了你就换成 'cpu'
    QS.lsq_optimization_setting.block_size         = 4
    QS.lsq_optimization_setting.lr                 = 1e-5
    QS.lsq_optimization_setting.gamma              = 0
    QS.lsq_optimization_setting.is_scale_trainable = True

QS.dispatching_table.append(operation='OP NAME', platform=TargetPlatform.FP32) #不适用于量化节点 转成fp32
# 在量化之前，将 Resize 的动态输入绑定为静态尺寸
# QS.dispatching_table.append(operation='/model.10/Resize', platform=TargetPlatform.FP32)
# QS.dispatching_table.append(operation='/model.13/Resize', platform=TargetPlatform.FP32)

print('正准备量化你的网络，检查下列设置:')
print(f'WORKING DIRECTORY    : {WORKING_DIRECTORY}')
print(f'TARGET PLATFORM      : {TARGET_PLATFORM.name}')
print(f'NETWORK INPUTSHAPE   : {NETWORK_INPUTSHAPE}，注意最后生成engine最大batchsize取决于输入数据的batchsize')
print(f'CALIBRATION BATCHSIZE: {CALIBRATION_BATCHSIZE}')
if need_quantized:
    dataloader = load_calibration_dataset(
        directory    = WORKING_DIRECTORY,
        input_shape  = NETWORK_INPUTSHAPE,
        batchsize    = CALIBRATION_BATCHSIZE,
        input_format = INPUT_LAYOUT)

    quantized = quantize_onnx_model(
        setting=QS,                     # setting 对象用来控制标准量化逻辑
        onnx_import_file="/home/wangsiyuan/ppq/working_yolo/yolov5nu.onnx",
        calib_dataloader=dataloader,
        calib_steps=32,
        input_shape=NETWORK_INPUTSHAPE, # 如果你的网络只有一个输入，使用这个参数传参
        inputs=None,                    # 如果你的网络有多个输入，使用这个参数传参，就是 input_shape=None, inputs=[torch.zeros(1,3,640,640), torch.zeros(1,3,640,640)]
        collate_fn=lambda x: x.to(EXECUTING_DEVICE),  # collate_fn 跟 torch dataloader 的 collate fn 是一样的，用于数据预处理，
                                                        # 你当然也可以用 torch dataloader 的那个，然后设置这个为 None
        platform=TARGET_PLATFORM,
        device=EXECUTING_DEVICE,
        do_quantize=True)
    
    print('网络量化结束，正在生成目标文件:')
    export_ppq_graph(
        graph=quantized, platform=TARGET_PLATFORM,
        graph_save_to = output_onnx,
        config_save_to = output_config)
    #check_dynamic_batch_onnx(output_onnx)
    
if need_compile_model:
    print('网络量化结束，正在生成engine:')
    builder = trt_infer.EngineBuilder()
    builder.create_network(output_onnx)
    builder.create_engine(engine_path=output_engine, precision="int8")
    # check_dynamic_batch(output_engine)

if need_accuracy:
    print("计算量化后模型精确度")
    from ultralytics import YOLO

    #engine
    model = YOLO(output_engine)
    results = model.val(data="/home/wangsiyuan/ppq/working_yolo/coco8.yaml", imgsz=640, batch=CALIBRATION_BATCHSIZE)
    #base pt
    model = YOLO("/home/wangsiyuan/ppq/working_yolo/yolov5nu.pt")
    results = model.val(data="/home/wangsiyuan/ppq/working_yolo/coco8.yaml", imgsz=640, batch=CALIBRATION_BATCHSIZE)

if need_performance:
    from ppq.utils.TensorRTUtil import build_engine, Benchmark, Profiling
    Benchmark(output_engine)
    Profiling(output_engine)
