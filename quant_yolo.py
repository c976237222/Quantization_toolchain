from ppq import *                                       
from ppq.api import *
import os
from ppq.utils.TensorRTUtil import build_engine, Benchmark, Profiling
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
CALIBRATION_BATCHSIZE = 1                                   # batchsize of calibration dataset
NETWORK_INPUTSHAPE    = [CALIBRATION_BATCHSIZE, 3, 640, 640]                  # input shape of your network
EXECUTING_DEVICE      = 'cuda'                            # 'cuda' or 'cpu'.
REQUIRE_ANALYSE       = False
TRAINING_YOUR_NETWORK = True                              # 是否需要 Finetuning 一下你的网络
name = "base"
base_model = False
need_quantized = False
need_compile_model = False
need_accuracy = False
need_performance = True

output_onnx = os.path.join(WORKING_DIRECTORY, f'yolo_{name}.onnx')
output_config = os.path.join(WORKING_DIRECTORY, f'yolo_{name}_cfg.json')
output_engine = os.path.join(WORKING_DIRECTORY, f'yolo_{name}.engine')
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
print(f'NETWORK INPUTSHAPE   : {NETWORK_INPUTSHAPE}, 注意最后生成engine最大batchsize取决于输入数据的batchsize')
print(f'CALIBRATION BATCHSIZE: {CALIBRATION_BATCHSIZE}')
if need_quantized:
    dataloader = load_calibration_dataset(
        directory    = WORKING_DIRECTORY,
        input_shape  = NETWORK_INPUTSHAPE,
        batchsize    = CALIBRATION_BATCHSIZE,
        input_format = INPUT_LAYOUT)

    quantized = quantize_onnx_model(
        setting=QS,                     # setting 对象用来控制标准量化逻辑
        onnx_import_file="/home/wangsiyuan/ppq/working_yolo/yolov5_s.onnx",
        calib_dataloader=dataloader,
        calib_steps=32,
        input_shape=NETWORK_INPUTSHAPE, # 如果你的网络只有一个输入，使用这个参数传参
        inputs=None,                    # 如果你的网络有多个输入，使用这个参数传参，就是 input_shape=None, inputs=[torch.zeros(1,3,640,640), torch.zeros(1,3,640,640)]
        collate_fn=lambda x: x.to(EXECUTING_DEVICE),  # collate_fn 跟 torch dataloader 的 collate fn 是一样的，用于数据预处理，
                                                        # 你当然也可以用 torch dataloader 的那个，然后设置这个为 None
        platform=TARGET_PLATFORM,
        device=EXECUTING_DEVICE,
        do_quantize=not base_model)
    print("SNR")
    snr_report = graphwise_error_analyse(
        graph=quantized, running_device=EXECUTING_DEVICE, 
        dataloader=dataloader, collate_fn=lambda x: x.to(EXECUTING_DEVICE))
    print('网络量化结束，正在生成目标文件:')
    export_ppq_graph(
        graph=quantized, platform=TARGET_PLATFORM,
        graph_save_to = output_onnx,
        config_save_to = output_config)
    #check_dynamic_batch_onnx(output_onnx)
    
if need_compile_model:
    print('网络量化结束，正在生成engine:')
    if not base_model:
        build_engine(onnx_file=output_onnx, 
                    int8_scale_file=output_config, 
                    engine_file=output_engine, int8=True, fp16 = True)
    else:
        build_engine(onnx_file=output_onnx, 
                engine_file=output_engine, int8=False, fp16 = False)

if need_accuracy:
    print("计算量化后模型精确度")
 #int8:
# Get ground truth result done.
#    Get map.
#    90.81% = aeroplane AP   ||      score_threhold=0.5 : F1=0.77 ; Recall=64.00% ; Precision=96.97%
#    88.68% = bicycle AP     ||      score_threhold=0.5 : F1=0.78 ; Recall=64.29% ; Precision=97.83%
#    78.46% = bird AP        ||      score_threhold=0.5 : F1=0.72 ; Recall=58.49% ; Precision=92.54%
#    82.55% = boat AP        ||      score_threhold=0.5 : F1=0.75 ; Recall=66.67% ; Precision=85.71%
#    74.40% = bottle AP      ||      score_threhold=0.5 : F1=0.61 ; Recall=46.91% ; Precision=86.36%
#    88.01% = bus AP         ||      score_threhold=0.5 : F1=0.79 ; Recall=75.00% ; Precision=83.33%
#    80.79% = car AP         ||      score_threhold=0.5 : F1=0.75 ; Recall=68.02% ; Precision=84.36%
#    89.74% = cat AP         ||      score_threhold=0.5 : F1=0.68 ; Recall=53.45% ; Precision=93.94%
#    69.56% = chair AP       ||      score_threhold=0.5 : F1=0.67 ; Recall=61.70% ; Precision=72.50%
#    89.04% = cow AP         ||      score_threhold=0.5 : F1=0.87 ; Recall=78.67% ; Precision=96.72%
#    64.03% = diningtable AP         ||      score_threhold=0.5 : F1=0.34 ; Recall=21.43% ; Precision=81.82%
#    81.82% = dog AP         ||      score_threhold=0.5 : F1=0.68 ; Recall=52.85% ; Precision=95.59%
#    88.67% = horse AP       ||      score_threhold=0.5 : F1=0.85 ; Recall=77.53% ; Precision=93.24%
#    86.60% = motorbike AP   ||      score_threhold=0.5 : F1=0.78 ; Recall=70.13% ; Precision=88.52%
#    87.51% = person AP      ||      score_threhold=0.5 : F1=0.82 ; Recall=78.68% ; Precision=84.82%
#    63.82% = pottedplant AP         ||      score_threhold=0.5 : F1=0.51 ; Recall=37.50% ; Precision=78.69%
#    79.24% = sheep AP       ||      score_threhold=0.5 : F1=0.75 ; Recall=67.19% ; Precision=84.31%
#    71.87% = sofa AP        ||      score_threhold=0.5 : F1=0.68 ; Recall=55.26% ; Precision=87.50%
#    77.82% = train AP       ||      score_threhold=0.5 : F1=0.74 ; Recall=64.81% ; Precision=85.37%
#    80.82% = tvmonitor AP   ||      score_threhold=0.5 : F1=0.75 ; Recall=63.38% ; Precision=91.84%
#    mAP = 80.71%
#    Get map done.

#base
#Get ground truth result done.
#Get map.
#93.26% = aeroplane AP   ||      score_threhold=0.5 : F1=0.85 ; Recall=76.00% ; Precision=97.44%
#88.26% = bicycle AP     ||      score_threhold=0.5 : F1=0.82 ; Recall=72.86% ; Precision=94.44%
#78.48% = bird AP        ||      score_threhold=0.5 : F1=0.72 ; Recall=58.49% ; Precision=92.54%
#81.86% = boat AP        ||      score_threhold=0.5 : F1=0.77 ; Recall=71.11% ; Precision=84.21%
#74.54% = bottle AP      ||      score_threhold=0.5 : F1=0.73 ; Recall=67.90% ; Precision=78.57%
#88.75% = bus AP         ||      score_threhold=0.5 : F1=0.79 ; Recall=75.00% ; Precision=83.33%
#81.36% = car AP         ||      score_threhold=0.5 : F1=0.76 ; Recall=70.27% ; Precision=83.87%
#88.58% = cat AP         ||      score_threhold=0.5 : F1=0.67 ; Recall=51.72% ; Precision=93.75%
#69.17% = chair AP       ||      score_threhold=0.5 : F1=0.68 ; Recall=60.99% ; Precision=76.79%
#88.87% = cow AP         ||      score_threhold=0.5 : F1=0.87 ; Recall=80.00% ; Precision=95.24%
#58.03% = diningtable AP         ||      score_threhold=0.5 : F1=0.27 ; Recall=16.67% ; Precision=77.78%
#82.79% = dog AP         ||      score_threhold=0.5 : F1=0.68 ; Recall=52.85% ; Precision=95.59%
#88.38% = horse AP       ||      score_threhold=0.5 : F1=0.84 ; Recall=75.28% ; Precision=94.37%
#87.96% = motorbike AP   ||      score_threhold=0.5 : F1=0.77 ; Recall=70.13% ; Precision=85.71%
#87.48% = person AP      ||      score_threhold=0.5 : F1=0.82 ; Recall=78.68% ; Precision=84.73%
#66.65% = pottedplant AP         ||      score_threhold=0.5 : F1=0.57 ; Recall=44.53% ; Precision=79.17%
#81.70% = sheep AP       ||      score_threhold=0.5 : F1=0.73 ; Recall=65.62% ; Precision=82.35%
#72.53% = sofa AP        ||      score_threhold=0.5 : F1=0.72 ; Recall=60.53% ; Precision=88.46%
#78.25% = train AP       ||      score_threhold=0.5 : F1=0.75 ; Recall=66.67% ; Precision=85.71%
#81.44% = tvmonitor AP   ||      score_threhold=0.5 : F1=0.77 ; Recall=67.61% ; Precision=90.57%
#mAP = 80.93%
#Get map done.
if need_performance:
    print("base")
    base_engine = f"{WORKING_DIRECTORY}/yolo_base.engine"
    Benchmark(base_engine, steps=1000) #2.4391 sec
    # Profiling(base_engine, steps=1000)
    print("int8")
    int8_engine = f"{WORKING_DIRECTORY}/yolo_int8.engine"
    Benchmark(int8_engine, steps=1000) #1.2440 sec
    # Profiling(int8_engine, steps=1000) 

