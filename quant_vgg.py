from ppq import *                                       
from ppq.api import *
import os
from vgg_task6 import *
from vgg_task1 import *
from ppq.utils.TensorRTUtil import build_engine
import time
import tensorrt as trt
import sys
sys.path.append('/home/wangsiyuan/ppq/ppq/samples/TensorRT')
import trt_infer
import onnx
#For vgg16
def check_dynamic_batch_onnx(onnx_path):
    model = onnx.load(onnx_path)
    # 获取模型的输入和输出信息
    inputs = model.graph.input
    outputs = model.graph.output
    print(f"ONNX Model: {onnx_path}")
    # 检查输入张量
    print("Inputs:")
    for input_tensor in inputs:
        name = input_tensor.name
        shape = []
        is_dynamic = False
        for dim in input_tensor.type.tensor_type.shape.dim:
            if dim.dim_value > 0:
                shape.append(dim.dim_value)
            else:
                shape.append("dynamic")
                is_dynamic = True

        print(f"  Input '{name}': Shape {shape}, Dynamic: {is_dynamic}")

    # 检查输出张量
    print("Outputs:")
    for output_tensor in outputs:
        name = output_tensor.name
        shape = []
        is_dynamic = False
        for dim in output_tensor.type.tensor_type.shape.dim:
            if dim.dim_value > 0:
                shape.append(dim.dim_value)
            else:
                shape.append("dynamic")
                is_dynamic = True

        print(f"  Output '{name}': Shape {shape}, Dynamic: {is_dynamic}")

def check_dynamic_batch(engine_path):
    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, 'rb') as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    print(f"Engine has {engine.num_bindings} bindings.")

    for i in range(engine.num_bindings):
        binding_name = engine.get_binding_name(i)
        binding_shape = engine.get_binding_shape(i)
        is_input = engine.binding_is_input(i)
        is_dynamic = any(dim == -1 for dim in binding_shape)
        binding_type = "Input" if is_input else "Output"

        print(f"{binding_type} Binding '{binding_name}': Shape {binding_shape}, Dynamic: {is_dynamic}")

        if is_dynamic:
            print(f"  Note: '{binding_name}' supports dynamic dimensions.")
            
# modify configuration below:
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
WORKING_DIRECTORY = 'working'                             # choose your working directory
TARGET_PLATFORM   = TargetPlatform.TRT_INT8          # choose your target platform
MODEL_TYPE        = NetworkFramework.ONNX                 # or NetworkFramework.CAFFE
INPUT_LAYOUT          = 'chw'                             # input data layout, chw or hwc
CALIBRATION_BATCHSIZE = 64                                # batchsize of calibration dataset
NETWORK_INPUTSHAPE    = [CALIBRATION_BATCHSIZE, 3, 224, 224]                  # input shape of your network
EXECUTING_DEVICE      = 'cuda'                            # 'cuda' or 'cpu'.
REQUIRE_ANALYSE       = False
TRAINING_YOUR_NETWORK = True                              # 是否需要 Finetuning 一下你的网络
need_accuracy = False
need_performance = True
need_quantized = False
need_compile_model = False
label_path = os.path.join(WORKING_DIRECTORY, 'val.txt')
graph = None
output_onnx = os.path.join(WORKING_DIRECTORY, 'vgg_int8.onnx')
output_config = os.path.join(WORKING_DIRECTORY, 'vgg_int8_cfg.json')
output_engine = os.path.join(WORKING_DIRECTORY, 'vgg_int8.engine')
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
        onnx_import_file="working/vgg16.onnx",
        calib_dataloader=dataloader,
        calib_steps=32,
        input_shape=NETWORK_INPUTSHAPE, # 如果你的网络只有一个输入，使用这个参数传参
        inputs=None,                    # 如果你的网络有多个输入，使用这个参数传参，就是 input_shape=None, inputs=[torch.zeros(1,3,224,224), torch.zeros(1,3,224,224)]
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
    check_dynamic_batch_onnx(output_onnx)
if need_compile_model:
    print('网络量化结束，正在生成engine:')
    build_engine(onnx_file=output_onnx, 
                int8_scale_file=output_config, 
                engine_file=output_engine, int8=True, fp16 = True)
    #check_dynamic_batch(output_engine)

if need_accuracy:
    print("计算量化后模型精确度")
    result = evaluate_quantized_model(quantized, "working/data", label_path, 16) #评估graph
    result1 = evaluate_vgg16_model("working/data", label_path, 16) #base pt





CFG_VALID_RESULT = False
def infer_trt(model_path: str, samples: List[np.ndarray]) -> List[np.ndarray]:
    """ Run a tensorrt model with given samples
    """
    logger = trt.Logger(trt.Logger.ERROR)
    with open(model_path, 'rb') as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    results = []
    if CFG_VALID_RESULT:
        with engine.create_execution_context() as context:
            inputs, outputs, bindings, stream = trt_infer.allocate_buffers(context.engine)
            for sample in tqdm(samples, desc='TensorRT is running...'):
                inputs[0].host = convert_any_to_numpy(sample)
                [output] = trt_infer.do_inference(
                    context, bindings=bindings, inputs=inputs, 
                    outputs=outputs, stream=stream, batch_size=1)[0]
                results.append(convert_any_to_torch_tensor(output).reshape([-1, 1000]))
    else:
        with engine.create_execution_context() as context:
            inputs, outputs, bindings, stream = trt_infer.allocate_buffers(context.engine)
            inputs[0].host = convert_any_to_numpy(samples[0])
            for sample in tqdm(samples, desc='TensorRT is running...'):
                trt_infer.do_inference(
                    context, bindings=bindings, inputs=inputs, 
                    outputs=outputs, stream=stream, batch_size=1)
    return results

if need_performance:
    BATCHSIZE = 64
    #print("测试engine模型精确度")
    #accuracy_fp32 = evaluate_quantized_model_trt('working/vgg_fp32.engine', "working/data", label_path, BATCHSIZE)
    #accuracy_fp16 = evaluate_quantized_model_trt('working/vgg_fp16.engine', "working/data", label_path, BATCHSIZE)
    #accuracy_int8 = evaluate_quantized_model_trt('working/vgg_int8.engine', "working/data", label_path, BATCHSIZE)
    
    #print(f"TensorRT 模型评估准确率: FP32={accuracy_fp32:.2f}%, FP16={accuracy_fp16:.2f}%, INT8={accuracy_int8:.2f}%")

    
    print(f"测试engine模型推理速度,batchsize{BATCHSIZE}")

    from ppq.utils.TensorRTUtil import build_engine, Benchmark, Profiling
    Benchmark("working/vgg_fp32.engine", steps = 200)
    Benchmark("working/vgg_fp16.engine", steps = 200)
    Benchmark("working/vgg_int8.engine", steps = 200)
    #Profiling("int8.engine")



debug = False
if debug:
    print('网络量化结束，正在生成engine:')
    output_onnx1 = os.path.join(WORKING_DIRECTORY, 'vgg_int8_3.onnx')
    output_config1 = os.path.join(WORKING_DIRECTORY, 'vgg_int8_cfg_3.json')
    output_engine1 = os.path.join(WORKING_DIRECTORY, 'vgg_int8_4.engine')


    # from trt_infer import EngineBuilder
    # builder = EngineBuilder()
    # builder.create_network("/home/wangsiyuan/ppq/vgg16.onnx")
    # builder.create_engine(engine_path="a.engine", precision="fp32")

    #check_dynamic_batch(output_engine1)
    #check_dynamic_batch_onnx(output_onnx_path)
