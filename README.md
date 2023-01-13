# YOLOv8 TensorRT 快速部署工具

## 环境准备

- CUDA, CuDNN, TensorRT（Toolkit）
- 所有requirements.txt中的三方库

**注意** end2end模型要求TensorRT版本为8.0.0及以上

其中 **torch2trt** 安装如下
```shell
git clone https://github.com/NVIDIA-AI-IOT/torch2trt.git
cd torch2trt
python setup.py install
```
python 的 **tensorrt** 通过 TensorRT Toolkit中提供的安装包安装

## 使用

- 支持转onnx、tensorrt engine文件以及将engine文件与相关信息打包的，便于python中使用的pt文件
- 支持常规模型以及端到端（End2End）模型

### 下载模型
通过yolov8官方方式下载模型
```shell
python download_all_models.py
```

## 导出模型
```shell
python export.py --mode trt  # onnx：导出onnx模型， trt：导出onnx及tensorrt常规模型，end2end：导出onnx常规及end2end模型，并转化end2end的engine模型
                 --weights yolov8s.pt
                 --batch 1
                 --img-size 640 640
                 --opset 11
                 
                 # 以下为trt或end2end所需参数，如为onnx模式不用写
                 --workspace 10   # (GB)
                 --fp16           # int8, best
                 
                 # 以下为end2end所需参数(Batched_NMS中所需参数)，如为onnx或trt模式不用写
                 --conf-thres 0.2
                 --nms-thres 0.6
                 --topk 2000
                 --keep 100
```

不同模式导出不同模型，以yolov8s.pt为例

- onnx模式
```shell
./yolo_export/yolov8s/yolov8s.onnx   # 常规模型
./yolo_export/yolov8s/yolov8s.yaml   # 模型相关信息
./yolo_export/yolov8s/yolov8s.json   # 模型相关信息
```
- trt模式
```shell
./yolo_export/yolov8s/yolov8s.onnx   # 常规模型
./yolo_export/yolov8s/yolov8s.yaml   # 常规模型相关信息
./yolo_export/yolov8s/yolov8s.json   # 常规模型相关信息

./yolo_export/yolov8s/yolov8s.engine # 用于C++部署
./yolo_export/yolov8s/yolov8s.pt     # 用于python部署
```

- end2end模式
```shell
./yolo_export/yolov8s/yolov8s.onnx   # 常规模型
./yolo_export/yolov8s/yolov8s.yaml   # 常规模型相关信息
./yolo_export/yolov8s/yolov8s.json   # 常规模型相关信息

./yolo_export/yolov8s/yolov8s_end2end.onnx   # 端到端模型
./yolo_export/yolov8s/yolov8s_end2end.json   # 端到端模型相关信息
./yolo_export/yolov8s/yolov8s_end2end.engine # 用于C++部署
./yolo_export/yolov8s/yolov8s_end2end.pt     # 用于python部署
```

## 推理

### python
由本项目生成的，不论是常规模型还是端到端模型，均可使用
```shell
python trt_infer.py --weight ./yolo_export/yolov8s/yolov8s.pt
                    --source path/to/your/video/file or (rtsp/rtmp/http)://xxx.xxx.xxx.xxx/live/xxx or 0
                    --no-label   # 声明此项则不会显示类别与置信度标签，仅显示目标框
                    
                    # 以下为常规模型所需参数，端到端模型无效
                    --conf 0.2     # 置信度阈值
                    --nms 0.6      # NMS 目标框重合度阈值
```

### c++
即将推出，着急用可到其他项目中找找