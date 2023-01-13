import torch
from ultralytics import YOLO


def parser_args():
    import argparse
    parser = argparse.ArgumentParser("YOLOv8 exporter parser")
    parser.add_argument("--mode", type=str, default="onnx", help="onnx, trt, end2end")
    parser.add_argument("-w", "--weights", type=str, default="yolov8l.pt", help="weight file name")
    parser.add_argument("-b", "--batch", type=int, default=1, help="input batch size")
    parser.add_argument("-s", "--simplify", action="store_true", help="simplify onnx")
    parser.add_argument("--img-size", type=int, default=[640, 640], nargs="+", help="input image size")
    parser.add_argument("--dynamic", action="store_true", help="use dynamic input & output")
    parser.add_argument("--input-names", type=str, default="input_0", nargs="+", help="onnx input names")
    parser.add_argument("--output-names", type=str, default="output_0", nargs="+", help="onnx output names")
    parser.add_argument("--opset", type=int, default=11, help="opset version")
    parser.add_argument("--dist", type=str, default="./yolo_export/")

    # the followings are for trt mode and end2end mode
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--workspace", type=float, default=8.0, help="export memory workspace(GB)")

    # choose one of them
    parser.add_argument("--fp16", action="store_true", help="fp16")
    parser.add_argument("--int8", action="store_true", help="int8")
    parser.add_argument("--best", action="store_true", help="best")

    # the followings are for end2end mode
    parser.add_argument("--conf-thres", type=float, default=0.2, help="confidence threshold of End2End model")
    parser.add_argument("--nms-thres", type=float, default=0.6, help="NMS threshold of End2End model")
    parser.add_argument("--topk", type=int, default=2000, help="let top k results to be NMS inputs")
    parser.add_argument("--keep", type=int, default=200, help="number of final results reserved")

    return parser.parse_args()


def export_onnx():

    args = parser_args()
    mode = args.mode

    workspace = args.workspace
    fp16 = args.fp16
    int8 = args.int8
    best = args.best
    device = args.device
    conf_thres = args.conf_thres
    nms_thres = args.nms_thres
    topk = args.topk
    keep = args.keep

    model = YOLO(args.weights)
    model.fuse()
    args = model.export(
        format="onnx",
        imgsz=([args.img_size] * 2) if isinstance(args.img_size, int) else args.img_size,
        dynamic=args.dynamic,
        batch=args.batch,
        opset=args.opset,
        simplify=args.simplify,
        dist=args.dist,
        input_names=args.input_names,
        output_names=args.output_names
    )

    names = [v for _, v in model.model.names.items()]
    try:
        import yaml
        yaml_file = args.f + ".yaml"
        yaml_data = {
            "batch_size": args.batch,
            "img_size": [*args.imgsz],
            "input_name": [*args.input_names],
            "output_name": [*args.output_names],
            "names": names
        }
        # print(yaml_data)
        yaml.dump(yaml_data, open(yaml_file, "w"), yaml.Dumper)
    except Exception as e:
        print(f"write yaml error: {e}")
        raise

    # not support classification and segmentation now.
    try:
        import json
        json_file = args.f + ".json"
        json_data = {
            "batch_size": args.batch,
            "img_size": [*args.imgsz],
            "input_name": args.input_names[0],
            "output_name": args.output_names[0],
            "pixel_range": 1,            # input image pixel value range: 0-1 or 0-255
            "obj_conf_enabled": False,   # YOLOv8 use class conf only
            "classes": names,
            "end2end": False
        }
        # for k, v in json_data.items():
        #     print(k, type(v))
        json.dump(json_data, open(json_file, "w", encoding="utf8"))
    except Exception as e:
        print(f"write json error: {e}")
        raise

    if mode == "trt":
        args.onnx = args.f + ".onnx"
        args.yaml = yaml_file
        args.workspace = workspace
        args.fp16 = fp16
        args.int8 = int8
        args.best = best
        args.device = device

        onnx2trt(args)
    elif mode == "end2end":
        args.onnx = args.f + ".onnx"
        args.conf_thres = conf_thres
        args.nms_thres = nms_thres
        args.topk = topk
        args.keep = keep
        args.workspace = workspace
        args.fp16 = fp16
        args.int8 = int8
        args.best = best
        args.device = device
        end2end(args, json_data)


def end2end(args, json_data):
    from utils.export_utils import ONNXEnd2EndConverter
    import os
    args.f = args.f.replace('\\', '/')
    args.onnx = args.onnx.replace('\\', '/')
    converter = ONNXEnd2EndConverter(args.onnx, len(json_data["classes"]))
    converter.end2end(
        dist_file=args.f + "_end2end.onnx",
        conf_thres=args.conf_thres,
        nms_thres=args.nms_thres,
        topk=args.topk,
        keep=args.keep
    )
    try:
        import json
        json_file = args.f + "_end2end.json"
        json_data["end2end"] = True
        json_data["output_name"] = ["num_detections", "nmsed_boxes", "nmsed_scores", "nmsed_classes"]
        json.dump(json_data, open(json_file, "w", encoding="utf8"))

    except Exception as e:
        print(f"write json error: {e}")
        raise
    from loguru import logger

    command = f"trtexec --onnx={args.f}_end2end.onnx --saveEngine={args.f}_end2end.engine --workspace={int(args.workspace * 1024)}" \
              f"{' --fp16' if args.fp16 else ' --int8' if args.int8 else ' --best' if args.best else ''}"

    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.device}"
    os.system(command)

    if not os.path.isfile(f"{args.f}.engine"):
        logger.error("Convert to engine file failed.")
        return
    else:
        logger.info(f"{args.f}_end2end.engine is saved for c++ inference")

    ckpt_data = {
        "model": {
            "engine": bytearray(open(f"{args.f}_end2end.engine", "rb").read()),
            "input_names": [json_data["input_name"]],
            "output_names": ["num_detections", "nmsed_boxes", "nmsed_scores", "nmsed_classes"]
        },
        "names": json_data["classes"],
        "img_size": json_data["img_size"],
        "batch_size": json_data["batch_size"],
        "pixel_range": 1,
        "end2end": True
    }
    torch.save(ckpt_data, args.f + "_end2end.pt")
    logger.info(f"{args.f}_end2end.pt is saved for python inference")


def onnx2trt(args=None):
    import yaml
    import argparse
    import os.path as osp
    import os
    from loguru import logger
    import torch

    def get_args():
        parser = argparse.ArgumentParser("yolov8 onnx2tensorrt parser")
        parser.add_argument("-o", "--onnx", type=str, default="yolov7.onnx", help="ONNX file")
        parser.add_argument("-y", "--yaml", type=str, default="yolov7.yaml", help="export params file")
        parser.add_argument("-w", "--workspace", type=int, default=8, help="export memory workspace(GB)")
        parser.add_argument("--fp16", action="store_true", help="fp16")
        parser.add_argument("--int8", action="store_true", help="int8")
        parser.add_argument("--best", action="store_true", help="best")
        parser.add_argument("-d", "--dist", type=str, default="./yolo_export/")
        parser.add_argument("--batch", type=int, default=0, help="batch-size")
        parser.add_argument("--device", type=int, default=0)
        return parser.parse_args()

    args = get_args() if args is None else args

    assert osp.isfile(args.onnx), f"No such file named {args.onnx}."
    assert osp.isfile(args.yaml), f"No such file named {args.yaml}."

    args.dist = osp.join(args.dist, osp.basename(osp.dirname(args.onnx)))

    os.makedirs(args.dist, exist_ok=True)

    name = args.onnx.replace("\\", "/").split("/")[-1][:-len(args.onnx.split(".")[-1])]

    engine_file = osp.join(args.dist, name + "engine").replace("\\", "/")
    pt_file = osp.join(args.dist, name + "pt").replace("\\", "/")
    cls_file = osp.join(args.dist, name + "txt").replace("\\", "/")
    params = yaml.load(open(args.yaml).read(), yaml.Loader)

    # Tensorrt 7.x.x
    workspace = int(args.workspace * 1024)
    command = f"trtexec --onnx={args.onnx}" \
              f"{' --fp16' if args.fp16 else ' --int8' if args.int8 else ' --best' if args.best else ''} " \
              f"--saveEngine={engine_file} --workspace={workspace} " \
              f"--batch={args.batch if not args.batch > 0 else params['batch_size'] if 'batch_size' in params else 1}"
    try:
        import tensorrt as trt
        if int(trt.__version__.split(".")[0]) > 7:
            # Tensorrt 8.x.x
            command = f"trtexec --onnx={args.onnx}" \
                      f"{' --fp16' if args.fp16 else ' --int8' if args.int8 else ' --best' if args.best else ''} " \
                      f"--saveEngine={engine_file} --workspace={workspace} " \
                      f"--explicitBatch"
    except:
        pass

    logger.info("start converting onnx to tensorRT engine file.")
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.device}"
    os.system(command)

    if not osp.isfile(engine_file):
        logger.error("tensorRT engine file convertion failed.")
        return

    logger.info(f"tensorRT engine saved to {engine_file}")

    try:
        data = {
            "model": {
                "engine": bytearray(open(engine_file, "rb").read()),
                "input_names": params["input_name"],
                "output_names": params["output_name"]
            },
            "names": params["names"],
            "img_size": params["img_size"],
            "batch_size": params["batch_size"],
            "end2end": False
        }
        class_str = ""
        for name in params["names"]:
            class_str += name + "\n"
        with open(cls_file, "w") as cls_f:
            cls_f.write(class_str[:-1])
            logger.info(f"class names txt pt saved to {cls_file}")
        torch.save(data, pt_file)
        logger.info(f"tensorRT pt saved to {pt_file}")
    except Exception as e:
        logger.error(f"convert2pt error: {e}")


if __name__ == '__main__':
    export_onnx()
