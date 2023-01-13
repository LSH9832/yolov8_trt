import os

import cv2
import torch
import torch2trt as trt
import tensorrt
import numpy as np
from time import time
from loguru import logger

from .color import get_color
from .ultra_process import preprocess, postprocess


LOGGER = tensorrt.Logger(tensorrt.Logger.ERROR)
tensorrt.init_libnvinfer_plugins(LOGGER, "")


class TRTDetector:
    strides = [8, 16, 32]

    def __init__(self, weight_file, conf_thres, nms_thres, *args, **kwargs):
        os.environ["CUDA_MODULE_LOADING"] = "LAZY"
        import torch2trt
        if weight_file is not None:
            logger.info(f"loading weights from {weight_file}")
        self.model = torch2trt.TRTModule()
        self.model.eval()
        self.model.cuda()
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres
        if "ckpt" in kwargs.keys():
            ckpt = kwargs["ckpt"]
        else:
            ckpt = torch.load(weight_file, map_location="cpu")

        self.use_decoder = kwargs["use_decoder"] if "use_decoder" in kwargs else False

        # for k in ckpt:
        #     print(k)
        self.model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
        self.names = self.class_names = ckpt["names"] if "names" in ckpt else [str(i) for i in range(80)]
        self.input_size = ckpt["img_size"] if "img_size" in ckpt else kwargs["input_size"] if "input_size" in kwargs else [640, 640]
        self.pixel_range = ckpt["pixel_range"] if "pixel_range" in ckpt else 1

        if isinstance(self.input_size, int):
            self.input_size = [self.input_size] * 2
        # print(self.input_size)
        self.batch_size = ckpt["batch_size"] if "batch_size" in ckpt else 1

        x = torch.ones([self.batch_size, 3, *self.input_size]).cuda()
        logger.info(f"tensorRT input shape: {x.shape}")
        logger.info(f"tensorRT output shape: {self.model(x).shape[-3:]}")
        logger.info("tensorRT model loaded")

    def __preprocess(self, imgs):
        pad_ims = []
        rs = []
        for img in imgs:
            pad_im, r = preprocess(img, self.input_size)
            pad_ims.append(torch.from_numpy(pad_im).unsqueeze(0))
            rs.append(r)
        assert len(pad_ims) == self.batch_size, "batch size not match!"
        self.t0 = time()
        ret_ims = pad_ims[0] if len(pad_ims) == 1 else torch.cat(pad_ims)
        return ret_ims.float(), rs

    def __postprocess(self, results, rs):
        # print(results, results.shape)
        outs = postprocess(results, len(self.class_names), self.conf_thres, self.nms_thres, True, False)
        for i, r in enumerate(rs):
            if outs[i] is not None:
                outs[i][..., :4] /= r
                outs[i] = outs[i].cpu()
        return outs

    def decode_outputs(self, outputs):
        dtype = outputs.type()
        grids = []
        strides = []

        for stride in self.strides:
            hsize, wsize = self.input_size[0] / stride, self.input_size[1] / stride
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride, dtype=torch.long))

        grids = torch.cat(grids, dim=1).type(dtype)
        strides = torch.cat(strides, dim=1).type(dtype)

        outputs[..., :2] = (outputs[..., :2] + grids) * strides
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
        return outputs

    def __call__(self, imgs):
        if isinstance(imgs, np.ndarray):
            imgs = [imgs]

        with torch.no_grad():

            inputs, ratios = self.__preprocess(imgs)

            inputs = inputs.cuda()
            if self.pixel_range == 1:
                inputs /= 255
            # if self.fp16:
            #     inputs = inputs.half()

            net_outputs = self.model(inputs)
            # print(net_outputs.shape)
            if len(net_outputs.shape) == 4:
                net_outputs = net_outputs[0]
            _, na, la = net_outputs.shape
            if na < la:
                net_outputs = net_outputs.permute(0, 2, 1)
            if self.use_decoder:
                net_outputs = self.decode_outputs(net_outputs)
            outputs = self.__postprocess(net_outputs, ratios)
            self.dt = time() - self.t0

        return outputs

    @staticmethod
    def draw(imgs, results, class_names, draw_label=True, line_thickness=3):
        single = False
        if isinstance(imgs, np.ndarray):
            imgs = [imgs]
            single = True
        tf = max(line_thickness - 1, 1)
        for img, result in zip(imgs, results):
            # print(img.shape)
            if result is not None:
                # print(result.shape)
                flag = result.shape[-1] == 7
                for *xywh, conf, cls in result:
                    c1 = (int(xywh[0]), int(xywh[1]))
                    c2 = (int(xywh[2]), int(xywh[3]))
                    color = get_color(int(cls), True)
                    cv2.rectangle(img, c1, c2, color, line_thickness, cv2.LINE_AA)
                    if draw_label:
                        label = f'{class_names[int(cls)]} {conf * (xywh[-1] if flag else 1):.2f}'
                        t_size = cv2.getTextSize(label, 0, fontScale=line_thickness / 3, thickness=tf)[0]
                        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
                        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, line_thickness / 3, [225, 255, 255],
                                    thickness=tf, lineType=cv2.LINE_AA)

        return imgs[0] if single else imgs


class End2EndDetector:
    names = None
    model = trt.TRTModule()
    file = "engine2pt.pt"
    ckpt = {}
    batch = 1
    img_size = [640, 640]
    loaded = False
    pixel_range = 1

    def __init__(self, file: str = None):
        os.environ["CUDA_MODULE_LOADING"] = "LAZY"

        if file is not None:
            self.load(file)

    def load(self, file=None, ckpt=None):

        if file is not None:
            if file.endswith(".engine"):
                with open(file, "rb") as f:
                    file = file[:-6] + "pt"
                    torch.save(
                        {
                            "model": {
                                "engine": bytearray(f.read()),
                                "input_names": ["input_0"],
                                "output_names": ["num_detections", "nmsed_boxes", "nmsed_scores", "nmsed_classes"]
                            },
                            "names": [],
                            "img_size": [640, 640],
                            "batch_size": 1
                        },
                        file
                    )
            self.file = file
            self.ckpt = torch.load(file, map_location="cpu")
        else:
            assert ckpt is not None
            self.ckpt = ckpt

        self.model.eval()
        self.model.cuda()

        self.model.load_state_dict(self.ckpt["model"])

        self.names = self.ckpt["names"]
        self.batch = self.ckpt["batch_size"]
        self.img_size = self.ckpt["img_size"]
        self.pixel_range = self.ckpt["pixel_range"]
        self.loaded = True

    def set_names(self, names):
        self.names = names
        self.ckpt["names"] = names

    def set_batch_size(self, batch=1):
        self.batch = batch
        self.ckpt["batch_size"] = batch

    def set_input_names(self, input_names):
        if isinstance(input_names, str):
            input_names = [input_names]
        self.ckpt["model"]["input_names"] = input_names

    def set_output_names(self, output_names):
        if isinstance(output_names, str):
            output_names = [output_names]
        self.ckpt["model"]["output_names"] = output_names

    def set_image_size(self, image_size):
        if isinstance(image_size, int):
            image_size = [image_size] * 2

        if len(image_size) == 1:
            image_size *= 2
        self.img_size = image_size
        self.ckpt["img_size"] = image_size

    def reload(self):
        torch.save(self.ckpt, self.file)
        self.model.load_state_dict(self.ckpt["model"])

    def preprocess(self, img, swap=(2, 0, 1)):

        if len(img.shape) == 3:
            padded_img = np.ones((self.img_size[0], self.img_size[1], 3), dtype=np.uint8) * 114
        else:
            padded_img = np.ones(self.img_size, dtype=np.uint8) * 114

        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

        padded_img = padded_img.transpose(swap)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        return padded_img, r

    def __call__(self, img):
        assert self.loaded
        with torch.no_grad():
            pad_img, ratio = self.preprocess(img)
            pad_img = torch.from_numpy(pad_img).unsqueeze(0)
            pad_img = pad_img.cuda()
            if self.pixel_range == 1:
                pad_img /= 255
            no, bbox, conf, cls = self.model(pad_img)
            bbox /= ratio
        return [no.cpu(), bbox.cpu(), conf.cpu(), cls.cpu()]

    @staticmethod
    def draw(imgs, outputs, names, draw_label=True, line_thickness=2):
        single = False
        if isinstance(imgs, np.ndarray):
            imgs = [imgs]
            single = True

        tf = max(line_thickness - 1, 1)
        for img, num_object, boxes, conf, cls in zip(imgs, *outputs):
            # print(int(num_object))
            for i in range(num_object):

                c1, c2 = [int(boxes[i][0]), int(boxes[i][1])], [int(boxes[i][2]), int(boxes[i][3])]
                now_cls = int(cls[i])
                color = get_color(now_cls, True)

                cv2.rectangle(img, c1, c2, color, line_thickness, cv2.LINE_AA)
                if draw_label:
                    label = f'{names[now_cls]} {conf[i]:.2f}'
                    t_size = cv2.getTextSize(label, 0, fontScale=line_thickness / 3, thickness=tf)[0]

                    out_image = c1[1] - t_size[1] - 3 < 0

                    c2 = c1[0] + t_size[0], c1[1] + t_size[1] + 3 if out_image else c1[1] - t_size[1] - 3

                    cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
                    cv2.putText(img, label, (c1[0], c1[1] - 2 if not out_image else c2[1] - 2), 0, line_thickness / 3,
                                [225, 255, 255], thickness=tf,
                                lineType=cv2.LINE_AA)
        return imgs[0] if single else imgs


def TensorRTDetector(
        file_name: str,
        conf_thres=0.2,
        nms_thres=0.6,
) -> TRTDetector or End2EndDetector:
    assert file_name.endswith(".pt")
    logger.info(f"loading weights from {file_name}")
    ckpt = torch.load(file_name, map_location="cpu")
    if "end2end" in ckpt.keys() and ckpt["end2end"]:
        logger.info("load End2End detector")
        detector = End2EndDetector()
        detector.load(None, ckpt)
    else:
        logger.info("load normal detector")
        detector = TRTDetector(
            weight_file=None,
            conf_thres=conf_thres,
            nms_thres=nms_thres,
            ckpt=ckpt
        )
    return detector
