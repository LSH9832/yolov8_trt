import onnx_graphsurgeon as gs
import numpy as np
import onnx
from loguru import logger


class ONNXEnd2EndConverter:

    input_shape = [1, 3, 640, 640]
    output_shape = [1, 84, 8400]
    stride = 32
    num_class = 80
    ori_output = None
    ori_output_name = None
    batch = 1

    def __init__(self,
                 model: str or onnx.ModelProto,
                 num_class=80,
                 stride=32):
        if isinstance(model, str):
            model = onnx.load(model)

        self.model: onnx.ModelProto = model
        self.__get_stride_and_num_class(stride, num_class)
        self.graph = gs.import_onnx(self.model)

        try:
            self.__get_origin_output()
            self.__get_io_shape()
            is_standard = True
        except:
            is_standard = False
            logger.info("not a origin onnx without post_process")
        if is_standard:
            assert len(self.model.graph.input) == len(self.model.graph.output) == 1
        self.info()

    def info(self):
        logger.info(f"origin input size: {self.input_shape}")
        logger.info(f"origin output size: {self.output_shape}")

    def add_postprocess(self, dist_file=None):
        batch, num_array, len_array = self.output_shape
        transpose = len_array > num_array
        if transpose:
            len_array, num_array = num_array, len_array
        use_obj_conf = self.num_class + 5 == len_array

        # print(len_array, self.num_class)
        assert use_obj_conf or self.num_class + 4 == len_array

        extent_layers = []

        output_t = output_t_node = None
        if transpose:
            logger.info(f"add transpose layer: {self.output_shape} ---→ {[batch, num_array, len_array]}")
            output_t = gs.Variable(name="output_t", shape=(batch, num_array, len_array), dtype=np.float32)
            output_t_node = gs.Node(op="Transpose",
                                    inputs=[self.ori_output.outputs[0]],
                                    outputs=[output_t],
                                    attrs={"perm": [0, 2, 1]})
            extent_layers.append(output_t_node)
            self.ori_output = output_t_node
            self.ori_output_name = "output_t"

        def add_without_obj_conf():    # yolov6, yolov8
            logger.info("add layers without obj conf")
            starts_wh = gs.Constant("starts_wh", values=np.array([0, 0, 0], dtype=np.int64))
            ends_wh = gs.Constant("ends_wh", values=np.array([batch, num_array, 4], dtype=np.int64))

            starts_conf = gs.Constant("starts_conf", values=np.array([0, 0, 4], dtype=np.int64))
            ends_conf = gs.Constant("ends_conf", values=np.array([batch, num_array, len_array], dtype=np.int64))

            box_xywh_0 = gs.Variable(name="box_xywh_0", shape=(batch, num_array, 4), dtype=np.float32)
            label_conf_0 = gs.Variable(name='label_conf_0', shape=(batch, num_array, self.num_class), dtype=np.float32)

            box_xywh_node = gs.Node(op="Slice",
                                    inputs=[self.ori_output.outputs[0], starts_wh, ends_wh],
                                    outputs=[box_xywh_0])
            box_conf_node = gs.Node(op="Slice",
                                    inputs=[self.ori_output.outputs[0], starts_conf, ends_conf],
                                    outputs=[label_conf_0])

            # identity
            box_xywh = gs.Variable(name="box_xywh", shape=(batch, num_array, 4), dtype=np.float32)
            label_conf = gs.Variable(name='label_conf', shape=(batch, num_array, self.num_class), dtype=np.float32)

            identity_node_wh = gs.Node(op="Identity", inputs=[box_xywh_0], outputs=[box_xywh])
            identity_node_conf = gs.Node(op="Identity", inputs=[label_conf_0], outputs=[label_conf])

            # input
            starts_1 = gs.Constant("starts_x", values=np.array([0, 0, 0], dtype=np.int64))   # x
            ends_1 = gs.Constant("ends_x", values=np.array([batch, num_array, 1], dtype=np.int64))

            starts_2 = gs.Constant("starts_y", values=np.array([0, 0, 1], dtype=np.int64))   # y
            ends_2 = gs.Constant("ends_y", values=np.array([batch, num_array, 2], dtype=np.int64))

            starts_3 = gs.Constant("starts_w", values=np.array([0, 0, 2], dtype=np.int64))   # w
            ends_3 = gs.Constant("ends_w", values=np.array([batch, num_array, 3], dtype=np.int64))

            starts_4 = gs.Constant("starts_h", values=np.array([0, 0, 3], dtype=np.int64))   # h
            ends_4 = gs.Constant("ends_h", values=np.array([batch, num_array, 4], dtype=np.int64))

            # output
            x = gs.Variable(name="x_center", shape=(batch, num_array, 1), dtype=np.float32)
            y = gs.Variable(name="y_center", shape=(batch, num_array, 1), dtype=np.float32)
            w = gs.Variable(name="w", shape=(batch, num_array, 1), dtype=np.float32)
            h = gs.Variable(name="h", shape=(batch, num_array, 1), dtype=np.float32)

            # xywh_split_node = gs.Node(op="Split",inputs=[box_xywh],outputs= [x,y,w,h] )
            x_node = gs.Node(op="Slice", inputs=[box_xywh, starts_1, ends_1], outputs=[x])
            y_node = gs.Node(op="Slice", inputs=[box_xywh, starts_2, ends_2], outputs=[y])
            w_node = gs.Node(op="Slice", inputs=[box_xywh, starts_3, ends_3], outputs=[w])
            h_node = gs.Node(op="Slice", inputs=[box_xywh, starts_4, ends_4], outputs=[h])

            # 变换1
            # input
            div_val = gs.Constant("div_val", values=np.array([2], dtype=np.float32))
            div_val_ = gs.Constant("div_val_", values=np.array([-2], dtype=np.float32))
            # output
            w_ = gs.Variable(name="w_half_", shape=(batch, num_array, 1), dtype=np.float32)
            wplus = gs.Variable(name="w_half_plus", shape=(batch, num_array, 1), dtype=np.float32)
            h_ = gs.Variable(name="h_half_", shape=(batch, num_array, 1), dtype=np.float32)
            hplus = gs.Variable(name="h_half_plus", shape=(batch, num_array, 1), dtype=np.float32)

            w_node_ = gs.Node(op="Div", inputs=[w, div_val_], outputs=[w_])
            w_node_plus = gs.Node(op="Div", inputs=[w, div_val], outputs=[wplus])
            h_node_ = gs.Node(op="Div", inputs=[h, div_val_], outputs=[h_])
            h_node_plus = gs.Node(op="Div", inputs=[h, div_val], outputs=[hplus])

            # 变换2
            # output
            x1 = gs.Variable(name="x1", shape=(batch, num_array, 1), dtype=np.float32)
            y1 = gs.Variable(name="y1", shape=(batch, num_array, 1), dtype=np.float32)
            x2 = gs.Variable(name="x2", shape=(batch, num_array, 1), dtype=np.float32)
            y2 = gs.Variable(name="y2", shape=(batch, num_array, 1), dtype=np.float32)

            x1_node = gs.Node(op="Add", inputs=[x, w_], outputs=[x1])
            x2_node = gs.Node(op="Add", inputs=[x, wplus], outputs=[x2])
            y1_node = gs.Node(op="Add", inputs=[y, h_], outputs=[y1])
            y2_node = gs.Node(op="Add", inputs=[y, hplus], outputs=[y2])

            # concat
            # output

            boxes_0 = gs.Variable(name="boxes_0", shape=(batch, num_array, 4), dtype=np.float32)
            boxes_node_0 = gs.Node(op="Concat", inputs=[x1, y1, x2, y2], outputs=[boxes_0], attrs={"axis": 2})

            shapes = gs.Constant("shape", values=np.array([batch, num_array, 1, 4], dtype=np.int64))

            # output
            boxes = gs.Variable(name="boxes", shape=(batch, num_array, 1, 4), dtype=np.float32)
            boxes_node = gs.Node(op="Reshape", inputs=[boxes_0, shapes], outputs=[boxes])

            extent_layers.extend([box_xywh_node, box_conf_node, identity_node_wh, identity_node_conf,
                                  x_node, y_node, w_node, h_node, w_node_, w_node_plus, h_node_, h_node_plus,
                                  x1_node, x2_node, y1_node, y2_node, boxes_node_0, boxes_node])

            self.graph.outputs = [boxes, label_conf]

        def add_with_obj_conf():
            logger.info("not support obj conf models now!")
            pass

        add_with_obj_conf() if use_obj_conf else add_without_obj_conf()


        self.graph.nodes.extend(extent_layers)
        self.graph.cleanup().toposort()

        if dist_file is not None:
            onnx.save(gs.export_onnx(self.graph), dist_file)
            logger.info(f"onnx file saved to {dist_file}")
        return num_array

    def add_nms(self,
                dist_file=None,
                conf_thres=0.2,
                nms_thres=0.6,
                topk=1000,
                keep=200):

        attrs = {
            "shareLocation": 1,
            "backgroundLabelId": -1,
            "numClasses": self.num_class,
            "topK": topk,
            "keepTopK": keep,
            "scoreThreshold": conf_thres,
            "iouThreshold": nms_thres,
            "isNormalized": 0,
            "clipBoxes": 0,
            "scoreBits": 16,
            "plugin_version": "1"
        }

        logger.info("start add nms layers")
        tensors = self.graph.tensors()
        boxes_tensor = tensors["boxes"]
        confs_tensor = tensors["label_conf"]

        num_detections = gs.Variable(name="num_detections").to_variable(dtype=np.int32, shape=[self.batch, 1])
        nmsed_boxes = gs.Variable(name="nmsed_boxes").to_variable(dtype=np.float32, shape=[self.batch, keep, 4])
        nmsed_scores = gs.Variable(name="nmsed_scores").to_variable(dtype=np.float32, shape=[self.batch, keep])
        nmsed_classes = gs.Variable(name="nmsed_classes").to_variable(dtype=np.float32, shape=[self.batch, keep])

        new_outputs = [num_detections, nmsed_boxes, nmsed_scores, nmsed_classes]

        mns_node = gs.Node(
            op="BatchedNMS_TRT",
            attrs=attrs,
            inputs=[boxes_tensor, confs_tensor],
            outputs=new_outputs)

        self.graph.nodes.append(mns_node)
        self.graph.outputs = new_outputs
        self.graph.cleanup().toposort()

        if dist_file is not None:
            onnx.save(gs.export_onnx(self.graph), dist_file)
            logger.info(f"onnx file saved to {dist_file}")

    def end2end(self,
                dist_file=None,
                conf_thres=0.2,
                nms_thres=0.6,
                topk=1000,
                keep=200):

        num_array = self.add_postprocess(None)
        topk = min(topk, num_array)
        keep = min(keep, topk)
        self.add_nms(
            dist_file=dist_file,
            conf_thres=conf_thres,
            nms_thres=nms_thres,
            topk=topk,
            keep=keep,
        )


    def __get_io_shape(self):
        assert len(self.graph.inputs) == len(self.graph.outputs) == 1
        self.input_shape = self.graph.inputs[0].shape
        self.output_shape = self.graph.outputs[0].shape
        self.batch = self.output_shape[0]
        assert self.batch == self.input_shape[0]

    def __get_stride_and_num_class(self, stride=32, num_class=80):
        self.stride = stride
        self.num_class = num_class
        for prop in self.model.metadata_props:
            if prop.key == "stride":
                self.stride = prop.value
            elif prop.key == "names":
                import yaml
                names: dict = yaml.load(prop.value, yaml.SafeLoader)
                self.num_class = len(names.keys())

    def __get_origin_output(self):
        nodes = {}
        for node in self.graph.nodes:
            name, idx = node.name.split("_")
            nodes[int(idx)] = [name, node]
            # print(idx, name)

        node_type, self.ori_output = nodes[max(nodes.keys())]
        assert node_type == "Concat", "do not support this onnx structure!"
        self.ori_output_name = f"Concat_{max(nodes.keys())}"
        # print(self.ori_output.outputs[0])
