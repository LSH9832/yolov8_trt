import numpy as np
import cv2
import torch
import torchvision


def preprocess(img, input_size, swap=(2, 0, 1)):
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r


def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False, obj_conf_enabled=True):
    # print(prediction.shape)
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    # if not obj_conf_enabled:
    #     prediction[..., 4] = 1

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        if obj_conf_enabled:
            class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)
            conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
            detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float(), image_pred[:, 5 + num_classes:]), 1)
            detections = detections[conf_mask]
        else:
            class_conf, class_pred = torch.max(image_pred[:, 4: 4 + num_classes], 1, keepdim=True)
            conf_mask = (class_conf.squeeze() >= conf_thre).squeeze()
            detections = torch.cat((image_pred[:, :4], class_conf, class_pred.float(), image_pred[:, 4 + num_classes:]), 1)
            detections = detections[conf_mask]

        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)

        if not detections.size(0):
            continue

        if class_agnostic:
            nms_out_index = torchvision.ops.nms(
                detections[:, :4],
                detections[:, 4] * (detections[:, 5] if obj_conf_enabled else 1),
                nms_thre,
            )
        else:
            nms_out_index = torchvision.ops.batched_nms(
                detections[:, :4],
                detections[:, 4] * (detections[:, 5] if obj_conf_enabled else 1),
                detections[:, 6 if obj_conf_enabled else 5],
                nms_thre,
            )

        detections = detections[nms_out_index]
        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))

    return output
