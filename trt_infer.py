import cv2
import argparse

from time import time
from loguru import logger

from utils import TensorRTDetector


def parser_args():
    parser = argparse.ArgumentParser("tensorRT detector parser")

    parser.add_argument("-w", "--weights", type=str, default="yolo_export/yolov8l/yolov8l.pt")
    parser.add_argument("-s", "--source", type=str, default="E:/videos/test.avi", help="video stream supported only")
    parser.add_argument("-c", "--conf", type=float, default=0.2, help="useless if it's an end2end model")
    parser.add_argument("-n", "--nms", type=float, default=0.5, help="useless if it's an end2end model")
    parser.add_argument("--no-label", action="store_true", help="do not draw label tag")

    return parser.parse_args()


if __name__ == '__main__':
    args = parser_args()

    detector = TensorRTDetector(args.weights, args.conf, args.nms)

    cam = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)

    if not cam.isOpened():
        logger.warning(f"failed to open video source {args.source}")

    count = 0
    infer_time = 0
    t_start = time()
    while cam.isOpened():
        t_ = time()
        success, img = cam.read()

        if not success:
            cv2.destroyAllWindows()
            break

        t0 = time()

        outputs = detector(img)

        t1 = time()

        detector.draw(img, outputs, detector.names, not args.no_label, 2)

        t2 = time()

        cv2.imshow("test", img)
        if cv2.waitKey(1) == 27:
            cv2.destroyAllWindows()
            break


        print(f"\rlatency: {int((t1 - t0) * 1000)}ms, "
              f"draw latency: {int((t2 - t1) * 1000)}ms, "
              f"total latency: {int((time() - t_) * 1000)}ms", end="          ")
        if count:
            infer_time += t1 - t0
        count += 1

    print()
    if count:
        logger.info(f"{count} frames in total, "
                    f"average infer latency: {infer_time * 1000 / (count - 1)}ms, "
                    f"average total latency: {(time() - t_start) * 1000 / count - 1}ms")
