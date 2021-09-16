import argparse
import sys
import os

import numpy as np
import glob
from pathlib import Path

import tensorflow as tf
import tensorflow_datasets as tfds

IMG_FORMATS = [
    "bmp",
    "jpg",
    "jpeg",
    "png",
    "tif",
    "tiff",
    "dng",
    "webp",
    "mpo",
]  # acceptable image suffixes
VID_FORMATS = [
    "mov",
    "avi",
    "mp4",
    "mpg",
    "mpeg",
    "m4v",
    "wmv",
    "mkv",
]  # acceptable video suffixes

import cv2


def letterbox(
    im,
    new_shape=(640, 640),
    color=(114, 114, 114),
    auto=True,
    scaleFill=False,
    scaleup=True,
    stride=32,
):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(
        im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # add border
    return im, ratio, (dw, dh)


class LoadImages:  # for inference
    def __init__(self, path, img_size=640, stride=32, auto=True):
        p = str(Path(path).absolute())  # os-agnostic absolute path
        if "*" in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, "*.*")))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception(f"ERROR: {p} does not exist")

        images = [x for x in files if x.split(".")[-1].lower() in IMG_FORMATS]
        videos = [x for x in files if x.split(".")[-1].lower() in VID_FORMATS]
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.stride = stride
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = "image"
        self.auto = auto
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, (
            f"No images or videos found in {p}. "
            f"Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}"
        )

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = "video"
            ret_val, img0 = self.cap.read()
            if not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()

            self.frame += 1
            print(
                f"video {self.count + 1}/{self.nf} ({self.frame}/{self.frames}) {path}: ",
                end="",
            )

        else:
            # Read image
            self.count += 1
            img0 = cv2.imread(path)  # BGR
            assert img0 is not None, "Image Not Found " + path
            print(f"image {self.count}/{self.nf} {path}: ", end="")

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride, auto=self.auto)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return path, img, img0, self.cap

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf  # number of files


def quantize_model(
    height,
    width,
    pb_path,
    output_path,
    calib_num,
    tfds_root,
    download_flag,
    calib_dataset,
):
    raw_test_data = tfds.load(
        name="coco/2017",
        with_info=False,
        split="validation",
        data_dir=tfds_root,
        download=download_flag,
    )
    input_shapes = [(3, height, width)]
    dataset = LoadImages(calib_dataset, img_size=(height, width), auto=False)
    print(f"Samples in dataset :{len(dataset)}")
    # def representative_dataset_gen():
    #    for i, data in enumerate(raw_test_data.take(calib_num)):
    #        print("calibrating...", i)
    #        image = data["image"].numpy()
    #        images = []
    #        for shape in input_shapes:
    #            data = tf.image.resize(image, (shape[1], shape[2]))
    #            tmp_image = data / 255.0
    #            tmp_image = tmp_image[np.newaxis, :, :, :]
    #            images.append(tmp_image)
    #        yield images

    def representative_dataset_gen():
        # Representative dataset for use with converter.representative_dataset
        n = 0
        for path, img, im0s, vid_cap in dataset:
            # Get sample input data as a numpy array in a method of your choosing.
            n += 1
            input = np.transpose(img, [1, 2, 0])
            input = np.expand_dims(input, axis=0).astype(np.float32)
            input /= 255.0
            yield [input]
            if n >= 100:
                break

    input_arrays = ["inputs"]
    output_arrays = ["Identity", "Identity_1", "Identity_2"]
    converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
        pb_path, input_arrays, output_arrays
    )
    converter.experimental_new_quantizer = False
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.allow_custom_ops = False
    converter.inference_input_type = tf.uint8
    # To commonalize postprocess, output_type is float32
    converter.inference_output_type = tf.float32
    converter.representative_dataset = representative_dataset_gen
    tflite_model = converter.convert()
    with open(output_path, "wb") as w:
        w.write(tflite_model)
    print("Quantization Completed!", output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--calib-dataset", type=str, default="../calib_dataset")
    parser.add_argument(
        "--pb_path", default="/workspace/yolov5/tflite/model_float32.pb"
    )
    parser.add_argument(
        "--output_path", default="/workspace/yolov5/tflite/model_quantized.tflite"
    )
    parser.add_argument(
        "--calib_num", type=int, default=100, help="number of images for calibration."
    )
    parser.add_argument("--tfds_root", default="/workspace/TFDS/")
    parser.add_argument(
        "--download_tfds",
        action="store_true",
        help="download tfds. it takes a lot of time.",
    )
    args = parser.parse_args()
    quantize_model(
        args.height,
        args.width,
        args.pb_path,
        args.output_path,
        args.calib_num,
        args.tfds_root,
        args.download_tfds,
        args.calib_dataset,
    )
