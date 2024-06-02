import os
from VLMPipline.VLMM import VLMProcessWrapper
import cv2
from matplotlib.animation import FuncAnimation

import matplotlib.pyplot as plt

import multiprocessing
from multiprocessing import Process, Array, Value
from multiprocessing.managers import SharedMemoryManager
import ctypes
import numpy as np

import threading, time

os.environ["HOME"] = "/"
multiprocessing.set_start_method("spawn", force=True)


class NumpyArrayStruct(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("shape", ctypes.c_void_p),
        ("strides", ctypes.c_void_p),
    ]


def encode_str_list_to_bytes(str_list):
    return [s.encode("utf-8") for s in str_list]


def decode_bytes_to_str_list(bytes_str):
    return [s.decode("utf-8") for s in bytes_str]


def CPU_DUTY(x=123.43523):
    """use as many as possible cpu"""
    while True:
        x * x
        # time.sleep(0.0000001)


if __name__ == "__main__":
    # shutdown torch grad
    owlv2_model_path = os.path.expanduser(
        "~/models/google-owlv2-large-patch14-finetuned"
    )
    owlv2_model_path = os.path.expanduser("~/models/google-owlv2-base-patch16-ensemble")
    sam_model_path = os.path.expanduser("~/models/facebook-sam-vit-huge")
    # sam_model_path = os.path.expanduser("~/models/facebook-sam-vit-base")
    xmem_model_path = os.path.expanduser("~/models/XMem.pth")
    resnet_18_path = os.path.expanduser("~/models/resnet18.pth")
    resnet_50_path = os.path.expanduser("~/models/resnet50.pth")
    threading_pool = []

    # vlmpipeline = VLM(
    #     owlv2_model_path,
    #     sam_model_path,
    #     xmem_model_path,
    #     resnet_18_path,
    #     resnet_50_path,
    #     verbose=False,
    #     resize_to=[640, 640],
    #     verbose_frame_every=1,
    #     input_batch_size=1,
    # )

    video_path = "/shared/codes/VoxPoser/VID_20240427_135016.mp4"
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    # reshape cv2 frame to c h w
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB
    frame = frame.transpose(2, 0, 1)
    start_time_processing = cv2.getTickCount()
    array_frame = Array(ctypes.c_uint8, frame.flatten())

    vlmpipeline = VLMProcessWrapper(
        ["Wooden house"],
        frame.shape,
        owlv2_model_path,
        sam_model_path,
        xmem_model_path,
        resnet_18_path,
        resnet_50_path,
        verbose=False,
        resize_to=[640, 640],
        verbose_frame_every=1,
        input_batch_size=1,
    )
    vlmpipeline.start()

    for i in range(2):
        t = threading.Thread(target=CPU_DUTY)
        t.start()
        threading_pool.append(t)

    vlmpipeline.wait_for_ready()
    # vlmpipeline.process_first_frame(
    #     encode_str_list_to_bytes(["Wooden house"]), array_frame, frame.shape
    # )
    result = vlmpipeline.process_first_frame(frame)
    # result = vlmpipeline.process_first_frame(["Wooden house"], frame)
    end_time_processing = cv2.getTickCount()
    processing_time = (
        end_time_processing - start_time_processing
    ) / cv2.getTickFrequency()
    print(f"Processing time: {processing_time} seconds")
    # save mask to jpg
    # 使用 pyplot 来显示这个图像，使用 'hot' 颜色映射
    fig, (ax1, ax2) = plt.subplots(1, 2)
    img1 = ax1.imshow(result[0], cmap="hot")
    img2 = ax2.imshow(frame.transpose(1, 2, 0))
    ax1.set_title("Mask")
    ax2.set_title("Original Frame")
    times = []
    fig.savefig("mask.jpg")

    def update(frame):
        global times
        ret, frame = cap.read()
        if not ret:
            anim.event_source.stop()
            cap.release()
            # cv2.destroyAllWindows()
            return
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB
        frame = frame.transpose(2, 0, 1)
        start_time_processing = cv2.getTickCount()
        result = vlmpipeline.process_frame(frame)
        end_time_processing = cv2.getTickCount()
        processing_time = (
            end_time_processing - start_time_processing
        ) / cv2.getTickFrequency()
        times.append(processing_time)
        print(f"avg processing time: {sum(times) / len(times)} seconds")
        img1.set_array(result[0])
        img2.set_array(frame.transpose(1, 2, 0))
        return img1, img2

    anim = FuncAnimation(
        fig,
        update,
        frames=range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))),
        interval=33,
    )
    anim.save("animation_VID_20240427_135016.gif", writer="imagemagick")
