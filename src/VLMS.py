from typing import List
import traceback  # 添加在文件开头

from flask import Flask, request, jsonify
import numpy as np
import ctypes
import time
import torch
import base64

torch.set_grad_enabled(False)

from VLMPipline.VLM import VLM
from VLMPipline.utils import get_device, log_info, bcolors
from utils import timer_decorator

app = Flask(__name__)


class VLMHTTPServer:
    def __init__(
        self,
        labels,
        frame_shape,
        owlv2_model_path,
        sam_model_path,
        xmem_model_path,
        resnet_18_path=None,
        resnet_50_path=None,
        device=get_device(),
        resize_to=(480, 480),
        category_multiplier=100,
        verbose=None,
        verbose_frame_every=10,
        verbose_to_disk: bool = False,
        log_dir: str = None,
        input_batch_size=2,
    ):
        self.labels = labels  # not shared
        self.owlv2_model_path = owlv2_model_path
        self.sam_model_path = sam_model_path
        self.xmem_model_path = xmem_model_path
        self.resnet_18_path = resnet_18_path
        self.resnet_50_path = resnet_50_path
        self.device = device
        self.verbose = verbose
        self.resize_to = resize_to
        self.category_multiplier = category_multiplier
        self.verbose_frame_every = verbose_frame_every
        self.input_batch_size = input_batch_size
        self.verbose_to_disk = verbose_to_disk
        self.log_dir = log_dir

        self.is_processed_first_frame = False
        self.mask_shape = (self.input_batch_size,) + frame_shape[-2:]
        self.init_vlm()

    def init_vlm(self):
        print(
            bcolors.OKCYAN
            + f"[{self.get_current_process_id()}]: VLM init start"
            + bcolors.ENDC
        )
        with torch.no_grad():
            self.vlm = VLM(
                self.owlv2_model_path,
                self.sam_model_path,
                self.xmem_model_path,
                self.resnet_18_path,
                self.resnet_50_path,
                device=self.device,
                resize_to=self.resize_to,
                category_multiplier=self.category_multiplier,
                verbose=self.verbose,
                verbose_frame_every=self.verbose_frame_every,
                verbose_to_disk=self.verbose_to_disk,
                log_dir=self.log_dir,
                input_batch_size=self.input_batch_size,
            )
        print(
            bcolors.OKCYAN
            + f"[{self.get_current_process_id()}]: VLM init finished"
            + bcolors.ENDC
        )

    def get_current_process_id(self):
        return f"{time.time():.3f}"

    @timer_decorator
    def process_sole_frame(self, labels: List[str], frame: np.ndarray):
        with torch.no_grad():
            print(
                bcolors.OKCYAN
                + f"[{self.get_current_process_id()}]: set frame to VLM process"
                + bcolors.ENDC
            ) if self.verbose else None
            self.set_frame(frame)
            print(
                bcolors.OKCYAN
                + f"[{self.get_current_process_id()}]: start VLM process sole frame"
                + bcolors.ENDC
            ) if self.verbose else None
            masks = self.vlm.process_sole_frame(labels, frame, owlv2_threshold=0.075)
            ret = np.stack(masks)
            print(
                bcolors.OKCYAN
                + f"[{self.get_current_process_id()}]: VLM process finished"
                + bcolors.ENDC
            ) if self.verbose else None
            return ret

    @timer_decorator
    def process_first_frame(self, frame: np.ndarray):
        print(
            bcolors.OKCYAN
            + f"[{self.get_current_process_id()}]: set frame to VLM process"
            + bcolors.ENDC
        ) if self.verbose else None
        self.set_frame(frame)
        print(
            bcolors.OKCYAN
            + f"[{self.get_current_process_id()}]: start VLM process first frame"
            + bcolors.ENDC
        ) if self.verbose else None
        with torch.no_grad():
            masks = self.vlm.process_first_frame(
                self.labels, frame, owlv2_threshold=0.02
            )
        self.result = np.stack(masks).flatten()
        print(
            bcolors.OKCYAN
            + f"[{self.get_current_process_id()}]: VLM process finished"
            + bcolors.ENDC
        ) if self.verbose else None
        self.is_processed_first_frame = True
        ret = self.result.reshape(self.mask_shape)
        print(
            bcolors.OKCYAN
            + f"[{self.get_current_process_id()}]: post process frame finished"
            + bcolors.ENDC
        ) if self.verbose else None
        return ret

    @timer_decorator
    def process_frame(self, frame: np.ndarray):
        print(
            bcolors.OKCYAN
            + f"[{self.get_current_process_id()}]: set frame to VLM process"
            + bcolors.ENDC
        ) if self.verbose else None
        self.set_frame(frame)
        print(
            bcolors.OKCYAN
            + f"[{self.get_current_process_id()}]: start VLM process frame"
            + bcolors.ENDC
        ) if self.verbose else None
        masks = self.vlm.process_frame(frame, release_video_memory=False)
        self.result = np.stack(masks).flatten()
        print(
            bcolors.OKCYAN
            + f"[{self.get_current_process_id()}]: VLM process finished"
            + bcolors.ENDC
        ) if self.verbose else None
        ret = self.result.reshape(self.mask_shape)
        print(
            bcolors.OKCYAN
            + f"[{self.get_current_process_id()}]: post process frame finished"
            + bcolors.ENDC
        ) if self.verbose else None
        return ret

    def set_frame(self, frame: np.ndarray):
        self.frame = frame.copy()


vlm_server = VLMHTTPServer(
    labels=["label1", "label2"],  # Example labels
    frame_shape=(3, 480, 480),
    owlv2_model_path="/models/google-owlv2-large-patch14-finetuned",
    sam_model_path="/models/facebook-sam-vit-huge",
    xmem_model_path="/models/XMem.pth",
    resnet_18_path="/models/resnet18.pth",
    resnet_50_path="/models/resnet50.pth",
    device="cuda:3",
    verbose=True,
    verbose_to_disk=True,
    log_dir="/shared/codes/VoxPoser.worktrees/VoxPoser/logs/isaac/vlms",
    verbose_frame_every=1,
)


@app.route("/process_first_frame", methods=["POST"])
def process_first_frame():
    try:
        data = request.json
        labels = data["label"]
        data_base64 = data["data"]
        shape = tuple(data["shape"])

        # 解码 base64 数据
        data_bytes = base64.b64decode(data_base64)

        # 将字节数据转换为 NumPy 数组
        frame = np.frombuffer(data_bytes, dtype=np.uint8).reshape(shape)
        # 确保数组是可写的
        if not frame.flags.writeable:
            frame = np.copy(frame)
        # 处理帧
        vlm_server.labels = labels
        result = vlm_server.process_first_frame(frame)

        return jsonify(result.tolist()), 200
    except Exception as e:
        error_msg = f"Error: {str(e)}\nStack trace:\n{traceback.format_exc()}"
        print(error_msg)  # 在服务器端打印完整错误信息

        # 在开发环境中可以返回详细错误信息，生产环境建议只返回基本错误
        if app.debug:
            return jsonify(
                {"error": str(e), "stack_trace": traceback.format_exc()}
            ), 400
        return jsonify({"error": "Internal server error"}), 500


@app.route("/process_frame", methods=["POST"])
def process_frame():
    try:
        data = request.json
        data_base64 = data["data"]
        shape = tuple(data["shape"])

        # 解码 base64 数据
        data_bytes = base64.b64decode(data_base64)

        # 将字节数据转换为 NumPy 数组
        frame = np.frombuffer(data_bytes, dtype=np.uint8).reshape(shape)
        # 确保数组是可写的
        if not frame.flags.writeable:
            frame = np.copy(frame)
        # 处理帧
        with torch.no_grad():
            result = vlm_server.process_frame(frame)

        return jsonify(result.tolist()), 200
    except Exception as e:
        error_msg = f"Error: {str(e)}\nStack trace:\n{traceback.format_exc()}"
        print(error_msg)  # 在服务器端打印完整错误信息

        # 在开发环境中可以返回详细错误信息，生产环境建议只返回基本错误
        if app.debug:
            return jsonify(
                {"error": str(e), "stack_trace": traceback.format_exc()}
            ), 400
        return jsonify({"error": "Internal server error"}), 500


@app.route("/process_sole_frame", methods=["POST"])
def process_sole_frame():
    try:
        data = request.json
        labels = data["label"]
        data_base64 = data["data"]
        shape = tuple(data["shape"])

        # 解码 base64 数据
        data_bytes = base64.b64decode(data_base64)

        # 将字节数据转换为 NumPy 数组
        frame = np.frombuffer(data_bytes, dtype=np.uint8).reshape(shape)
        # 确保数组是可写的
        if not frame.flags.writeable:
            frame = np.copy(frame)
        # 处理帧
        result = vlm_server.process_sole_frame(labels, frame)

        return jsonify(result.tolist()), 200
    except Exception as e:
        error_msg = f"Error: {str(e)}\nStack trace:\n{traceback.format_exc()}"
        print(error_msg)  # 在服务器端打印完整错误信息

        # 在开发环境中可以返回详细错误信息，生产环境建议只返回基本错误
        if app.debug:
            return jsonify(
                {"error": str(e), "stack_trace": traceback.format_exc()}
            ), 400
        return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
