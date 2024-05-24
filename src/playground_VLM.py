import os
from VLMPipline.VLM import VLM
import cv2
import torch
from matplotlib.animation import FuncAnimation

import matplotlib.pyplot as plt

# shutdown torch grad
with torch.no_grad():
    owlv2_model_path = os.path.expanduser(
        "~/models/google-owlv2-large-patch14-finetuned"
    )
    owlv2_model_path = os.path.expanduser("~/models/google-owlv2-base-patch16-ensemble")
    sam_model_path = os.path.expanduser("~/models/facebook-sam-vit-huge")
    # sam_model_path = os.path.expanduser("~/models/facebook-sam-vit-base")
    xmem_model_path = os.path.expanduser("~/models/XMem.pth")
    resnet_18_path = os.path.expanduser("~/models/resnet18.pth")
    resnet_50_path = os.path.expanduser("~/models/resnet50.pth")

    vlmpipeline = VLM(
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

    video_path = "/shared/codes/VoxPoser/VID_20240427_135016.mp4"
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    # reshape cv2 frame to c h w
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB
    frame = frame.transpose(2, 0, 1)
    start_time_processing = cv2.getTickCount()
    result = vlmpipeline.process_first_frame(
        ["Wooden house"], frame, owlv2_threshold=0.2
    )
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
        result = vlmpipeline.process_frame(frame, release_video_memory=True)
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
        fig, update, frames=range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))), interval=33
    )
    anim.save("animation_VID_20240427_135016.gif", writer="imagemagick")
