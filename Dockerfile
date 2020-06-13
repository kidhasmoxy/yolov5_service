# Define base images
ARG app_image="ultralytics/yolov5:latest"

# App image:
FROM ${app_image}

WORKDIR /usr/src/app

# Install api requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Get weights
RUN python3 -c "from utils.google_utils import *; \
attempt_download('weights/yolov5s.pt'); \
attempt_download('weights/yolov5m.pt'); \
attempt_download('weights/yolov5l.pt'); \
attempt_download('weights/yolov5x.pt'); \
attempt_download('weights/yolov3-spp.pt')"

# Model to use (defaults to yolov5):
ARG weights_file="weights/yolov5l.pt"
ARG config_file="cfg/yolov4.cfg"
ARG meta_file="cfg/coco.data"
ARG img_size=640

ENV weights_file=${weights_file}
ENV config_file=${config_file}
ENV meta_file=${meta_file}
ENV img_size=${img_size}
ENV optimized_memory=${optimized_memory}
ENV augment=False
ENV agnostic_nms=False
ENV device=
ENV half=False
ENV iou_thres=0.5



COPY app.py .
COPY swagger.yaml .

CMD ["python3", "app.py"]