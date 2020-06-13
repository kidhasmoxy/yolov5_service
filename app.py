import os
import sys
import signal
import connexion
import urllib.request
import tempfile
import ast

from utils.datasets import *
from utils.utils import *

# Setup handler to catch SIGTERM from Docker
def sigterm_handler(_signo, _stack_frame):
        print('Sigterm caught - closing down')
        sys.exit()

def detect(img0s, threshold, iou_thres=0.5 ):

    # Padded resize
    img = letterbox(img0s, new_shape=imgsz)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img, augment=augment)[0]

    if half:
            pred = pred.float()

    # Apply NMS
    pred = non_max_suppression(pred, threshold, iou_thres,
                            fast=True, classes=None, agnostic=None)

    # Apply Classifier
    if classify:
        pred = apply_classifier(pred, modelc, img, im0s)
    r = []

    for i, det in enumerate(pred):  # detections per image
        # gn = torch.tensor(img0s.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0s.shape).round()
            for *xyxy, conf, cls in det:
                # xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                coords = "[{},{},{},{}]".format(*xyxy)
                #print(coords)
                coords = ast.literal_eval(coords)
                json = "[\"{}\",{},{}]".format(names[int(cls)], conf, coords)
                #print(json)
                r.append(ast.literal_eval(json))
                #r.append([names[int(cls)], conf, coords])
                # print(('%s ' + ' %g ' * 5 + '\n') % (names[int(cls)],conf, *xyxy))
    return r

def detect_from_url(url, threshold):
#       try:
        # Use mkstemp to generate unique temporary filename
        resp = urllib.request.urlopen(url)
        image_file = np.asarray(bytearray(resp.read()), dtype="uint8")
        image_file = cv2.imdecode(image_file, cv2.IMREAD_UNCHANGED)
#       except:
#               return 'Error getting/reading file', 500
        try:
                res = detect(image_file, threshold, iou_thres)
        except:
                return 'Error in detection', 500
        return res

def detect_from_file():
#       try:
        uploaded_file = connexion.request.files['image_file']
        threshold = float(connexion.request.form['threshold'])
        img0s = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        #img0s = cv2.imdecode(uploaded_file.read(), cv2.IMREAD_UNCHANGED)
#       except:
#               return 'Error in getting/reading file', 500
#       try:
        res = detect(img0s, threshold, iou_thres)
#       except:
#               return 'Error in detection', 500
        #print(res)
        return res

# Load YOLO model:
#configPath = os.environ.get("config_file")
weights = os.environ.get("weights_file")

imgsz= int(os.environ.get("img_size"))
src_device=os.environ.get("device")
half=(os.environ.get("half").lower() == 'true')
classify=(os.environ.get("classify").lower() == 'true')
augment=(os.environ.get("augment").lower() == 'true')
iou_thres=float(os.environ.get("iou_thres"))

# Initialize
device = torch_utils.select_device(src_device)

# Load model
google_utils.attempt_download(weights)
model = torch.load(weights, map_location=device)['model']
torch.save(torch.load(weights, map_location=device), weights)  # update model if SourceChangeWarning
model.fuse()

model.to(device).eval()

# Second-stage classifier
if classify:
    modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
    modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
    modelc.to(device).eval()

# Half precision
half = half and device.type != 'cpu'  # half precision only supported on CUDA
if half:
    model.half()

# Get names and colors
names = model.names if hasattr(model, 'names') else model.modules.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

# prepare inference
img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
_ = model(img.half() if half else img.float()) if device.type != 'cpu' else None  # run once

# Create API:
app = connexion.App(__name__)
app.add_api('swagger.yaml')

if __name__ == '__main__':
    signal.signal(signal.SIGTERM, sigterm_handler)
    app.run(port=8080, server='gevent')

