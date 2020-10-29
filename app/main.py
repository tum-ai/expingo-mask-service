import os
import numpy as np
import uvicorn

from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
from pydantic import BaseModel
from typing import List, Tuple
from PIL import Image

from mrcnn import utils
import mrcnn.model as modellib
from mrcnn.config import Config


ROOT_DIR = os.path.abspath(".")
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "../mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


class InferenceConfig(Config):
    NAME = "Coco Inference Config"
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Give the configuration a recognizable name

    # Number of classes (including background)
    NUM_CLASSES = 1 + 80  # COCO has 80 classes


config = InferenceConfig()
config.display()


# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)


# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


class MaskClasses(BaseModel):
    class_list: List[str]


app = FastAPI()

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/getMasks/")
async def create_upload_file(classes: List[str] = Query(None), file: UploadFile = File(...)):
    """ Massive improvements in error handling neccesasry!!!
        Also file saving needs to be improved. We should use a tmpdir with a unique hash per request to
        save the images and delete them after transfer. We need to keep in mind that this is async.
        https://fastapi.tiangolo.com/tutorial/background-tasks/ this might be useful but must be delayed.
    """
    content = await file.read()
    image = cv2.imdecode(np.fromstring(content, np.uint8), cv2.IMREAD_COLOR)
    valid_classes = drop_invalid_classes(classes)
    if len(valid_classes) == 0:
        raise HTTPException(status_code=400, detail="You must choose at least one valid class for masking.")

    result = model.detect([image], verbose=1)[0]
    class_ids = result['class_ids']

    masks = result['masks']
    detected_classes = [class_names[idx] for idx in class_ids]

    relevant_classes, relevant_masks = filter_selected_classes(
        selected_classes=classes,
        detected_classes=detected_classes,
        masks=masks
    )

    for i in range(len(relevant_masks)):
        save_image(relevant_masks[i], name=f"mask_{i}.png")

    return {
        "masks": [FileResponse(f"mask_{i}.png") for i in range(len(relevant_masks))],
        "classes": relevant_classes
    }


def drop_invalid_classes(classes: List[str]) -> List[str]:
    for c in classes:
        if c not in class_names:
            classes.remove(c)
    return classes


def save_image(image_arr: np.array, name: str):
    im = Image.fromarray(image_arr)
    im.save(f"{name}.png")


def filter_selected_classes(
        selected_classes: List[str],
        detected_classes: List[str],
        masks: np.array) -> Tuple[List[str], List[np.array]]:
    final_classes = []
    final_masks = []
    for i, c in enumerate(detected_classes):
        if c in selected_classes:
            final_classes.append(c)
            final_masks.append(masks[i])
    return final_classes, final_masks


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8080)
