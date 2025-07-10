import os
from io import BytesIO

import numpy as np
import orjson
import torch
import torch.nn.functional as F
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from flask import Flask, request
from PIL import Image, ImageFilter

from irl_dcb.config import JsonConfig
from irl_dcb.environment import IRL_Env4LHF
from irl_dcb.models import LHF_Policy_Cond_Small

# Flask server
app = Flask("hat-model-server")
app.logger.setLevel("DEBUG")

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
app.logger.info(f"Using device: {device}")


# Initialize model, config and load checkpoint
# Note: This server currently only supports target present mode!
hparams = JsonConfig("hparams/coco_search18.json")
preset_tasks = np.load("./all_task_ids.npy", allow_pickle=True).item()
preset_tasks = {str(k): int(v) for k, v in preset_tasks.items()}
task_eye = torch.eye(len(preset_tasks), device=device)
input_size = 80 + 54

model = LHF_Policy_Cond_Small(
    hparams.Data.patch_count, len(preset_tasks), task_eye, input_size
).to(device)
model_checkpoint = torch.load(
    "trained_models/trained_generator.pkg", map_location="cpu"
)
model.load_state_dict(model_checkpoint["model"])
model.eval()

# Initialize Detectron2 predictor on CPU
cfg = get_cfg()
cfg.merge_from_file(
    model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
)
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"
)
predictor = DefaultPredictor(cfg)


def extract_DCBs(img, predictor, radius=1):
    # Create DCB features for image using Detectron2
    high = img.convert("RGB").resize((hparams.Data.im_w, hparams.Data.im_h))
    low = high.filter(ImageFilter.GaussianBlur(radius=radius))

    high_panoptic_seg, high_segments_info = predictor(np.array(high))["panoptic_seg"]
    low_panoptic_seg, low_segments_info = predictor(np.array(low))["panoptic_seg"]

    def pred2feat(seg, info):
        seg = seg.cpu()
        feat = torch.zeros(
            [80 + 54, hparams.Data.im_h, hparams.Data.im_w], device=device
        )
        for pred in info:
            mask = (seg == pred["id"]).float()
            mask = mask.to(device)
            if pred["isthing"]:  # Things categories (0-79)
                feat[pred["category_id"], :, :] = mask * pred["score"]
            else:  # Stuff categories (80-133)
                feat[pred["category_id"] + 80, :, :] = mask
        return F.interpolate(
            feat.unsqueeze(0), size=hparams.Data.patch_num[::-1]
        ).squeeze(0)

    high_feat = pred2feat(high_panoptic_seg, high_segments_info)
    low_feat = pred2feat(low_panoptic_seg, low_segments_info)
    return high_feat, low_feat


def get_fixation_history(x_hist, y_hist, image_size):
    # Fixations are normalized and truncated to max_traj_length 6
    x = x_hist * hparams.Data.patch_num[0] // image_size[0]
    y = y_hist * hparams.Data.patch_num[1] // image_size[1]

    fixation_hist = y * hparams.Data.patch_num[0] + x
    return fixation_hist[-hparams.Data.max_traj_length :]


@app.route("/conditional_log_density/<int:task_id>", methods=["POST"])
def conditional_log_density(task_id):
    # Extract stimulus
    image_bytes = request.files["stimulus"].read()
    image = Image.open(BytesIO(image_bytes))

    # Extract scanpath history
    data = orjson.loads(request.form["json_data"])
    x_hist = np.array(data["x_hist"])
    y_hist = np.array(data["y_hist"])
    fixation_hist = get_fixation_history(x_hist, y_hist, image.size)
    fixation_hist = torch.tensor(fixation_hist, device=device, dtype=torch.long)

    # Extract DCBs
    high_feat, low_feat = extract_DCBs(image, predictor)

    # Set up environment
    env = IRL_Env4LHF(
        hparams.Data,
        max_step=hparams.Data.max_traj_length,
        mask_size=hparams.Data.IOR_size,
        status_update_mtd=hparams.Train.stop_criteria,
        device=device,
        inhibit_return=True,
        init_mtd="center",
    )
    env_data = {
        "label_coding": torch.zeros((1, 1, hparams.Data.patch_count), device=device),
        "img_name": ["test"],
        "cat_name": ["test"],
        "init_fix": torch.tensor([[0, 0]], dtype=torch.long, device=device),
        "action_mask": torch.zeros(
            (1, 1, hparams.Data.patch_count), dtype=torch.bool, device=device
        ),
        "history_map": torch.zeros((1, *hparams.Data.patch_num[::-1]), device=device),
        "task_id": torch.tensor([task_id], device=device, dtype=torch.long),
        "lr_feats": low_feat.unsqueeze(0),
        "hr_feats": high_feat.unsqueeze(0),
    }
    env.set_data(env_data)

    # Update the environment with the fixation history
    for i in fixation_hist:
        env.step(i.reshape(1, 1))

    # Observe environment and sample from the policy
    obs_fov = env.observe()
    probs, values = model(obs_fov, env.task_ids)
    probs = F.interpolate(
        probs.reshape(1, 1, *hparams.Data.patch_num[::-1]),
        size=(image.size[1], image.size[0]),
        mode="bicubic",
    ).squeeze()
    # bicubic interpolation can create negative value; clamp to epsilon for stability
    probs = probs.clamp(min=1e-30)
    probs /= probs.sum()
    log_density = torch.log(probs)

    log_density_list = log_density.cpu().tolist()
    response = orjson.dumps({"log_density": log_density_list})
    return response


@app.route("/type", methods=["GET"])
def type():
    type = "Scanpath_Prediction"
    version = "v1.0.0"
    return orjson.dumps({"type": type, "version": version})


@app.route("/task_ids_map", methods=["GET"])
def task_ids_map():
    return orjson.dumps({"task_ids_map": preset_tasks})


def main():
    app.run(host="localhost", port="4000", debug="True", threaded=True)


if __name__ == "__main__":
    main()
