![](./yolov11.webp)

# Traffic Sign Detection using YOLOv11

* [Traffic Sign Detection using YOLOv11](#traffic-sign-detection-using-yolov11)

  * [Data](#data)
  * [Model](#model)
  * [Fine Tuning](#fine-tuning)

    * [YOLO11n summary (fused)](#yolo11n-summary-fused)
    * [Run summary](#run-summary)
  * [Detections](#detections)
  * [Dependencies](#dependencies)
  * [Project Setup](#project-setup)
  * [Running Inference (Video or Webcam)](#running-inference-video-or-webcam)
  * [Limitations](#limitations)
  * [Conclusion](#conclusion)
  * [Acknowledgements](#acknowledgements)

## Data

The [Self-Driving Cars Dataset](https://universe.roboflow.com/selfdriving-car-qtywx/self-driving-cars-lfjou/dataset/6) is used to train the traffic sign detection model. It contains **4969** total images split into train, val and test sets with **3530**, **801** and **638** images of dimension `416x416` respectively. The dataset contains images of 15 different traffic signs.

The classes available in the dataset are:

1. all
2. Green Light
3. Red Light
4. Speed Limit 100
5. Speed Limit 110
6. Speed Limit 120
7. Speed Limit 20
8. Speed Limit 30
9. Speed Limit 40
10. Speed Limit 50
11. Speed Limit 60
12. Speed Limit 70
13. Speed Limit 80
14. Speed Limit 90
15. Stop

## Model

The `yolo11n` version of the model is used to fine-tune on the dataset. The model was trained for **50** epochs with batch size **16**.

*Note*: The .ipy notebook is not uploaded due to privacy issues.

## Fine Tuning

### YOLO11n summary (fused)

238 layers, 2,585,077 parameters, 0 gradients, 6.3 GFLOPs

| Class           | Images | Instances | Box(P | R     | mAP50 | mAP50-95) |
| --------------- | ------ | --------- | ----- | ----- | ----- | --------- |
| all             | 801    | 944       | 0.95  | 0.905 | 0.959 | 0.836     |
| Green Light     | 87     | 122       | 0.901 | 0.743 | 0.851 | 0.525     |
| Red Light       | 74     | 108       | 0.891 | 0.722 | 0.844 | 0.529     |
| Speed Limit 100 | 52     | 52        | 0.95  | 0.942 | 0.989 | 0.889     |
| Speed Limit 110 | 17     | 17        | 0.916 | 1     | 0.986 | 0.915     |
| Speed Limit 120 | 60     | 60        | 1     | 0.943 | 0.995 | 0.908     |
| Speed Limit 20  | 56     | 56        | 0.981 | 0.93  | 0.985 | 0.871     |
| Speed Limit 30  | 71     | 74        | 0.963 | 0.959 | 0.984 | 0.924     |
| Speed Limit 40  | 53     | 55        | 0.935 | 0.945 | 0.988 | 0.887     |
| Speed Limit 50  | 68     | 71        | 0.973 | 0.915 | 0.98  | 0.886     |
| Speed Limit 60  | 76     | 76        | 0.92  | 0.912 | 0.96  | 0.89      |
| Speed Limit 70  | 78     | 78        | 0.987 | 0.962 | 0.981 | 0.9       |
| Speed Limit 80  | 56     | 56        | 0.96  | 0.929 | 0.973 | 0.866     |
| Speed Limit 90  | 38     | 38        | 0.954 | 0.789 | 0.924 | 0.784     |
| Stop            | 81     | 81        | 0.975 | 0.982 | 0.988 | 0.929     |

### Run summary

| Parameter / Metric      | Value   |
| ----------------------- | ------- |
| lr/pg0                  | 2e-05   |
| lr/pg1                  | 2e-05   |
| lr/pg2                  | 2e-05   |
| metrics/mAP50(B)        | 0.95912 |
| metrics/mAP50-95(B)     | 0.83597 |
| metrics/precision(B)    | 0.95049 |
| metrics/recall(B)       | 0.90534 |
| model/GFLOPs            | 6.456   |
| model/parameters        | 2592765 |
| model/speed_PyTorch(ms) | 3.062   |
| train/box_loss          | 0.47508 |
| train/cls_loss          | 0.33472 |
| train/dfl_loss          | 0.90102 |
| val/box_loss            | 0.55826 |
| val/cls_loss            | 0.34385 |
| val/dfl_loss            | 0.95067 |

## Detections

<div style="display: flex; flex-direction: column; align-items: center;">
    <div>
        <img src="./runs/detect/predict/speed_limit_40.jpg" width="49%">
        <img src="./runs/detect/predict5/green_light.jpg" width="49%">
    </div>
    <div>
        <img src="./runs/detect/predict3/red_light.jpg" width="49%">
        <img src="./runs/detect/predict2/stop_sign.jpg" width="49%">
    </div>
    <div>
        <img src="./runs/detect/predict4/speed_limit_50.jpg" width="49%">
        <img src="./runs/detect/predict6/speed_limit_30.jpg" width="49%">
    </div>
</div>

## Dependencies

* python 3.x
* opencv_contrib_python
* opencv_python
* ultralytics

## Project Setup

1. Create a virtual environment:

   ```bash
   python3 -m venv myenv
   ```

2. Activate the virtual environment:

   ```bash
   source myenv/bin/activate
   ```

3. Clone the repository:

   ```bash
   git clone https://github.com/bhaskrr/traffic-sign-detection-using-yolov11.git
   ```

4. Navigate to the project directory:

   ```bash
   cd traffic-sign-detection-using-yolov11
   ```

5. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Running Inference (Video or Webcam)

The inference script supports **both video files and real-time webcam input**, selectable directly from the terminal using command-line arguments.

### Webcam inference

```bash
python detect.py --source webcam
```

### Video inference

```bash
python detect.py --source video --video_path ./data/input/traffic_signs.mp4
```

### Arguments

* `--source` : Input source (`webcam` or `video`)
* `--video_path` : Path to the input video (required only when using `video` source)

Press **`q`** to exit the inference window.

## Limitations

The model performs best on images and shows reliable performance on videos under favorable lighting and camera angles. For improved real-time performance or complex environments, fine-tuning a larger YOLOv11 variant or optimizing inference resolution is recommended.

## Conclusion

This project demonstrates how a fine-tuned YOLOv11 model can be used for accurate traffic sign detection across images, videos, and real-time webcam streams.

### Use cases

1. Autonomous vehicle navigation
2. Driver assistance systems
3. Road safety training and simulations
4. Smart city traffic monitoring
5. Road network and traffic analysis

## Acknowledgements

The media files used to test the model predictions are taken from [pexels.com](https://www.pexels.com/).
