# Few Shot Protypical Anomaly Detection

    This code is was created as a part of the Final-year Project, by Parth Patil, at IIT Bombay.


# Abstract

    Anomaly detection refers to the act of identifying behaviour in data which is different than normal. This can range from detecting malicious activity in a video to detecting earthquake in a seismic data. Anomaly detection methods based on convolutional neural networks (CNNs) typically leverage proxy tasks, such as reconstructing input video frames, to learn models describing normality without seeing anomalous samples at training time, and quantify the extent of abnormalities using the explicit reconstruction error at test time. The main drawbacks of these approaches are that they do not consider the diversity of normal patterns. Also, many existing approaches requires large number of normal samples from a particular scene before it could detect anomalies. This makes deployment in real world impractical. In this report we will have a look at approach which tackles both the issues. We will also look at the detail code which can enable any anomaly detection model to be adapted for a new scene using few frames.

# Usage

## Training

```python
python Train.py \
  --dataset_type <dataset_name> \
  --model_dir <path to pre-trained model> \
  --m_items_dir <path to pre-trained memory items> \
  --dataset_path <path to dataset> \
  --log_dir <path to log folder> \
  --iterations <no of iterations> \
  --epochs <no of epochs> \
```

## Testing
```python
python Evaluate.py \
  --dataset_type <dataset_name> \
  --model_dir <path to pre-trained model> \
  --m_items_dir <path to pre-trained memory items> \
  --dataset_path <path to dataset> \
  --log_dir <path to log folder> \
  --save_anomaly_list <boolean>
```