---
layout: post
title:  "can't open camera by index_error fix"
summary: yolov5 error fix
author: yeop-giraffe
date: '2022-08-10'
category: Drone
tag: error
thumbnail: /assets/img/posts/error_message.png
---

# 1. rror messasge
  File "/home/hms_yeop/Desktop/yolov5/utils/dataloaders.py", line 339, in __init__
    assert cap.isOpened(), f'{st}Failed to open {s}'
AssertionError: 1/1: 0... Failed to open 0

![error_code](/assets/img/posts/error_message.png){:style="border:1px solid #eaeaea; border-radius: 7px; padding: 0px;" }

# 2. Solution
## 1) Connection test
```terminal
v4l2-ctl --all 
```
- Driver Info에서 Card type : Video Capture #의 숫자 확인

## 2) yolov5 실행
```
python path/to/detect.py --weights yolov5s.pt --source 0
```
- source 0 대신 위에서 확인한 숫자# 넣기
- -- source #

## 3) opencv 확인(*yolov5 실행 안될 시)
```
import cv2

capture = cv2.VideoCapture(4)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cv2.waitKey(33) < 0:
    ret, frame = capture.read()
    cv2.imshow("VideoFrame", frame)

capture.release()
cv2.destroyAllWindows()
```
- capture = cv2.VideoCapture(#) : #부분에 위에서 확인한 숫자 넣기
- 카메라가 작동되는데 yolo가 안된다면 다른 문제를 해결해라