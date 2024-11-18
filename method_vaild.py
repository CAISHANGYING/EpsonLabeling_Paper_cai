import os
import sys
import time
import numpy as np
import copy
import json
import cv2
import numpy.linalg
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from detect.hole_detect_process import HoleDetectConvector

sys.path.append(os.getcwd())

from detect.hole.hole_detect_paper import HoleDetectModel
from workpiece.stand_workpiece import StandWorkpiece
from detect.push_back.hole_push_back import PushBackModel

with open('./RemoteServer/remote_0531.json') as f:
    config = json.load(f)

hole_detect = HoleDetectModel(config['yolov5'])

# stand workpiece initialize
with open(config['stand']['parent_foleder'] + '/' + config['stand']['stand_tool_name'] + '.json') as f:
    workpieceJsonData = dict(json.load(f))

stand_workpiece = StandWorkpiece(workpieceJsonData)
push_back = PushBackModel(config['push_back'], stand_workpiece)

# 计算IOU的函数
def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2

    xi1, yi1 = max(x1, x1_p), max(y1, y1_p)
    xi2, yi2 = min(x2, x2_p), min(y2, y2_p)

    inter_area = max(0, xi2 - xi1 + 1) * max(0, yi2 - yi1 + 1)
    box1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    box2_area = (x2_p - x1_p + 1) * (y2_p - y1_p + 1)

    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area

# 计算中心点距离误差的函数
def calculate_center_distance(box1, box2):
    center1 = [(box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2]
    center2 = [(box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2]
    distance = np.linalg.norm(np.array(center1) - np.array(center2))
    return distance

# 计算归一化误差的函数
def calculate_normalized_error(box1, box2):
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    normalized_error = abs(box1_area - box2_area) / max(box1_area, box2_area)
    return normalized_error

# 加载CSV文件
# file_path = 'botsort_w_test_paper2.csv'
file_path = 'botsort_w_test.csv'
data = pd.read_csv(file_path)

# 视频路径
video_path = './test_video/paper_test01.mp4'

results = []

# 尝试的次数
try_times = list(range(50, 750, 50))

for try_time in try_times:

    print(f'Try Time: {try_time}')
    print('-' * 100)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():

        print("Cannot open camera")
        exit()

    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ious_per_frame = []
    distances_per_frame = []
    errors_per_frame = []
    detect_time_frame = []
    pushback_time_frame = []

    pbar = tqdm(total=total_frames, desc=f'Try Time {try_time}')

    while True:

        ret, frame = cap.read()

        if not ret:

            break

        if frame_count >= total_frames:
            break

        height, width, _ = frame.shape

        hole_detail, det, detect_time = hole_detect.detect(frame)

        detect_time_frame.append(detect_time)

        # cv2.imshow('detect', det)

        result_of_convert = HoleDetectConvector.convert(hole_detail, (width, height), (width, height))
        hole_convert = result_of_convert['hole']
        wrench_convert = result_of_convert['wrench']
        boundary_convert = list(result_of_convert['boundary'])

        boundary_convert = [] ### set boundary_convert to empty list to disable perspective transformation

        if boundary_convert :

            start_time = time.time()

            stand_boundary = list(stand_workpiece.getBoundaryDetail())
            stand_boundary = HoleDetectConvector.boundary_convert(stand_boundary, [stand_workpiece.getShape()[1], stand_workpiece.getShape()[0]], (width, height))

            matrix, hole_tranform = HoleDetectConvector.holePerspectiveTransform(hole_convert, boundary_convert, np.array(stand_boundary, dtype=np.float32))

            try:
                _, hole_tranform_convert = HoleDetectConvector.holePerspectiveTransform(copy.deepcopy(hole_tranform), [], [], np.linalg.inv(matrix))

                result_of_pushback = push_back.getPushbackPosition(hole_tranform, tryTime=try_time, magnificationAgain=1.0, matrix=matrix)

            except numpy.linalg.LinAlgError:

                result_of_pushback = push_back.getPushbackPosition(hole_convert, tryTime=try_time, magnificationAgain=1.0)

            end_time = time.time()

            pushback_time = end_time - start_time
            pushback_time_frame.append(pushback_time)

        else:

            start_time = time.time()

            result_of_pushback = push_back.getPushbackPosition(hole_convert, tryTime=try_time, magnificationAgain=1.0)

            end_time = time.time()

            pushback_time = end_time - start_time
            pushback_time_frame.append(pushback_time)

        wrench_wrong_screw_index = HoleDetectConvector.wrenchOnWrongScrewPosition(wrench_convert, result_of_pushback, "1")
        # print("wrong:" + wrench_wrong_screw_index.__str__())

        hole_result_convert = HoleDetectConvector.covertToResultType(result_of_pushback, (width, height), 0)
        push_back_frame = HoleDetectConvector.getPushbackFrame(frame, result_of_pushback)

        for key, value in result_of_pushback.items():

            if 'coordinate_real' in value:

                coordinate = value['coordinate_real']

            else:
                coordinate = value['coordinate']

            result_of_pushback[key]['box'] = [
                int(coordinate['left_top'][0]),
                int(coordinate['left_top'][1]),
                int(coordinate['right_bottom'][0]),
                int(coordinate['right_bottom'][1])
            ]

        if frame_count < len(data):
            frame_data = data.iloc[frame_count]
            frame_ious = []
            frame_distances = []
            frame_errors = []

            for i in range(1, 17):
                hole_str = frame_data[f'hole_{i}']
                hole = json.loads(hole_str.replace("'", '"'))
                x1, y1, x2, y2 = map(int, hole)

                tag = str(i)

                for key, value in result_of_pushback.items():
                    if value['tag'] == tag:
                        standard_box = value['box']
                        iou = calculate_iou([x1, y1, x2, y2], standard_box)
                        distance = calculate_center_distance([x1, y1, x2, y2], standard_box)
                        error = calculate_normalized_error([x1, y1, x2, y2], standard_box)
                        frame_ious.append(iou)
                        frame_distances.append(distance)
                        frame_errors.append(error)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f'{tag}: IOU={iou:.2f}, Dist={distance:.2f}, Err={error:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        break

            ious_per_frame.append(sum(frame_ious) / len(frame_ious) if frame_ious else 0)
            distances_per_frame.append(sum(frame_distances) / len(frame_distances) if frame_distances else 0)
            errors_per_frame.append(sum(frame_errors) / len(frame_errors) if frame_errors else 0)

        # cv2.imshow('test result', _3)
        # cv2.imshow('Vaild', frame)

        if cv2.waitKey(1) == ord('q'):
            break

        frame_count += 1
        pbar.update(1)

    cap.release()
    cv2.destroyAllWindows()
    pbar.close()

    # 计算总的平均值
    total_iou = sum(ious_per_frame) / len(ious_per_frame) if ious_per_frame else 0
    total_distance = sum(distances_per_frame) / len(distances_per_frame) if distances_per_frame else 0
    total_error = sum(errors_per_frame) / len(errors_per_frame) if errors_per_frame else 0
    avg_detect_time = sum(detect_time_frame) / len(detect_time_frame) if detect_time_frame else 0
    avg_pushback_time = sum(pushback_time_frame) / len(pushback_time_frame) * 1000 if pushback_time_frame else 0

    # 将结果存储到列表中
    current_result = {
        "try_time": try_time,
        "average_iou": total_iou,
        "average_center_distance": total_distance,
        "average_relative_area_error": total_error,
        "average_detect_time_ms": avg_detect_time,
        "average_pushback_time_ms": avg_pushback_time
    }

    results.append(current_result)


    print(f'Results for try_time {try_time}:')

    current_result_df = pd.DataFrame([current_result])
    print(current_result_df)

print()
print('=' * 150)
print('Results Summary:')
print('-' * 100)
# 将结果转换为 DataFrame
results_df = pd.DataFrame(results)

# 输出结果
print(results_df)

# 保存结果为 CSV 文件
results_df.to_csv('results_summary.csv', index=False)

# 绘制结果并保存为图表
plt.figure(figsize=(14, 8))

plt.subplot(2, 3, 1)
plt.plot(results_df['try_time'], results_df['average_iou'], marker='o')
plt.title('Average IOU vs. Try Time')
plt.xlabel('Try Time')
plt.ylabel('Average IOU')

plt.subplot(2, 3, 2)
plt.plot(results_df['try_time'], results_df['average_center_distance'], marker='o')
plt.title('Average Center Distance vs. Try Time')
plt.xlabel('Try Time')
plt.ylabel('Average Center Distance')

plt.subplot(2, 3, 3)
plt.plot(results_df['try_time'], results_df['average_relative_area_error'], marker='o')
plt.title('Average Relative Area Error vs. Try Time')
plt.xlabel('Try Time')
plt.ylabel('Average Relative Area Error')

plt.subplot(2, 3, 4)
plt.plot(results_df['try_time'], results_df['average_detect_time_ms'], marker='o')
plt.title('Average Detect Time vs. Try Time')
plt.xlabel('Try Time')
plt.ylabel('Average Detect Time (ms)')

plt.subplot(2, 3, 5)
plt.plot(results_df['try_time'], results_df['average_pushback_time_ms'], marker='o')
plt.title('Average Pushback Time vs. Try Time')
plt.xlabel('Try Time')
plt.ylabel('Average Pushback Time (ms)')

plt.tight_layout()
plt.savefig('results_summary.png')
plt.show()

## Merge two results_summary.csv files
#
# import pandas as pd
# import matplotlib.pyplot as plt
#
# # 读取文件
# file_path1 = 'results_summary_have.csv'  # 替换为你的文件路径
# file_path2 = 'results_summary_no.csv'  # 替换为你的文件路径
#
# results_df1 = pd.read_csv(file_path1)
# results_df2 = pd.read_csv(file_path2)
#
# # 绘制图表
# plt.figure(figsize=(14, 8))
#
# plt.subplot(2, 3, 1)
# plt.plot(results_df1['try_time'], results_df1['average_iou'], marker='o', label='Use Perspective Transformation')
# plt.plot(results_df2['try_time'], results_df2['average_iou'], marker='o', label='No Use Perspective Transformation')
# plt.title('Average IOU vs. Try Time')
# plt.xlabel('Try Time')
# plt.ylabel('Average IOU')
#
# plt.subplot(2, 3, 2)
# plt.plot(results_df1['try_time'], results_df1['average_center_distance'], marker='o', label='Use Perspective Transformation')
# plt.plot(results_df2['try_time'], results_df2['average_center_distance'], marker='o', label='No Use Perspective Transformation')
# plt.title('Average Center Distance vs. Try Time')
# plt.xlabel('Try Time')
# plt.ylabel('Average Center Distance')
#
# plt.subplot(2, 3, 3)
# plt.plot(results_df1['try_time'], results_df1['average_relative_area_error'], marker='o', label='Use Perspective Transformation')
# plt.plot(results_df2['try_time'], results_df2['average_relative_area_error'], marker='o', label='No Use Perspective Transformation')
# plt.title('Average Relative Area Error vs. Try Time')
# plt.xlabel('Try Time')
# plt.ylabel('Average Relative Area Error')
#
# plt.subplot(2, 3, 4)
# plt.plot(results_df1['try_time'], results_df1['average_detect_time_ms'], marker='o', label='Use Perspective Transformation')
# plt.plot(results_df2['try_time'], results_df2['average_detect_time_ms'], marker='o', label='No Use Perspective Transformation')
# plt.title('Average Detect Time vs. Try Time')
# plt.xlabel('Try Time')
# plt.ylabel('Average Detect Time (ms)')
#
# plt.subplot(2, 3, 5)
# plt.plot(results_df1['try_time'], results_df1['average_pushback_time_ms'], marker='o', label='Use Perspective Transformation')
# plt.plot(results_df2['try_time'], results_df2['average_pushback_time_ms'], marker='o', label='No Use Perspective Transformation')
# plt.title('Average Pushback Time vs. Try Time')
# plt.xlabel('Try Time')
# plt.ylabel('Average Pushback Time (ms)')
#
# # 在右下角空位添加图例
# handles, labels = plt.gca().get_legend_handles_labels()
# plt.subplot(2, 3, 6)
# plt.axis('off')  # 关闭坐标轴
# plt.legend(handles, labels, loc='center', frameon=False, fontsize=14)  # 在空白区域居中显示图例
#
# plt.tight_layout()
# plt.savefig('results_summary_merge.png')
# plt.show()


## Grenerate the results_summary.csv and results_summary.png
#
# import pandas as pd
# import matplotlib.pyplot as plt
#
# # 读取文件
# file_path = 'results_summary.csv'
# results_df = pd.read_csv(file_path)
#
#
# # 绘制图表
# plt.figure(figsize=(14, 8))
#
# plt.subplot(2, 3, 1)
# plt.plot(results_df['try_time'], results_df['average_iou'], marker='o')
# plt.title('Average IOU vs. Try Time')
# plt.xlabel('Try Time')
# plt.ylabel('Average IOU')
#
# plt.subplot(2, 3, 2)
# plt.plot(results_df['try_time'], results_df['average_center_distance'], marker='o')
# plt.title('Average Center Distance vs. Try Time')
# plt.xlabel('Try Time')
# plt.ylabel('Average Center Distance')
#
# plt.subplot(2, 3, 3)
# plt.plot(results_df['try_time'], results_df['average_relative_area_error'], marker='o')
# plt.title('Average Relative Area Error vs. Try Time')
# plt.xlabel('Try Time')
# plt.ylabel('Average Relative Area Error')
#
# plt.subplot(2, 3, 4)
# plt.plot(results_df['try_time'], results_df['average_detect_time_ms'], marker='o')
# plt.title('Average Detect Time vs. Try Time')
# plt.xlabel('Try Time')
# plt.ylabel('Average Detect Time (ms)')
#
# plt.subplot(2, 3, 5)
# plt.plot(results_df['try_time'], results_df['average_pushback_time_ms'], marker='o')
# plt.title('Average Pushback Time vs. Try Time')
# plt.xlabel('Try Time')
# plt.ylabel('Average Pushback Time (ms)')
#
# plt.tight_layout()
# plt.savefig('results_summary.png')
# plt.show()

