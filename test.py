import mmcv
from mmcv import imread
import mmengine
from mmengine.registry import init_default_scope
import numpy as np

from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

local_runtime = False   #코드가 로컬머신에서 실행되는지 여부

try:
    from google.colab.patches import cv2_imshow  # for image visualization in colab 코랩환경에서 이미지 시각화
except:
    local_runtime = True

img = '26.jpg'   #이미지 파일 경로
pose_config = 'configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py'  #포즈추정기 모델의 구성 파일 경로, 아키텍처와 설정 정의
pose_checkpoint = 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth'  #포즈 추정기 모델의 체크포인트 파일 경로
det_config = 'demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py'  #객체 검출기 모델의 구성 파일 경로
det_checkpoint = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'   #객체 검출기 모델의 체크포인트 파일경로

device = 'cpu'  #GPU사용
cfg_options = dict(model=dict(test_cfg=dict(output_heatmaps=True)))  #히트맵 출력


# build detector
detector = init_detector(   #객체검출기 초기화
    det_config,   #객체 검출기의 설정 파일
    det_checkpoint,   #체크포인트 파일
    device=device    #GPU
)


# build pose estimator
pose_estimator = init_pose_estimator(  #포즈 추정기 초기화
    pose_config,   #포즈 추정기의 설정 파일
    pose_checkpoint,   #체크포인트 파일
    device=device,   #GPU
    #cfg_options=cfg_options   #모델의 추가 설정(히트맵 출력)
)

# init visualizer
pose_estimator.cfg.visualizer.radius = 3   #반지름
pose_estimator.cfg.visualizer.line_width = 1   #선두께
visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)   #시각화부분 초기화
# the dataset_meta is loaded from the checkpoint and
# then pass to the model in init_pose_estimator
visualizer.set_dataset_meta(pose_estimator.dataset_meta)   #시각화 도구에 데이터셋 메타 정보를 설정하는 부분




def visualize_img(img_path, detector, pose_estimator, visualizer,
                  show_interval, out_file):
    """Visualize predicted keypoints (and heatmaps) of one image."""

    # predict bbox
    scope = detector.cfg.get('default_scope', 'mmdet')   #객체 검출기의 설정에서 default_scope값을 가져옴, 없으면 mmdet설정
    if scope is not None:
        init_default_scope(scope)  #가져온 설정값을 사용하여 디폴트 스코프를 초기화
    detect_result = inference_detector(detector, img_path)   #초기화된 객체 검출기를 사용하여 이미지에서 객체를 검출하고, 검출된 결과를 반환
    pred_instance = detect_result.pred_instances.cpu().numpy()   #검출된 결과에서 예측된 인스턴스를 추출하고, 이를 CPU로 이동하여 넘파이 배열로 변환 / 바운딩 박스좌표, 클래스 점수
    bboxes = np.concatenate(
        (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)  #객체의 경계 상자 좌표와 점수를 결함하여 배열을 생성합니다. / score는 객체가 해당클래스에 속할 확률또는 신뢰도
    bboxes = bboxes[np.logical_and(pred_instance.labels == 0,
                                   pred_instance.scores > 0.3)]   #점수가 0.3보다 크고, 라벨이 0인 객체에 대해서만 선택
    bboxes = bboxes[nms(bboxes, 0.3)][:, :4]   #선택된 객체들에 대해 비최대 억제를 수행하고, 결과에서 상위 4개의 값을 선택하여 경계상자를 최종결정

    # predict keypoints
    pose_results = inference_topdown(pose_estimator, img_path, bboxes)  #바운딩박스를 사용하여 해당 객체에 대한 포즈를 추정
    data_samples = merge_data_samples(pose_results)   #여러개의 포즈결과를 하나의 데이터 샘플로 병합

    # show the results
    img = mmcv.imread(img_path, channel_order='rgb')  #이미지 읽어들이고 RGB형식으로 채널 순서 지정

    visualizer.add_datasample(  #시각화할 데이터 샘플 추가
        'result',   #시각화된 결과의 이름 지정
        img,   #이미지 데이터
        data_sample=data_samples,  #시각화할 데이터 샘플
        draw_gt=False,   #Ground Truth그릴지 여부
        draw_heatmap=False,   #히트맵 박스 그릴지 여부
        draw_bbox=True,   #바운딩 박스 그릴지 여부
        show=False,   #결과를 화면에 표시할지 여부
        wait_time=show_interval,   #결과표시 후 대기하는 시간
        out_file=out_file,   #결과를 저장할 파일 경로
        kpt_thr=0.3)   #키포인트를 그릴때 사용할 임계값
    return data_samples


data_samples = visualize_img(   #객체 검출 및 포즈 추정 결과 시각화, 결과 이미지를 반환하는 함수
    img,   #이미지 파일 경로
    detector,   #객체 검출기 모델
    pose_estimator,   #포즈 추정기 모델
    visualizer,   #시각화를 수행하는 객체
    show_interval=0,   #0으로 설정하면 결과 이미지를 반환하고, 양수값으로 설정하면 지정된 시간마다 이미지가 표시
    out_file=None)   #저장할 이미지 파일 경로

import numpy as np

# pred_instances.keypoint_scores의 평균 계산
mean_scores = np.mean(data_samples.pred_instances.keypoint_scores, axis=1)

# 평균이 0.3 이상인 인덱스 추출
indices = np.where(mean_scores >= 0.3)[0]

# 평균이 0.3 이상인 인스턴스만 추출
filtered_instances = data_samples.pred_instances[indices]


print(filtered_instances.keypoints)