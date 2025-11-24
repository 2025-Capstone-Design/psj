import cv2
import mss
import numpy as np
import time
from ultralytics import YOLO

from PIL import ImageFont, ImageDraw, Image

# 시스템에 설치된 한글 폰트 경로를 지정해야 합니다.
# 윈도우 기본 폰트 경로 예시:
# FONT_PATH = "C:/Windows/Fonts/malgun.ttf" # 맑은 고딕
# Linux 환경을 위해 기본 폰트 경로를 주석 처리하고 사용자에게 알립니다.
# 젯슨(Linux) 환경에서 폰트 문제 발생 시, 'nanumgothic.ttf' 등으로 변경 필요
FONT_PATH = "C:/Windows/Fonts/malgun.ttf" # 맑은 고딕 (사용자 환경에 맞게 변경 필요)
FONT_SIZE = 30 # 폰트 크기

# ==============================================================================
# 1. 전역 변수 및 마우스 이벤트 핸들러 (화면 캡처 영역 설정)
# ==============================================================================

# 캡처할 영역의 좌표를 저장하는 딕셔너리
capture_area = {'top': 0, 'left': 0, 'width': 0, 'height': 0}
is_drawing = False
is_roi_selected = False
temp_frame = None

def select_roi(event, x, y, flags, param):
    global capture_area, is_drawing, is_roi_selected, temp_frame
    
    # 마우스 왼쪽 버튼 클릭: 드래그 시작
    if event == cv2.EVENT_LBUTTONDOWN:
        is_drawing = True
        is_roi_selected = False
        capture_area['left'] = x
        capture_area['top'] = y

    # 마우스 이동 중: 현재 드래그 영역 표시
    elif event == cv2.EVENT_MOUSEMOVE:
        if is_drawing:
            # 실시간으로 드래그 영역을 직사각형으로 표시
            frame_copy = temp_frame.copy()
            cv2.rectangle(frame_copy, (capture_area['left'], capture_area['top']), (x, y), (0, 255, 0), 2)
            cv2.imshow("Select Capture Area", frame_copy)

    # 마우스 왼쪽 버튼 떼기: 드래그 종료 및 영역 확정
    elif event == cv2.EVENT_LBUTTONUP:
        is_drawing = False
        is_roi_selected = True
        
        # 영역 계산 및 보정 (음수/제로 너비/높이 방지)
        x_end = x
        y_end = y
        capture_area['left'] = min(capture_area['left'], x_end)
        capture_area['top'] = min(capture_area['top'], y_end)
        capture_area['width'] = abs(x_end - capture_area['left'])
        capture_area['height'] = abs(y_end - capture_area['top'])
        
        # 최소 크기 보장 (너무 작은 영역 방지)
        if capture_area['width'] < 10 or capture_area['height'] < 10:
             print("영역이 너무 작습니다. 다시 선택해주세요.")
             is_roi_selected = False

def put_korean_text(img, text, pos, font_path, font_size, color=(0, 0, 255)):
    # 폰트 로드
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print(f"경고: 한글 폰트 파일({font_path})을 찾을 수 없습니다. 기본 폰트를 사용합니다.")
        # 폰트를 찾지 못할 경우 기본 폰트를 사용
        font = ImageFont.load_default() 
        font_size = 20 # 기본 폰트 크기 조정
        
    # OpenCV 이미지를 PIL 이미지로 변환
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # PIL을 사용하여 텍스트 그리기
    # pos는 (x, y) 좌표, color는 RGB 순서여야 함
    # (cv2.COLOR_BGR2RGB 변환 후 다시 BGR로 변환하는 과정에서 발생)
    draw.text(pos, text, font=font, fill=(color[2], color[1], color[0])) # BGR -> RGB 변환하여 사용
    
    # PIL 이미지를 OpenCV 이미지로 다시 변환
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# ==============================================================================
# 2. 메인 실행 함수
# ==============================================================================

def main():
    global temp_frame, is_roi_selected

    # ------------------ [상수 정의] ------------------
    PERSON_CLASS_ID = 0
    KNIFE_CLASS_ID = 43      # COCO: knife
    SCISSORS_CLASS_ID = 44   # COCO: scissors
    DANGER_OBJECT_IDS = [KNIFE_CLASS_ID, SCISSORS_CLASS_ID]
    # 감지 대상 클래스 (사람, 칼, 가위)
    TARGET_CLASSES = [PERSON_CLASS_ID] + DANGER_OBJECT_IDS

    # 감지 결과 시각화에 사용할 색상 (BGR 포맷)
    PERSON_COLOR = (255, 0, 0) # 파란색 (사람)
    DANGER_COLOR = (0, 0, 255) # 빨간색 (위험 물건)
    
    # 흉기 감지 지속 프레임 수 (약 1.5초 유지)
    DANGER_HOLD_FRAMES = 45 
    
    # YOLOv8 추적 메모리: {track_id: 남은_위험_프레임_수}
    DANGER_MEMORY = {}

    # YOLOv8 모델 로드
    
    # [⭐ 젯슨 오린 나노 최적화 설정 완료 ⭐]
    # 이 코드는 TensorRT 엔진 파일에 최적화되어 있습니다.
    # 1. yolov8x.pt 모델을 yolov8x.engine 파일로 변환하세요.
    # 2. 이 코드를 젯슨에서 실행하면 초고속 실시간 감지가 가능합니다.
    model_path = 'yolov8x.engine' # 🚀 젯슨 오린 나노 최적화 파일
    # model_path = 'yolov8x.pt'       # PC 환경에서 다시 테스트하려면 이 줄의 주석을 해제하세요.
    
    try:
        model = YOLO(model_path) 
    except Exception as e:
        print(f"오류: 모델 파일 로드 실패. 파일 경로를 확인하거나 변환을 수행하세요. ({model_path})")
        print(f"에러 메시지: {e}")
        return
    
    # 모델 로드 후 클래스 이름 딕셔너리를 사용합니다.
    CLASS_NAMES = model.names 

    # ------------------ 2D 평면도 설정 ------------------
    map_width, map_height = 500, 500
    floor_map = np.full((map_height, map_width, 3), 255, dtype=np.uint8) # 흰색 배경

    # ------------------ 스크린 영역 선택 ------------------
    # 젯슨 환경에서는 mss 대신 카메라 입력을 사용할 가능성이 높습니다.
    # 이 코드는 데스크톱 화면 캡처(mss)를 사용하도록 설계되었으므로, 
    # 젯슨에서 웹캠 입력을 사용하려면 이 부분을 cv2.VideoCapture(0) 등으로 변경해야 합니다.
    print("화면 캡처 영역을 마우스 드래그로 선택해주세요. 'q'를 누르면 종료됩니다.")
    
    try:
        with mss.mss() as sct:
            # 초기 화면 전체 캡처
            sct_img = sct.grab(sct.monitors[0]) # 첫 번째 모니터 전체
            temp_frame = np.array(sct_img)
            temp_frame = cv2.cvtColor(temp_frame, cv2.COLOR_BGRA2BGR)
            
            cv2.namedWindow("Select Capture Area")
            cv2.setMouseCallback("Select Capture Area", select_roi)
            
            # 영역이 선택될 때까지 화면을 표시
            while not is_roi_selected:
                cv2.imshow("Select Capture Area", temp_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    return
            
            cv2.destroyWindow("Select Capture Area")
            print(f"캡처 영역 설정 완료: {capture_area}")

            # ------------------ 실시간 분석 루프 ------------------
            print("\n--- 실시간 분석 모드 시작: 흉기 감지 로직 활성화 (conf=0.20) ---")
            while True:
                start_time = time.time()
                
                # 1. 윈도우 화면 캡처 (설정된 영역)
                sct_img = sct.grab(capture_area)
                frame = np.array(sct_img)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                
                # 2. 객체 추적 및 감지 (사람, 칼, 가위만 감지하며, 흉기 감지 민감도 조정)
                results = model.track(
                    frame, 
                    classes=TARGET_CLASSES, # 사람, 칼, 가위만 감지
                    verbose=False, 
                    iou=0.30, 
                    conf=0.20, # 흉기 감지 민감도 (실험적 값)
                    persist=True, 
                    tracker='bytetrack.yaml' 
                    )

                # 3. 위험 물건 감지 및 시각화 로직
                current_map = floor_map.copy()
                annotated_frame = frame.copy()
                
                # 이번 프레임에서 위험 물건을 소지한 것으로 판단된 사람의 트랙 ID 집합
                current_danger_track_ids = set() 
                
                if results and len(results[0].boxes) > 0:
                    boxes_tensor = results[0].boxes
                    boxes = boxes_tensor.xyxy.cpu().numpy().astype(int)
                    classes = boxes_tensor.cls.cpu().numpy().astype(int)
                    conf_scores = boxes_tensor.conf.cpu().numpy()
                    track_ids = boxes_tensor.id.cpu().numpy().astype(int) if boxes_tensor.id is not None else [-1] * len(boxes)
                    
                    person_data = [] # (box, track_id, center_x, center_y, size)
                    danger_data = []  # (box, center_x, center_y, conf_score)
                    
                    # 객체 분류 및 데이터 준비
                    for box, cls_id, conf, track_id in zip(boxes, classes, conf_scores, track_ids):
                        x1, y1, x2, y2 = box
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        
                        if cls_id == PERSON_CLASS_ID:
                            # 사람 데이터 수집 (크기 계산 포함)
                            width = x2 - x1
                            height = y2 - y1
                            avg_size = (width + height) / 2
                            person_data.append((box, track_id, center_x, center_y, avg_size))
                            # 사람 박스 그리기 (기본: 파란색)
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), PERSON_COLOR, 2)
                            
                        elif cls_id in DANGER_OBJECT_IDS:
                            # 위험 물건 데이터 수집
                            danger_data.append((box, center_x, center_y, conf))
                            # 위험 물건 박스 그리기 (빨간색)
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), DANGER_COLOR, 3)
                            label = CLASS_NAMES.get(cls_id, 'Danger')
                            # 신뢰도 점수도 함께 표시
                            cv2.putText(annotated_frame, f"{label} ({conf:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, DANGER_COLOR, 2)

                    # 4. **[알고리즘 개선] 근접 기반 위험 연관 분석** (로직 복구)
                    for d_box, d_cx, d_cy, d_conf in danger_data:
                        min_distance = float('inf')
                        closest_person_id = -1
                        closest_person_size = 0

                        for p_box, p_track_id, p_cx, p_cy, p_size in person_data:
                            if p_track_id == -1:
                                continue

                            # 유클리드 거리 계산
                            distance = np.sqrt((d_cx - p_cx)**2 + (d_cy - p_cy)**2)
                            
                            if distance < min_distance:
                                min_distance = distance
                                closest_person_id = p_track_id
                                closest_person_size = p_size
                                
                        # **위험 판별 임계값:** 가장 가까운 사람의 크기(size)의 60% 이내에 흉기가 있다면 소지한 것으로 간주
                        PROXIMITY_THRESHOLD = closest_person_size * 0.6 
                        
                        if closest_person_id != -1 and min_distance < PROXIMITY_THRESHOLD:
                            current_danger_track_ids.add(closest_person_id)


                    # 5. 메모리 업데이트 및 최종 시각화 (로직 복구)
                    
                    # 메모리 업데이트: 이번 프레임에서 소지자로 확인된 사람은 프레임 수를 최대로 설정
                    for track_id in current_danger_track_ids:
                        DANGER_MEMORY[track_id] = DANGER_HOLD_FRAMES
                    
                    # 메모리 감소: 이번 프레임에서 소지자로 확인되지 않은 사람들의 메모리 감소
                    for track_id in list(DANGER_MEMORY.keys()):
                        if track_id not in current_danger_track_ids and DANGER_MEMORY[track_id] > 0:
                            DANGER_MEMORY[track_id] -= 1
                        
                        # 메모리 만료 시 키 삭제
                        if DANGER_MEMORY[track_id] <= 0:
                            del DANGER_MEMORY[track_id]

                    # 최종 시각화
                    for p_box, p_track_id, p_cx, p_cy, p_size in person_data:
                        # 박스 좌표를 루프 시작 시 언팩
                        px1, py1, px2, py2 = p_box 
                        
                        is_highlighted_danger = DANGER_MEMORY.get(p_track_id, 0) > 0
                        
                        if is_highlighted_danger:
                            # 사람 박스를 빨간색 테두리로 다시 그려 강조
                            cv2.rectangle(annotated_frame, (px1, py1), (px2, py2), DANGER_COLOR, 4)
                            
                            # 폰트 경로 오류를 대비하여 예외 처리된 put_korean_text 사용
                            annotated_frame = put_korean_text(
                                 annotated_frame, 
                                 f"ID {p_track_id} : 흉기 소지 (위험)", 
                                 (px1, py1 - 40), 
                                 FONT_PATH, 
                                 FONT_SIZE, 
                                 color=DANGER_COLOR
                             )
                            
                            # 2D 평면도에 빨간색 점으로 표시
                            # 2D 맵에 표시할 때 캡처 영역에 대한 상대 좌표를 사용
                            map_x = int(p_cx / capture_area['width'] * map_width)
                            # 사람 발밑(y2) 위치를 기준으로 맵에 표시
                            map_y = int(py2 / capture_area['height'] * map_height)
                            cv2.circle(current_map, (map_x, map_y), 8, DANGER_COLOR, -1) 

                        else:
                            # 일반 사람 시각화 (2D 맵에 일반 사람 위치 표시)
                            map_x = int(p_cx / capture_area['width'] * map_width)
                            map_y = int(py2 / capture_area['height'] * map_height)
                            cv2.circle(current_map, (map_x, map_y), 5, PERSON_COLOR, -1) # 파란색 점
                else:
                    # 감지 결과가 없을 경우, 원본 캡처 프레임을 그대로 사용
                    annotated_frame = frame.copy()


                # ----------------------------------------------------
                # 6. 결과 화면 표시 및 경고 문구 추가
                # ----------------------------------------------------
                
                # 캡처된 화면의 높이를 가져와서 2D 맵 크기 조정
                target_height = annotated_frame.shape[0]
                resized_current_map = cv2.resize(
                    current_map, 
                    (map_width, target_height), 
                    interpolation=cv2.INTER_LINEAR
                )
                
                # FPS 계산
                end_time = time.time()
                fps = 1 / (end_time - start_time)
                cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # 전체 위험 감지 플래그 (메모리에 위험 인물이 남아있는지 확인)
                overall_danger = len(DANGER_MEMORY) > 0
                
                alert_text = "*********흉기 소지 감지됨: 즉각 대응 필요**********" if overall_danger else "위험 없음"
                text_color = DANGER_COLOR if overall_danger else (0, 255, 0) # 녹색으로 변경 (안전)

                annotated_frame = put_korean_text(
                    annotated_frame, 
                    alert_text, 
                    (10, 60), 
                    FONT_PATH, 
                    FONT_SIZE, 
                    color=text_color
                )

                # 두 이미지 수평 합치기
                combined_display = np.hstack((annotated_frame, resized_current_map))

                cv2.imshow("Crowd Analysis System (Combined)", combined_display)

                # 'q'를 누르면 종료
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cv2.destroyAllWindows()
            
    except Exception as e:
        print(f"시스템 오류 발생: {e}")
        print("mss 모듈이 젯슨 오린 나노 환경에서 화면 캡처 대신 웹캠 입력을 사용하도록 수정해야 할 수 있습니다.")


if __name__ == "__main__":
    main()