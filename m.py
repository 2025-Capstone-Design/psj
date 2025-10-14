import cv2
import mss
import numpy as np
import time
from ultralytics import YOLO

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


# ==============================================================================
# 2. 메인 실행 함수
# ==============================================================================

def main():
    global temp_frame, is_roi_selected

    # YOLOv8 모델 로드 (가벼운 'n' 모델 사용)
    # Jetson 나중에 TensorRT로 최적화하면 속도가 훨씬 빨라집니다.
    model = YOLO('yolov8l.pt') 

    # ------------------ 2D 평면도 설정 ------------------
    # 평면도 이미지를 로드합니다. (실제 프로젝트에서 사용하는 이미지 경로로 변경)
    # 테스트를 위해 임시로 흰색 배경 이미지를 생성합니다.
    map_width, map_height = 500, 500
    floor_map = np.full((map_height, map_width, 3), 255, dtype=np.uint8) # 흰색 배경

    # ------------------ 스크린 영역 선택 ------------------
    print("화면 캡처 영역을 마우스 드래그로 선택해주세요.")
    
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
        while True:
            start_time = time.time()
            
            # 1. 윈도우 화면 캡처 (설정된 영역)
            sct_img = sct.grab(capture_area)
            frame = np.array(sct_img) 
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR) 
            
            # 2. 객체 감지 (사람 인식)
            # 클래스 필터링: 'person' (COCO 데이터셋에서 클래스 ID는 0, iou 및 conf로 감도 설정 낮출수록 인식률 증가하지만 정확도 낮아질수도 있음)
            results = model(frame, classes=[0], verbose=False, iou=0.01, conf=0.01) 
            
            # 3. 2D 평면도 시각화 및 원본 프레임 업데이트
            
            # 평면도를 다시 그리기 위해 초기화
            current_map = floor_map.copy() 
            
            if results:
                # 감지된 객체 정보 (바운딩 박스)
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                
                # 원본 프레임에 결과 시각화
                annotated_frame = results[0].plot() # YOLOv8의 기본 시각화 함수 사용

                for box in boxes:
                    x1, y1, x2, y2 = box
                    
                    # 바운딩 박스의 중심 좌표 (footprint 위치로 가정)
                    # 일반적으로 아래쪽 중앙을 씁니다: (x_center, y_bottom)
                    center_x = (x1 + x2) // 2
                    center_y = y2 

                    # 4. 2D 평면도에 좌표 매핑 (가장 단순한 선형 매핑)
                    # 실제 프로젝트에서는 원근 변환(Perspective Transform)이 필요합니다.
                    
                    # 현재 프레임 좌표를 평면도 좌표로 변환 (0~100% 비율 사용)
                    # (x 좌표) / (캡처 영역 너비) * (평면도 너비)
                    # (y 좌표) / (캡처 영역 높이) * (평면도 높이)
                    map_x = int(center_x / capture_area['width'] * map_width)
                    map_y = int(center_y / capture_area['height'] * map_height)
                    
                    # 평면도 좌표에 빨간 점 표시
                    cv2.circle(current_map, (map_x, map_y), 5, (0, 0, 255), -1) # BGR: 빨간색 (0, 0, 255)

            else:
                 # 감지 결과가 없을 경우, 원본 캡처 프레임을 그대로 사용
                 annotated_frame = frame.copy()


            # ----------------------------------------------------
            # 5. 결과 화면 표시 및 크기 조정 (수정된 부분)
            # ----------------------------------------------------
            
            # 캡처된 화면의 높이를 가져옵니다.
            target_height = annotated_frame.shape[0]

            # 2D 평면도의 높이를 캡처된 화면의 높이에 맞게 조정 (Resizing)
            # 평면도 너비(map_width=500)는 유지하고 높이만 맞춥니다.
            resized_current_map = cv2.resize(
                current_map, 
                (map_width, target_height), # (너비, 높이) 순서
                interpolation=cv2.INTER_LINEAR
            )
            
            # FPS 계산
            end_time = time.time()
            fps = 1 / (end_time - start_time)
            cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # 두 이미지의 높이가 같아졌으므로 수평 합치기 가능
            combined_display = np.hstack((annotated_frame, resized_current_map))

            cv2.imshow("Crowd Analysis System (Combined)", combined_display)

            # 'q'를 누르면 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()