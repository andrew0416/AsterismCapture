import cv2
import numpy as np

def calculate_mean_color(image, brightness_threshold=50):
    """Calculate the mean color of an image excluding dark areas."""
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = hsv_image[:, :, 2] > brightness_threshold  # 명도가 threshold 이상인 부분만 선택
    masked_pixels = image[mask]  # 마스크를 적용하여 픽셀 선택
    if masked_pixels.size == 0:  # 마스크된 픽셀이 없는 경우, 평균을 기본 색상으로 반환
        return [0, 0, 0]  # 어두운 이미지일 경우 기본 검은색 반환
    mean_color = np.mean(masked_pixels, axis=0)
    return [int(c) for c in mean_color]  # 색상을 정수로 변환

def color_distance(c1, c2):
    """Calculate the Euclidean distance between two colors."""
    return np.linalg.norm(np.array(c1) - np.array(c2))

def remove_close_locations(locations, min_dist=10):
    """중복 위치를 제거합니다."""
    if not locations:
        return []
    
    # 중복 제거된 위치를 저장할 리스트
    filtered_locations = []

    # 모든 위치를 검사하며 중복을 제거
    for loc in locations:
        keep = True
        for kept_loc in filtered_locations:
            # 기존에 저장된 위치와의 거리 계산
            dist = np.sqrt((loc[0] - kept_loc[0]) ** 2 + (loc[1] - kept_loc[1]) ** 2)
            if dist < min_dist:
                keep = False
                break
        if keep:
            filtered_locations.append(loc)

    return filtered_locations

# 색상과 라벨 설정
label_colors = {
    0: [180, 50, 50],   # Blue
    1: [50, 50, 180],   # Red
    2: [50, 180, 50],   # Green
    3: [0, 130, 130],   # Yellow
    4: [0, 80, 160]     # Orange
}

# 템플릿 이미지와 임계값 설정
template_settings = [
    {'path': './Asterism/activate_fire.png', 'threshold': 0.5, 'color': (0, 0, 255)},
    {'path': './Asterism/activate_metal.png', 'threshold': 0.5, 'color': (0, 165, 255)},
    {'path': './Asterism/activate_water.png', 'threshold': 0.5, 'color': (255, 0, 0)},
    {'path': './Asterism/activate_wood.png', 'threshold': 0.5, 'color': (0, 255, 0)},
    {'path': './Asterism/activate_earth.png', 'threshold': 0.5, 'color': (0, 255, 255)},
    {'path': './Asterism/fire.png', 'threshold': 0.6}
]
# 템플릿 이미지 로드 및 임계값 저장
templates = [{'image': cv2.imread(setting['path'], 0), 'threshold': setting['threshold'], 'color': setting.get('color', (255, 255, 255)), 'path': setting['path']} for setting in template_settings]

# 대상 컬러 이미지 로드
image_path = './Asterism/sample3.png'
image_color = cv2.imread(image_path)
image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)

# 모든 템플릿에 대해 템플릿 매칭 실행 및 바운딩 박스 중앙 좌표 저장
bounding_boxes = []

for template in templates:
    result = cv2.matchTemplate(image_gray, template['image'], cv2.TM_CCOEFF_NORMED)
    
    # 임계값 이상의 매칭 결과 찾기
    threshold = template['threshold']
    locations = np.where(result >= threshold)
    locations = list(zip(*locations[::-1]))  # (x, y) 형식으로 변환
    
    # 비슷한 위치 제거
    locations = remove_close_locations(locations, min_dist=10)

    # 매칭된 위치에 사각형 그리기
    for loc in locations:
        top_left = loc
        bottom_right = (top_left[0] + template['image'].shape[1], top_left[1] + template['image'].shape[0])
        center_x = (top_left[0] + bottom_right[0]) // 2
        center_y = (top_left[1] + bottom_right[1]) // 2

        if template['path'] == './Asterism/fire.png':
            # 박스 내부의 평균 색상 계산
            cropped_image = image_color[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
            avg_color = calculate_mean_color(cropped_image, brightness_threshold=50)
            
            # 가장 비슷한 색상 라벨 찾기
            best_label = min(label_colors.keys(), key=lambda k: color_distance(label_colors[k], avg_color))
            bounding_boxes.append((center_x, center_y, best_label))
            label_color = [int(c) for c in label_colors[best_label]]  # 라벨의 색상을 정수로 변환
            cv2.putText(image_color, str(best_label), (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 2)
            cv2.rectangle(image_color, top_left, bottom_right, avg_color, 2)
        else:
            color = template['color']
            cv2.rectangle(image_color, top_left, bottom_right, color, 2)
            bounding_boxes.append((center_x, center_y, 5))

# 바운딩 박스 리스트를 x 기준으로 정렬
bounding_boxes.sort()

# Load graph templates
template_paths = [
    './Asterism/graph1.png',
    './Asterism/graph2.png',
    './Asterism/graph3.png'
]

# Load images
img1 = cv2.imread(image_path, 0)  # Ensure the path is correct
img1 = cv2.GaussianBlur(img1, (5, 5), 0)  # Apply Gaussian Blur with a kernel size of 5x5

# Set all pixels above a certain brightness threshold to black
brightness_threshold = 100  # Example threshold, adjust as needed
_, mask = cv2.threshold(img1, brightness_threshold, 255, cv2.THRESH_BINARY)
img1[mask == 255] = 0

img2 = cv2.imread(template_paths[0], 0)
img3 = cv2.imread(template_paths[1], 0)
img4 = cv2.imread(template_paths[2], 0)

# Perform template matching
res2 = cv2.matchTemplate(img1, img2, cv2.TM_CCOEFF_NORMED)
res3 = cv2.matchTemplate(img1, img3, cv2.TM_CCOEFF_NORMED)
res4 = cv2.matchTemplate(img1, img4, cv2.TM_CCOEFF_NORMED)

# Get the maximum match value for each comparison
max_val2 = cv2.minMaxLoc(res2)[1]
max_val3 = cv2.minMaxLoc(res3)[1]
max_val4 = cv2.minMaxLoc(res4)[1]

# Print the scores for each template
#print(f"Score for template 1: {max_val2}")
#print(f"Score for template 2: {max_val3}")
#print(f"Score for template 3: {max_val4}")

# Determine which has the highest match value

import networkx as nx
import matplotlib.pyplot as plt

# 주어진 인접 리스트
type_1 = {0: [1], 1: [0, 2, 3, 4], 2: [1, 3], 3: [1, 2, 4], 4: [1, 3, 5], 5: [4, 6], 6: [5, 7, 8], 7: [6],
          8: [6, 10, 11], 9: [10], 10: [8, 9, 11], 11: [8, 10]}
type_2 = {0: [2], 1: [2, 3], 2: [0, 1, 3], 3: [2, 4, 5], 4: [3, 6], 5: [3, 7], 6: [4, 7, 8], 7: [5, 6, 8],
          8: [6, 7, 10], 9: [10], 10: [8, 9, 11], 11: [10]}
type_3 = {0: [1], 1: [0, 2, 4], 2: [1, 4], 3: [4, 5], 4: [1, 2, 6], 5: [3, 6, 7], 6: [4, 5, 7], 7: [5, 6, 8],
          8: [7, 9], 9: [8, 10, 11], 10: [9], 11: [9]}

# 그래프 생성 함수
def create_graph(adj_list):
    G = nx.Graph()
    for key, values in adj_list.items():
        for value in values:
            G.add_edge(key, value)
    return G

G = None
ordered_box = []
max_val = max(max_val2, max_val3, max_val4)
match_type = ""
if max_val == max_val2:
    match_type = "type 1"
    
    ordered_box += [bounding_boxes[0]]
    
    temp = [bounding_boxes[1],bounding_boxes[2]]
    temp.sort(key=lambda x: x[1])
    ordered_box += temp
    
    ordered_box += [bounding_boxes[3]]
    ordered_box += [bounding_boxes[4]]
    
    temp = [bounding_boxes[5],bounding_boxes[6],bounding_boxes[7]]
    temp.sort(key=lambda x: x[1])
    ordered_box += temp

    ordered_box += [bounding_boxes[8]]
    
    temp = [bounding_boxes[9],bounding_boxes[10],bounding_boxes[11]]
    temp.sort(key=lambda x: x[1])
    ordered_box += temp
    
    # 그래프 생성
    G = create_graph(type_1)
       
elif max_val == max_val3:
    match_type = "type 2"
    
    ordered_box += [bounding_boxes[0]]
    
    temp = [bounding_boxes[1],bounding_boxes[2]]
    temp.sort(key=lambda x: x[1])
    ordered_box += temp
    
    ordered_box += [bounding_boxes[3]]
    
    temp =  [bounding_boxes[4],bounding_boxes[5]]
    temp.sort(key=lambda x: x[1])
    ordered_box += temp
    
    temp =  [bounding_boxes[6],bounding_boxes[7]]
    temp.sort(key=lambda x: x[1])
    ordered_box += temp

    ordered_box += [bounding_boxes[8]]
    
    temp = [bounding_boxes[9],bounding_boxes[10],bounding_boxes[11]]
    temp.sort(key=lambda x: x[1])
    ordered_box += temp
    
    G = create_graph(type_2)
    
elif max_val == max_val4:
    match_type = "type 3"
    
    ordered_box += [bounding_boxes[0]]
    
    temp = [bounding_boxes[1],bounding_boxes[2]]
    temp.sort(key=lambda x: x[1])
    ordered_box += temp
    
    temp = [bounding_boxes[3],bounding_boxes[4]]
    temp.sort(key=lambda x: x[1])
    ordered_box += temp
    
    temp = [bounding_boxes[5],bounding_boxes[6]]
    temp.sort(key=lambda x: x[1])
    ordered_box += temp
    
    ordered_box += [bounding_boxes[7]]
    
    temp = [bounding_boxes[8],bounding_boxes[9],bounding_boxes[10]]
    temp.sort(key=lambda x: x[1])
    ordered_box += temp
    
    ordered_box += [bounding_boxes[11]]
    
    G = create_graph(type_3)
    


# Draw the match type on the image
cv2.putText(image_color, match_type, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# Save the result image with the type labeled
#result_image_path = './Asterism/result_labeled.png'
#cv2.imwrite(result_image_path, image_color)

# Save the preprocessed image for verification
cv2.imwrite('./Asterism/test.png', img1)

print(match_type)
# Print sorted bounding box centers and labels
for bbox in ordered_box:
    print(f"Center: ({bbox[0]}, {bbox[1]}), Label: {bbox[2]}")


for edge in list(G.edges()):
    pt1 = (ordered_box[edge[0]][0],ordered_box[edge[0]][1])
    pt2 = (ordered_box[edge[1]][0],ordered_box[edge[1]][1])
    cv2.line(image_color, pt1, pt2, (255, 255, 255), 1)

# Save the result image with the type labeled
result_image_path = './Asterism/result_labeled.png'
cv2.imwrite(result_image_path, image_color)











