# 프로그램 설명
- 이 프로그램은 별무리록이라는 미니게임의 현재 상태를 화면 캡쳐를 통해 알아내는 것을 목적으로 하는 프로그램입니다.
- 이 게임을 풀어내기 위한 로직을 개발중인데 그러기 위해서는 매 턴마다 판의 상태를 체크할 필요가 있고 이것을 하나하나 설정하는 것은 불편합니다.
- 그래서 이것을 좀 더 편하게 하기 위한 용도로 이 프로그램을 만들었습니다.

# 구성 요소 설명
- 게임판은 이렇게 생겼습니다.<br>
![sample2](https://github.com/andrew0416/AsterismCapture/assets/5708754/485eec08-c513-45bd-a5ce-08449c7c37a4)
- 이하의 아이콘이 존재할 수 있습니다.<br>
![image](https://github.com/andrew0416/AsterismCapture/assets/5708754/c240d1e3-395f-4551-a072-0138eae6fe35)
- 3종류의 그래프 형태가 존재합니다.<br>
![graph1](https://github.com/andrew0416/AsterismCapture/assets/5708754/7aaee194-c821-44ac-9d2b-7463ad970dc3)
![graph2](https://github.com/andrew0416/AsterismCapture/assets/5708754/df569e23-6a28-4dc8-89ac-f02250d15464)
![graph3](https://github.com/andrew0416/AsterismCapture/assets/5708754/fed0604c-d9bb-444f-a354-d1af6cf51167)
- 이러한 데이터를 활용하여 게임판의 현재 상태를 추출하는 것이 목표입니다.

# 프로그램 목표
- 게임판 캡쳐 사진을 기반으로 게임판의 현재 상태를 알아냅니다.
- 현재의 상태를 정확하게 판별했는지 알아내기 위해 알아낸 정보를 기반으로 원본 사진에 그림을 그립니다.
- 사용 용도가 명백하므로 다양한 환경에서 로버스트한 결과를 내는것보다는 정해진 환경에서 정확한 결과만 내도 충분합니다.

# 실행 결과
총 3개의 샘플 이미지를 준비했습니다.

### sample 1
- 원본<br>
![sample1](https://github.com/andrew0416/AsterismCapture/assets/5708754/236b4b8e-4c06-44a2-9c5b-e34cd91c24be)
- 결과<br>
![result_labeled](https://github.com/andrew0416/AsterismCapture/assets/5708754/cdceb5f6-acfd-48af-be20-5e0220501ac6)

### sample 2
- 원본<br>
![sample2](https://github.com/andrew0416/AsterismCapture/assets/5708754/563b0b8f-730f-4ffe-bedc-bd80e5efd9bd
- 결과<br>
![result_labeled](https://github.com/andrew0416/AsterismCapture/assets/5708754/384760c9-92da-43cf-a10d-ffa20bedf4ed)

### sample 3
- 원본<br>
![sample3](https://github.com/andrew0416/AsterismCapture/assets/5708754/f8e9852a-79f9-4bbd-a2a8-85ff1d492849)
- 결과<br>
![result_labeled](https://github.com/andrew0416/AsterismCapture/assets/5708754/310e44c0-58d1-44e5-bc80-e51ac3154628)

### 개선 사항
- 당장은 시각화를 더 중점적으로 두어 추출한 현재 상태를 목표 프로그램에게 전달하는 위한 기능이 없다.
- 보낼 정보들 자체는 프로그램 내부에 있으므로 나중에 써야할 때 수정할 예정이다.
