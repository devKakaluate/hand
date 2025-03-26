import cv2
import mediapipe as mp


class HandDetector():
    def __init__(self):
        # 获取视频中有没有手的信息
        self.hand_detector = mp.solutions.hands.Hands()
        # 给mp.solutions.drawing_utils一个比较简单的名字drawer
        self.drawer = mp.solutions.drawing_utils

    def process(self, img, draw):
        # 将opencv获取到的视频色彩由bgr(blue,green,red)转化为rgb
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 处理视频图片
        self.hand_data = self.hand_detector.process(img_rgb)

        if draw:
            # 如果有手势数据
            if self.hand_data.multi_hand_landmarks:
                # 遍历手的21个数据
                for handlms in self.hand_data.multi_hand_landmarks:
                    # 用mp把21个handlms数据画到img图片上,第三个参数表示把关节点连起来
                    self.drawer.draw_landmarks(img, handlms, mp.solutions.hands.HAND_CONNECTIONS)

    def find_position(self, img):
        # 获取视频的高度、宽度
        h, w, c = img.shape
        # 定义一个position变量，存放每一只手的数据，存放在字典中
        position = {'Left': {}, 'Right': {}}
        # 如果有手势数据
        if self.hand_data.multi_hand_landmarks:
            i = 0
            for point in self.hand_data.multi_handedness:
                # 获取手的分类（是伸的左手还是右手）
                # 有三个数据 index、score(表示是哪一手的概率）、label(哪一只手)
                """
                 classification{ 
                    index:0 
                    score:0.9611464738845825 
                    label:"Left" 
                  """
                # 把score数据读出来
                score = point.classification[0].score
                if score >= 0.8:
                    # 获取的左手或右手的数据
                    label = point.classification[0].label
                    # 把21个关节取出来，分别存到hand_lms中
                    hand_lms = self.hand_data.multi_hand_landmarks[i].landmark
                    # id从0开始，lm有3个值，分别为x,y,z
                    for id, lm in enumerate(hand_lms):
                        # 获取点的位置
                        x, y = int(lm.x * w), int(lm.y * h)
                        # 放在position字典中，记录哪只手的哪个关节在哪个坐标
                        position[label][id] = (x, y)
                i = i + 1
        return position