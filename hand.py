import cv2
from handms import HandDetector

"""
1.使用opencv来获取摄像头信息
2.使用mediapipe将手的信息传给opencv
"""
# 打开电脑上的摄像头,笔记本的参数是0
camera = cv2.VideoCapture(0)
# 在另一个文件中创建一个类，里面存放mediapipe实现的功能，并导入该文件
hand_detector = HandDetector()
while True:
    # 从摄像头里面读,获取摄像头视频帧的数据
    # 会返回两个数据，一个是判断是否读成功，另一个是返回视频帧的图片
    success, img = camera.read()
    # 如果读取成功
    if success:
        # opencv获取到的视频是翻转的，需要将视频改为镜像的
        img = cv2.flip(img, 1)
        # 调用类里面的函数process,将img放到mediapipe中进行处理
        hand_detector.process(img, True)
        # 储存position字典
        position = hand_detector.find_position(img)
        # 获取到左手的食指指尖，如果获取不到返回None
        left_finger = position['Left'].get(8, None)
        if left_finger:
            # 如果获取到，在指尖画一个小圆，0表示x坐标，1表示y坐标，半径，颜色，实心
            cv2.circle(img, (left_finger[0], left_finger[1]), 10, (0, 0, 255), cv2.FILLED)
        # 获取到右手的食指指尖，如果获取不到返回None
        right_finger = position['Right'].get(8, None)
        if right_finger:
            # 如果获取到，在指尖画一个小圆，0表示x坐标，1表示y坐标，半径，颜色，实心
            cv2.circle(img, (right_finger[0], right_finger[1]), 10, (0, 255, 0), cv2.FILLED)
        # 代码运行后，会产生一个窗口，名为Video，
        cv2.imshow('Video', img)
    # 按下一个按键，使其停止运行（结束while循环）
    # 等待按键按下，如果在1毫秒之内感知到的话，waitKey就会接收该键值，并把该键赋值给k
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

# 关闭摄像头，解除程序占用摄像头
camera.release()
# cv2把所有打开的窗口关闭掉
cv2.destroyAllWindows()