import pika
import cv2
import os
import uuid
import time

RTSP_URL = "rtsp://admin:admin@192.168.2.210:554/stream/main"
PATH = '/home/ai/java/image'
PATH1 = 'C:\image1'


def get_image(flag=True,CM=210):
    if flag:
        credentials = pika.PlainCredentials('admin', 'swsc2018!')
        save_connection = pika.BlockingConnection(
            pika.ConnectionParameters(host='192.168.2.93', port=5672, virtual_host='/', credentials=credentials,
                                      heartbeat=20)
        )
        save_channel = save_connection.channel()
        save_channel.queue_declare(queue='images', durable=True)
    if CM!=210:
        RTSP_URL = "rtsp://admin:admin@192.168.2.211:554/stream/main"
    cap = cv2.VideoCapture(RTSP_URL)
    ret, frame = cap.read(0)
    # print(frame)
    while frame is None:
        print('网络错误,读取不到图片')
        time.sleep(2)
        cap = cv2.VideoCapture(RTSP_URL)
        ret,frame = cap.read(0)
    img_name = '2019' + str(uuid.uuid4()) + '.jpg'
    if flag==True:
        image_path = os.path.join(PATH,img_name)
    else:
        image_path = os.path.join(PATH1,img_name)
    if flag:
        frame = cv2.resize(frame, (964,540), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(image_path,frame)
    print('save image to', image_path)
    if CM!=210:
        image_path = image_path+","+"192.168.2.211"
    else:
        image_path = image_path+","+"192.168.2.210"
    if flag:
        save_channel.basic_publish(exchange='face-exchange',routing_key='images',body=image_path)
        save_connection.close()
    else:
        return image_path



if __name__ == '__main__':
    get_image(flag=False)