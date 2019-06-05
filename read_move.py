from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cv2
import tensorflow as tf
import numpy as np
import pika
import json
import base64
import os
import datetime
import uuid
import requests
from get_image import get_image
from time import sleep
from scipy import misc
from detection.align import detection_face
from detection.net import OPSnet
from config.settings import RDSP_URL, MODEL_DIR, EMB_IMG_DIR, SMALL_DIR, POST_URL, F_SMALL_DIR

POST_URL_NEW = "http://192.168.2.93:10077/faceapp/api/v1/syncResult"


def main():
    # with tf.Graph().as_default():
    #     gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    #     sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    #     with sess.as_default():
    #         pnet, rnet, onet = detection_face.create_sscnn(sess, None)
    with tf.Graph().as_default():
        with tf.Session() as sess:
            model = MODEL_DIR
            OPSnet.load_model(model)
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # 人脸库匹配位置
            # emb_dir = './img/emb_img'
            emb_dir = EMB_IMG_DIR
            # all_obj = []
            # image = []
            # nrof_images = 0
            # for i in os.listdir(emb_dir):
            #     all_obj.append(i)
            #     img = misc.imread(os.path.join(emb_dir, i), mode='RGB')
            #     prewhitened = OPSnet.prewhiten(img)
            #     image.append(prewhitened)
            #     nrof_images = nrof_images + 1
            # try:
            #     images = np.stack(image)
            #     feed_dict = {images_placeholder: images, phase_train_placeholder: False}
            #     compare_emb = sess.run(embeddings, feed_dict=feed_dict)
            #     compare_num = len(compare_emb)
            # except:
            #     pass

            credentials = pika.PlainCredentials('admin', 'swsc2018!')
            connection_clear = pika.BlockingConnection(
                pika.ConnectionParameters(host='192.168.2.93', port=5672, virtual_host='/', credentials=credentials)
            )
            clear_ack = connection_clear.channel()
            ack = clear_ack.basic_get(queue='ack', auto_ack=True)
            while ack[0]:
                ack = clear_ack.basic_get(queue='ack', auto_ack=True)
            connection_clear.close()
            print('清空队列')
            while True:
                print('循环识别')
                all_obj = []
                image = []
                nrof_images = 0
                for i in os.listdir(emb_dir):
                    all_obj.append(i)
                    img = misc.imread(os.path.join(emb_dir, i), mode='RGB')
                    prewhitened = OPSnet.prewhiten(img)
                    image.append(prewhitened)
                    nrof_images = nrof_images + 1
                try:
                    images = np.stack(image)
                    feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                    compare_emb = sess.run(embeddings, feed_dict=feed_dict)
                    compare_num = len(compare_emb)
                except:
                    pass
                ip_list = ["192.168.2.210","192.168.2.211"]
                ip_str = []
                for ip in ip_list:
                    if "210" in ip:
                        image_path = get_image(flag=False)
                    else:
                        image_path = get_image(flag=False,CM=211)
                    frame = cv2.imread(image_path)
                    os.remove(image_path)
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mark = load_and_align_data(rgb_frame, 160, 22)
                    if mark:
                        ip_str.append(ip)
                if ip_str:
                    ip_str = ','.join(ip_str)
                    credentials = pika.PlainCredentials('admin', 'swsc2018!')
                    save_connection = pika.BlockingConnection(
                        pika.ConnectionParameters(host='192.168.2.93', port=5672, virtual_host='/',
                                                  credentials=credentials,
                                                  heartbeat=20)
                    )
                    save_channel = save_connection.channel()
                    save_channel.queue_declare(queue='camera', durable=True)
                    save_channel.basic_publish(exchange='face-exchange', routing_key='images', body=ip_str)
                    save_connection.close()
                else:
                    sleep(20)
                    continue
                credentials = pika.PlainCredentials('admin', 'swsc2018!')

                image_connection = pika.BlockingConnection(
                    pika.ConnectionParameters(host='192.168.2.93', port=5672, virtual_host='test', credentials=credentials)
                )
                ack_connection = pika.BlockingConnection(
                    pika.ConnectionParameters(host='192.168.2.93', port=5672, virtual_host='/',
                                              credentials=credentials)
                )
                image_channel = image_connection.channel()
                ack_channel = ack_connection.channel()
                def facedecect(ch, method, properties, body):
                    path = str(body, encoding='utf-8')
                    path,ip = path.split(",")
                    print(path)
                    frame = cv2.imread(path)
                    os.remove(path)
                    frame = cv2.resize(frame, (964, 540), interpolation=cv2.INTER_CUBIC)
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mark, bounding_box, crop_image, crop_list, path_list = load_and_align_data1(rgb_frame, 160, 22)
                    print('mark is', mark)
                    if (mark):
                        feed_dict = {images_placeholder: crop_image, phase_train_placeholder: False}
                        emb = sess.run(embeddings, feed_dict=feed_dict)
                        temp_num = len(emb)
                        print("bbox num:", temp_num)
                        # if face_first_flag:
                        temp_num = len(emb)

                        fin_obj = []
                        # print(all_obj)
                        # 为bounding_box 匹配标签的到结果
                        score_list = []
                        print(
                            "*****************************************************************************")
                        for i in range(temp_num):
                            dist_list = []
                            for j in range(compare_num):
                                dist = np.sqrt(np.sum(np.square(np.subtract(emb[i, :], compare_emb[j, :]))))
                                dist_list.append(dist)
                            min_value = min(dist_list)
                            if min_value > 0.66 and min_value < 0.78:
                                fin_obj.append('pass')
                                score_list.append(min_value)
                            elif min_value > 0.86:
                                fin_obj.append("pass")
                                score_list.append(min_value)
                            elif min_value > 0.78 and min_value < 0.86:
                                fin_obj.append("unknown")
                                score_list.append(min_value)
                            else:
                                fin_obj.append(all_obj[dist_list.index(min_value)])
                                score_list.append(min_value)
                            print("min_value:", min_value)
                        print(
                            "*****************************************************************************")
                        res = deal_name(fin_obj)
                        print(res)
                        print(len(crop_list), len(path_list))
                        for i in range(len(crop_list)):
                            temp_dict = dict()
                            # cv2.imshow('jpg',image_list[i])
                            # img_path = save_small_pic(crop_image[i])
                            base64_str = img_to_base64(path_list[i])
                            temp_dict["image"] = base64_str
                            temp_dict["userId"] = str(res[i])
                            if temp_dict["userId"] == "pass":
                                continue
                            if temp_dict["userId"] == "unknown":
                                continue
                            temp_dict[
                                'picName'] = '/home/ai/java/image/2019186c85d8_ee96-4951-9cd5_18c97dbbba2e.jpg'
                            score = (1 - score_list[i]) + 0.4
                            temp_dict["similar"] = str(score)
                            post_data = json.dumps(temp_dict)
                            print(post_data)
                            temp_dict["ip"] = ip
                            post_data = json.dumps(temp_dict)

                            response = requests.post(url=POST_URL_NEW, data=post_data,
                                                     headers={"Content-Type": "application/json"})
                            print("Posted finished POST_URL_NEW1", response.json())


                    message = ack_channel.basic_get(queue='ack', auto_ack=True)
                    print(message)
                    if message[0]:
                        image_channel.stop_consuming()
                    ch.basic_ack(delivery_tag=method.delivery_tag)

                image_channel.basic_qos(prefetch_count=1)
                image_channel.basic_consume(on_message_callback=facedecect, queue='images')
                image_channel.start_consuming()


def load_and_align_data(img, image_size, margin):
    """生成load_and_align_data网络，目的是检测人脸框"""
    # print('Creating networks and loading parameters')
    # with tf.Graph().as_default():
    #     gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    #     sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    #     with sess.as_default():
    #         pnet, rnet, onet = detection_face.create_sscnn(sess, None)
    # print('adadadadawdw')
    minsize = 20
    threshold = [0.6, 0.7, 0.88]
    factor = 0.709
    img_size = np.asarray(img.shape)[0:2]
    bounding_boxes, _ = detection_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)

    det = bounding_boxes

    det[:, 0] = np.maximum(det[:, 0] - margin / 2, 0)
    det[:, 1] = np.maximum(det[:, 1] - margin / 2, 0)
    det[:, 2] = np.minimum(det[:, 2] + margin / 2, img_size[1] - 1)
    det[:, 3] = np.minimum(det[:, 3] + margin / 2, img_size[0] - 1)

    det = det.astype(int)
    if len(det)>1:
        return True
    else:
        return False
def load_and_align_data1(img, image_size, margin):
    """生成load_and_align_data网络，目的是检测人脸框"""
    minsize = 20
    threshold = [0.6, 0.7, 0.92]
    factor = 0.709
    img_size = np.asarray(img.shape)[0:2]
    bounding_boxes, _ = detection_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)

    det = bounding_boxes

    det[:, 0] = np.maximum(det[:, 0] - margin / 2, 0)
    det[:, 1] = np.maximum(det[:, 1] - margin / 2, 0)
    det[:, 2] = np.minimum(det[:, 2] + margin / 2, img_size[1] - 1)
    det[:, 3] = np.minimum(det[:, 3] + margin / 2, img_size[0] - 1)

    det = det.astype(int)
    crop = []
    crop_list = []
    path_list = []
    for i in range(len(bounding_boxes)):
        x0 = det[i, 0]
        x1 = det[i, 2]
        w = abs(x1 - x0)
        if w > 80:
            crop_list.append(i)
            temp_crop = img[det[i, 1]:det[i, 3], det[i, 0]:det[i, 2], :]
            path = save_small_pic(temp_crop)
            path_list.append(path)
            aligned = misc.imresize(temp_crop, (image_size, image_size), interp='bilinear')
            prewhitened = OPSnet.prewhiten(aligned)
            crop.append(prewhitened)
    if crop:
        crop_image = np.stack(crop)
        crop_list = crop_list
        flag = True
    else:
        crop_image = crop
        flag = False
    return flag, det, crop_image, crop_list, path_list


def img_to_base64(path):
    """Local picture transform to base64 string"""
    img_im = cv2.imread(path)
    base64_str = base64.b64encode(cv2.imencode('.jpg', img_im)[1]).decode()
    # os.remove(path)
    return base64_str


def deal_name(fin_obj):
    if len(fin_obj):
        res = []
        for i in fin_obj:
            res.append(i.split('.')[0])
        return res
    else:
        return [0, 0, 0, 0, 0, 0]


def save_small_pic(cropped):
    """Save bounding box pic to local,if it is existed,overwrite it"""
    # 20181226 comment resize to judge focus is fit or not
    # cropped = cv2.resize(cropped, interpolation=cv2.INTER_CUBIC)
    img_path = os.path.join(SMALL_DIR, str(uuid.uuid4()) + '.jpg')
    misc.imsave(img_path, cropped)
    return img_path


def first_small_pic(cropped):
    """Save bounding box pic to local,if it is existed,overwrite it"""
    # 20181226 comment resize to judge focus is fit or not
    # cropped = cv2.resize(cropped, interpolation=cv2.INTER_CUBIC)
    img_path = os.path.join(F_SMALL_DIR, str(uuid.uuid4()) + '.jpg')
    # cv2.imwrite(img_path, cropped)
    misc.imsave(img_path, cropped)
    return img_path


print('Creating networks and loading parameters')
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = detection_face.create_sscnn(sess, None)

if __name__ == '__main__':
    while True:
        try:
            main()
        except Exception as e:
            print(e)
            pass
