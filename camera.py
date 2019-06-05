from move_zoom_del import Move_Zoom
from get_image import get_image
from multiprocessing import Manager,Pool
import pika


def getImage(ip):
    if "210" in ip:
        movecamera  = Move_Zoom(ip="192.168.2.210")
        for i in range(1,8):
            movecamera.go_to_preset(str(i))
            get_image()
    else:
        movecamera  = Move_Zoom(ip="192.168.2.211")
        for i in range(1,11):
            movecamera.go_to_preset(str(i))
            get_image(CM=211)

def move(ch, method, properties, body):
    ips = str(body, encoding='utf-8')
    ip_list = ips.split(",")
    p = Pool(3)
    for ip in ip_list:
        p.apply_async(getImage,args=(ip,))
    p.close()
    p.join()
    credentials = pika.PlainCredentials('admin', 'swsc2018!')
    ack_connection = pika.BlockingConnection(
        pika.ConnectionParameters(host='192.168.2.93', port=5672, virtual_host='/', credentials=credentials)
    )
    ack_channel = ack_connection.channel()
    ack_channel.basic_publish(exchange='face-exchange', routing_key='ack', body='1')
    ack_connection.close()
    get_image()



def main():
    credentials = pika.PlainCredentials('admin', 'swsc2018!')
    connection_clear = pika.BlockingConnection(pika.ConnectionParameters(host='192.168.2.93',port=5672,virtual_host='/',credentials=credentials))
    channel = connection_clear.channel()
    p = channel.basic_get(queue='camera',auto_ack=True)
    while p[0]:
        p = channel.basic_get(queue='camera', auto_ack=True)
    print('清空队列')
    move = Move_Zoom()
    move.go_to_preset("8",flag=3)
    print('摄像头复位')
    while True:
        credentials = pika.PlainCredentials('admin', 'swsc2018!')
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(host='192.168.2.93', port=5672, virtual_host='/', credentials=credentials,heartbeat=20)
        )
        channel = connection.channel()
        channel.basic_qos(prefetch_count=1)
        channel.basic_consume(on_message_callback=move, queue='camera210',auto_ack=True)
        try:
            channel.start_consuming()
        except Exception as e:
            ack_connection = pika.BlockingConnection(
                pika.ConnectionParameters(host='192.168.2.93', port=5672, virtual_host='/', credentials=credentials)
            )
            ack_channel = ack_connection.channel()
            ack_channel.basic_publish(exchange='face-exchange', routing_key='ack', body='1')
            ack_connection.close()
            get_image()
            continue


if __name__ == '__main__':
    main()
