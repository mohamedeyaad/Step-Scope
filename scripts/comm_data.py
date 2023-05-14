#!/usr/bin/env python3

from std_msgs.msg import String
import socket
import rospy

my_socket = socket.socket()
port = 5055
ip = "172.20.10.2"
my_socket.connect((ip,port))
            
if __name__ == '__main__':
    try:
     #Node Initialization
     rospy.init_node('comm_data', anonymous=True)
     pub = rospy.Publisher('comm', String, queue_size=10)
     rate = rospy.Rate(10) # 10hz
     while not rospy.is_shutdown():
      msg = (my_socket.recv(1024).decode()) 
      #('utf-8') for string
      rospy.loginfo(msg)
      pub.publish(msg)
      #pub.publish("11")
      rate.sleep()
    except rospy.ROSInterruptException:
     pass
