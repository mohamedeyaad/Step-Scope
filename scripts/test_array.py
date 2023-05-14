#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float32MultiArray , Int32 , String
import random
import numpy as np

rospy.init_node('random_number_publisher')

pub = rospy.Publisher('random_numbers', Float32MultiArray, queue_size=10)
pub1 = rospy.Publisher('imu_data', Int32, queue_size=10)
pub2 = rospy.Publisher('imu_data2', Int32, queue_size=10)
pub3 = rospy.Publisher('status_pub', String, queue_size=10)

rate = rospy.Rate(4) # 10 Hz

while not rospy.is_shutdown():
    array = Float32MultiArray()
    array.data = np.random.randint(1, 1025, size=5)
    angle1_random = np.random.randint(1, 90)
    angle2_random = np.random.randint(1, 90)
    #array.data = [random.uniform(0, 1) for i in range(10)]
    pub.publish(array)
    pub1.publish(angle1_random)
    pub2.publish(angle2_random)
    pub3.publish("Successful")
    #rospy.loginfo(array)
    rate.sleep()

