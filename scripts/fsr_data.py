#!/usr/bin/env python3

from std_msgs.msg import Float32MultiArray
from std_msgs.msg import String
import rospy
import re
import ast
import numpy as np

pub = rospy.Publisher('fsr_data', Float32MultiArray, queue_size=10)

def fsr_data(msg):
	#string = "The FSR readings are: FSR readings = [10,12,13,4,15]"
	# Define the pattern to match the list of FSR readings
	pattern = r"FSR readings = \[(.*?)\]"
	# Use regex to find the match and extract the substring
	match = re.search(pattern, msg)
	if match:
		readings_str = match.group(1)
		readings_list = ast.literal_eval(readings_str)
		print(f"The FSR readings are: {readings_list}")
	else:
		print("No match found")    
	return readings_list

def callback(msg):
 try:
 	value = fsr_data(msg.data)

 #value = msg
 	array = Float32MultiArray()
 	array.data = value
 	rospy.loginfo(array)
 	pub.publish(array)
 except:
 	pass

if __name__ == '__main__':
    try:
     rospy.init_node('fsr_data', anonymous=True)
     #rospy.Subscriber("random_numbers", Float32MultiArray, callback) 
     rospy.Subscriber("comm", String, callback) 
     rate = rospy.Rate(10) # 10hz
     rospy.spin()
    except rospy.ROSInterruptException:
        pass
