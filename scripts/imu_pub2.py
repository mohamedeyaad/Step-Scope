#!/usr/bin/env python3

from std_msgs.msg import String
from std_msgs.msg import Int32
import rospy
import re

pub = rospy.Publisher('imu_data2', Int32, queue_size=10)
 
def imu_pattern(msg):
	#string = "imu 2 value = 90"

	# Define the pattern to match the integer value
	pattern = r"imu 2 value = (-?\d+)"

	# Use regex to find the match and extract the integer value
	match = re.search(pattern, msg)

	if match:
		value = int(match.group(1))
		print(f"The integer value is {value}")
	else:
		print("No match found")
		
	return value	
	
def callback(msg):
 imu_reading = imu_pattern(msg.data)
 rospy.loginfo(imu_reading)

 pub.publish(imu_reading)


if __name__ == '__main__':
    try:
     rospy.init_node('imu_data2', anonymous=True)
     rospy.Subscriber("comm", String, callback) 
     rate = rospy.Rate(10) # 10hz
     rospy.spin()
    except rospy.ROSInterruptException:
        pass
