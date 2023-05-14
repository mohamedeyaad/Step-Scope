#!/usr/bin/env python3

from std_msgs.msg import String
import rospy
    
def callback(msg):
 Keyword = "Successful"
 if Keyword in msg.data:
 	rospy.loginfo("True")
 	pub = rospy.Publisher('status', String, queue_size=10)
 	pub.publish("Successful")

if __name__ == '__main__':
    try:
     rospy.init_node('status_pub', anonymous=True)
     rospy.Subscriber("comm", String, callback) 
     rate = rospy.Rate(60) # 10hz
     rospy.spin()
    except rospy.ROSInterruptException:
        pass
