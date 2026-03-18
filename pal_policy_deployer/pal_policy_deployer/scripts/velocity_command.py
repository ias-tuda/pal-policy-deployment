#!/usr/bin/env python
# license removed for brevity
import rospy
from geometry_msgs.msg import Twist



def command_node():
    pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
    cmd = Twist()

    def set_command(linear = [0.0,0.0,0.0], angular = [0.0,0.0,0.0], time = 1.0):
        cmd.linear.x = linear[0]
        cmd.linear.y = linear[1]
        cmd.linear.z = linear[2]

        cmd.angular.x = angular[0]
        cmd.angular.y = angular[1]
        cmd.angular.z = angular[2]

        timeout = rospy.Time.now() + rospy.Duration(time)
        while rospy.Time.now() < timeout:
            pub.publish(cmd)
            rospy.loginfo("Published : {}".format(cmd))
            rospy.sleep(0.1)

    lin = [0.0, 0.0, 0.0]
    ang = [0.0, 0.0, 0.0]

    rospy.init_node('command_node', anonymous=True)
    rospy.sleep(0.1)
    
    lin[0] = 0.7
    set_command(lin, ang, 5.0)

    lin[0] = 0.0
    set_command(lin, ang, 1.0)

    lin[0] = -0.7
    set_command(lin, ang, 5.0)

    lin[0] = 0.0
    set_command(lin, ang, 1.0)


if __name__ == '__main__':
    try:
        command_node()
    except rospy.ROSInterruptException:
        pass