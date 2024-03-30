import rospy

from mobile_manipulation_central.ros_interface import MapInterface
from mmseq_control.map import SDF2D

def main():

    map_ros_interface = MapInterface(topic_name="/pocd_slam_node/occupied_ef_nodes")
    sdf = SDF2D()
    rate = rospy.Rate(100)

    while not map_ros_interface.ready() and not rospy.is_shutdown():
        rate.sleep()

    while not rospy.is_shutdown():
        is_map_updated, map = map_ros_interface.get_map()
        if is_map_updated:
            sdf.update_map(map)
            print(f"{rospy.get_time()} map updated")

            if sdf.valid:
                sdf.vis(map)

        rate.sleep()

if __name__ == "__main__":
    rospy.init_node("map_tester")
    main()