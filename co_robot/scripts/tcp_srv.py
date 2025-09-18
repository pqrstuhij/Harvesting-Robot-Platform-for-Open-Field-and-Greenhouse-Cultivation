#!/usr/bin/env python3
import rospy
from co_robot.srv import vision_robot
import socket
import sys
import moveit_commander
from moveit_msgs.msg import MoveItErrorCodes

SERVER_IP = "192.168.1.207"
SERVER_PORT = 9000
MAX_MESSAGE_SIZE = 100

class RobotController:
    def __init__(self):
        self.sock = None
        self.done_count = 0
        self.robot = None
        self.group = None
        self.valid_states = []
        self.initialize()

    def initialize(self):
        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = moveit_commander.RobotCommander()
        self.group = moveit_commander.MoveGroupCommander("robo")
        self.group.set_planning_time(10)
        self.group.set_num_planning_attempts(3)
        self.valid_states = self.group.get_named_targets()
        rospy.loginfo(f"Valid states: {self.valid_states}")

        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((SERVER_IP, SERVER_PORT))
            rospy.loginfo("TCP Connected")
        except Exception as e:
            rospy.logerr(f"Connection failed: {e}")
            sys.exit(1)

    def set_group_state(self, state_name):
        if state_name not in self.valid_states:
            rospy.logwarn(f"Invalid state: {state_name}")
            return False

        try:
            self.group.set_named_target(state_name)
            success, plan, _, _ = self.group.plan()
            if success and plan.joint_trajectory.points:
                self.group.execute(plan, wait=True)
                rospy.set_param("/current_state", state_name)
                rospy.loginfo(f"State changed to {state_name}")
                return True
            return False
        except Exception as e:
            rospy.logerr(f"Motion error: {e}")
            return False

    def handle_server_response(self, expected):
        try:
            response = self.sock.recv(MAX_MESSAGE_SIZE).decode().strip()
            rospy.loginfo(f"Server response: {response}")

            if response == "done":
                self.done_count += 1
                states = ["visd3", "vist3"]
                if self.done_count <= 2:
                    self.set_group_state(states[self.done_count - 1])
                if self.done_count == 2:
                    self.done_count = 0
            return response == expected
        except Exception as e:
            rospy.logerr(f"Response error: {e}")
            return False

    def robot_move(self, req):
        try:
            #if req.x > 280:
            #    rospy.loginfo("X > 300: return")
            #    return 100

            #adjusted_x = req.x - 70 if req.x >= 200 else req.x
            val_3 = str(req.p) if self.done_count == 0 else str(self.done_count)

            #if val_3 == "0" and (req.x >= 255 or req.x < 253):
            #    req.x = 270  #254.553

            #if val_3 == "1" and (req.x >= 186 or req.x < 200):
            #    req.x = 200 #185.459\

            #if val_3 == "0" and (req.x >= 270 or req.x < 270):
            #    req.x = 270  #254.553

            #if val_3 == "1" and (req.x >= 200 or req.x < 200):
            #    req.x = 200 #185.459


            message = f"{req.e},{req.u},{val_3},{req.x},{req.y},{req.z},{req.dgree}"
            # 예: 9,1,0,150,210,350,45

            if self.handle_server_response("ok"):
                self.sock.sendall(message.encode())
                print(message)

                if req.u == 1:
                    if self.handle_server_response("finish"):
                        return 100
                elif req.u == 0:
                    if self.handle_server_response("done"):
                        return 100

            return 500
        except Exception as e:
            rospy.logerr(f"Service error: {e}")
            return 500

def main():
    controller = RobotController()
    rospy.init_node("tcp_srv")
    rospy.Service("service", vision_robot, controller.robot_move)
    rospy.loginfo("Service ready")

    try:
        rospy.spin()
    finally:
        moveit_commander.roscpp_shutdown()
        if controller.sock:
            controller.sock.close()
        rospy.loginfo("Resources cleaned up")

if __name__ == '__main__':
    main()

