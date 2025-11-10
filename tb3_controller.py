#!/usr/bin/env python3
import math, sys, argparse
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from tf_transformations import euler_from_quaternion

def yaw_from_quat(q):
    return euler_from_quaternion([q.x, q.y, q.z, q.w])[2]

class TB3Controller(Node):
    def __init__(self, args):
        super().__init__('tb3_controller')
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.timer = self.create_timer(0.05, self.loop)  # 20 Hz
        self.pose = None
        self.args = args
        self.state = 'idle'
        self.wp_index = 0

        if args.mode == 'goto_xy':
            self.goal = (args.x, args.y, None)
            self.state = 'goto_xy'
        elif args.mode == 'goto_pose':
            self.goal = (args.x, args.y, args.theta)
            self.state = 'goto_pose'
        elif args.mode == 'follow_path':
            self.path = [(x, y) for x, y in zip(args.xs, args.ys)]
            self.goal = self.path[0]
            self.state = 'follow_path'

        self.get_logger().info(f"Mode: {self.state}. Goal(s) set.")

    def odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.pose = (p.x, p.y, yaw_from_quat(q))

    def stop(self):
        self.cmd_pub.publish(Twist())

    def loop(self):
        if self.pose is None or self.state == 'idle':
            return

        x, y, yaw = self.pose
        cmd = Twist()

        def go_to_xy(gx, gy, lin_k=0.8, ang_k=2.0, tol=0.05):
            dx, dy = gx - x, gy - y
            rho = math.hypot(dx, dy)
            target_yaw = math.atan2(dy, dx)
            yaw_err = (target_yaw - yaw + math.pi) % (2*math.pi) - math.pi
            if rho < tol:
                return True, Twist()
            cmd = Twist()
            cmd.linear.x = max(min(lin_k * rho, 0.25), -0.25)
            cmd.angular.z = max(min(ang_k * yaw_err, 1.5), -1.5)
            return False, cmd

        if self.state == 'goto_xy':
            done, cmd = go_to_xy(self.goal[0], self.goal[1])
            if done:
                self.get_logger().info("Reached XY goal.")
                self.stop()
                self.state = 'idle'
            else:
                self.cmd_pub.publish(cmd)

        elif self.state == 'goto_pose':
            gx, gy, gth = self.goal
            # Stage 1: go to point
            done, cmd = go_to_xy(gx, gy, tol=0.06)
            if not done:
                self.cmd_pub.publish(cmd)
            else:
                # Stage 2: align heading
                yaw_err = (gth - yaw + math.pi) % (2*math.pi) - math.pi
                if abs(yaw_err) < 0.03:
                    self.get_logger().info("Reached full pose (x,y,theta).")
                    self.stop()
                    self.state = 'idle'
                else:
                    turn = Twist()
                    turn.angular.z = max(min(2.0 * yaw_err, 1.2), -1.2)
                    self.cmd_pub.publish(turn)

        elif self.state == 'follow_path':
            gx, gy = self.goal
            done, cmd = go_to_xy(gx, gy, tol=0.07)
            if not done:
                self.cmd_pub.publish(cmd)
            else:
                self.wp_index += 1
                if self.wp_index >= len(self.path):
                    self.get_logger().info("Finished path.")
                    self.stop()
                    self.state = 'idle'
                else:
                    self.goal = self.path[self.wp_index]
                    self.get_logger().info(f"Next waypoint {self.wp_index}/{len(self.path)}: {self.goal}")

def parse_args(argv):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='mode', required=True)

    s1 = sub.add_parser('goto_xy')
    s1.add_argument('--x', type=float, required=True)
    s1.add_argument('--y', type=float, required=True)

    s2 = sub.add_parser('goto_pose')
    s2.add_argument('--x', type=float, required=True)
    s2.add_argument('--y', type=float, required=True)
    s2.add_argument('--theta', type=float, required=True,
                    help='heading in radians')

    s3 = sub.add_parser('follow_path')
    s3.add_argument('--xs', type=float, nargs='+', required=True)
    s3.add_argument('--ys', type=float, nargs='+', required=True)

    return ap.parse_args(argv)

def main():
    args = parse_args(sys.argv[1:])
    rclpy.init()
    node = TB3Controller(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Stopping TB3 controller.")
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
