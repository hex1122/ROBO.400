#!/usr/bin/env python3
"""
TB3 controller for Tasks 2.2 + 3
- Supports: goto_xy, goto_pose, follow_path (CLI) and ONLINE goals via topics
- Publishes status logs (rqt_console) and error topics (rqt_plot / rosbag2)
- Simple P-style controller with preemption

Usage examples:
  # Start sim in another terminal first (waffle_pi world)
  # Then run controller (idle): 
  python3 tb3_controller.py

  # Or start with a CLI goal:
  python3 tb3_controller.py goto_xy --x 1.0 --y 0.0

  # Send goals online from another terminal:
  ros2 topic pub -1 /tb3/goal_xy geometry_msgs/Point "{x: 1.0, y: 0.0, z: 0.0}"
  ros2 topic pub -1 /tb3/goal_pose2d geometry_msgs/Pose2D "{x: 0.0, y: 0.0, theta: 1.5708}"
  # Path (nav_msgs/Path) example shown in earlier messages.
"""


import math, sys, argparse
import rclpy
from rclpy.node import Node
# ROS messages we use 
from geometry_msgs.msg import Twist, Point, Pose2D, PoseStamped
from nav_msgs.msg import Odometry, Path

#Error topics for rqt_plot + rosbag:
from std_msgs.msg import Float32, Float32MultiArray, Int32

def yaw_from_quat(q):
    # geometry_msgs/Quaternion -> yaw (Z)
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)

class TB3Controller(Node):
    def __init__(self, args):
        super().__init__('tb3_controller')
        # I/O
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        # runtime interfaces
        self.create_subscription(Point, '/tb3/goal_xy', self.cb_goal_xy, 10)
        self.create_subscription(Pose2D, '/tb3/goal_pose2d', self.cb_goal_pose2d, 10)
        self.create_subscription(Path, '/tb3/path', self.cb_path, 10)

        # Publish errors so they can be plotted/recorded 
        # /tb3/error/distance: Float32 (meters)
        self.err_dist_pub = self.create_publisher(Float32, '/tb3/error/distance', 10)
        # /tb3/error/pose: Float32MultiArray -> [distance (m), yaw_error (rad)]
        self.err_pose_pub = self.create_publisher(Float32MultiArray, '/tb3/error/pose', 10)
        # (optional) publish current waypoint index during follow_path
        self.wp_idx_pub = self.create_publisher(Int32, '/tb3/path_wp_idx', 10)


        self.timer = self.create_timer(0.05, self.loop)  # 20 Hz
        self.pose = None
        self.state = 'idle'
        self.goal = None
        self.path = None
        self.wp_index = 0

        # CLI modes remain supported
        if args.mode == 'goto_xy':
            self.goal = (args.x, args.y, None); self.state = 'goto_xy'
        elif args.mode == 'goto_pose':
            self.goal = (args.x, args.y, args.theta); self.state = 'goto_pose'
        elif args.mode == 'follow_path':
            self.path = [(x, y) for x, y in zip(args.xs, args.ys)]
            if not self.path:
                self.get_logger().error("Empty path"); self.state='idle'
            else:
                self.goal = self.path[0]; self.state = 'follow_path'

        self.get_logger().info(f"Mode: {self.state}")

    #  Subscriptions
    def odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.pose = (p.x, p.y, yaw_from_quat(q))

    def cb_goal_xy(self, msg: Point):
        self.stop()
        self.path = None
        self.goal = (float(msg.x), float(msg.y), None)
        self.wp_index = 0
        self.state = 'goto_xy'
        self.get_logger().info(f"Preempted: new goto_xy -> ({msg.x:.3f}, {msg.y:.3f})")

    def cb_goal_pose2d(self, msg: Pose2D):
        self.stop()
        self.path = None
        self.goal = (float(msg.x), float(msg.y), float(msg.theta))
        self.wp_index = 0
        self.state = 'goto_pose'
        self.get_logger().info(f"Preempted: new goto_pose -> ({msg.x:.3f}, {msg.y:.3f}, {msg.theta:.3f})")

    def cb_path(self, msg: Path):
        self.stop()
        pts = []
        for ps in msg.poses:
            pts.append((float(ps.pose.position.x), float(ps.pose.position.y)))
        if not pts:
            self.get_logger().warn("Received empty Path; staying idle")
            return
        self.path = pts
        self.wp_index = 0
        self.goal = self.path[0]
        self.state = 'follow_path'
        self.get_logger().info(f"Preempted: new path with {len(self.path)} waypoints")

    # Motion loop 
    def stop(self):
        self.cmd_pub.publish(Twist())

    #TASK 3 helper to publish errors every control cycle
    def publish_errors(self, dist: float, yaw_err: float):
        """Publish error signals for evaluation/plots."""
        self.err_dist_pub.publish(Float32(data=float(dist)))
        arr = Float32MultiArray(); arr.data = [float(dist), float(yaw_err)]
        self.err_pose_pub.publish(arr)
   


    def loop(self):
        if self.pose is None or self.state == 'idle':
            return

        x, y, yaw = self.pose
        def go_to_xy(gx, gy, lin_k=0.8, ang_k=2.0, tol=0.05):
            dx, dy = gx - x, gy - y
            rho = math.hypot(dx, dy)
            target_yaw = math.atan2(dy, dx)
            yaw_err = (target_yaw - yaw + math.pi) % (2*math.pi) - math.pi

            # TASK 3 publish distance & yaw error for plotting
            self.publish_errors(rho, yaw_err)

            if rho < tol:
                return True, Twist()
            cmd = Twist()
            cmd.linear.x  = max(min(lin_k * rho, 0.25), -0.25)
            cmd.angular.z = max(min(ang_k * yaw_err, 1.5), -1.5)
            return False, cmd

        if self.state == 'goto_xy':
            done, cmd = go_to_xy(self.goal[0], self.goal[1])
            if done:
                self.get_logger().info("Reached XY goal.")
                self.stop(); self.state = 'idle'
            else:
                self.cmd_pub.publish(cmd)

        elif self.state == 'goto_pose':
            gx, gy, gth = self.goal
            done, cmd = go_to_xy(gx, gy, tol=0.06)
            if not done:
                self.cmd_pub.publish(cmd)
            else:
                yaw_err = (gth - yaw + math.pi) % (2*math.pi) - math.pi

                 #  TASK 3 still publish yaw error during final align
                self.publish_errors(0.0, yaw_err)
        
                if abs(yaw_err) < 0.03:
                    self.get_logger().info("Reached full pose (x,y,theta).")
                    self.stop(); self.state = 'idle'
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


                #TASK 3 publish waypoint index so it’s visible in plots
                self.wp_idx_pub.publish(Int32(data=self.wp_index))


                if self.wp_index >= len(self.path):
                    self.get_logger().info("Finished path.")
                    self.stop(); self.state = 'idle'
                else:
                    self.goal = self.path[self.wp_index]
                    self.get_logger().info(f"Next waypoint {self.wp_index}/{len(self.path)}: {self.goal}")

def parse_args(argv):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='mode', required=False)

    s1 = sub.add_parser('goto_xy')
    s1.add_argument('--x', type=float, required=True)
    s1.add_argument('--y', type=float, required=True)

    s2 = sub.add_parser('goto_pose')
    s2.add_argument('--x', type=float, required=True)
    s2.add_argument('--y', type=float, required=True)
    s2.add_argument('--theta', type=float, required=True, help='heading in radians')

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
