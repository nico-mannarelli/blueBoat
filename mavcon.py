import os

from pymavlink import mavutil

# Default is the real boat. For SITL testing set MAVCON_URL, e.g.
#   MAVCON_URL=tcp:127.0.0.1:5762 python main.py
# for test
# connection = mavutil.mavlink_connection('udp:0.0.0.0:14445')
# for boat
connection = mavutil.mavlink_connection(
    os.environ.get("MAVCON_URL", "tcp:192.168.2.2:5777"))
