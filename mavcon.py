from pymavlink import mavutil

# for test
# connection = mavutil.mavlink_connection('udp:0.0.0.0:14445')
# for boat
connection = mavutil.mavlink_connection('tcp:192.168.2.2:5777')