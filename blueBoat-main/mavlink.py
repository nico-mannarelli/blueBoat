import os
os.environ['MAVLINK20'] = '1'

from pymavlink import mavutil
import time
import math
from shapely.geometry import Point, Polygon
from pymavlink import mavutil, mavwp
import pymavlink.dialects.v20.all as dialect


from mavcon import connection

# # for test
# # connection = mavutil.mavlink_connection('udp:0.0.0.0:14445')
# # for boat
# connection = mavutil.mavlink_connection('tcp:192.168.2.2:5777')

connection.wait_heartbeat()
print("Heartbeat from system (system %u component %u)" % (connection.target_system, connection.target_component))

boot_time = time.time()

# FENCE

#west
# fence_list = ([38.143789, -76.524991, dialect.MAV_CMD_NAV_FENCE_POLYGON_VERTEX_EXCLUSION], # top left
#               [38.143895, -76.524179, dialect.MAV_CMD_NAV_FENCE_POLYGON_VERTEX_EXCLUSION], 
#               [38.143750, -76.523869, dialect.MAV_CMD_NAV_FENCE_POLYGON_VERTEX_EXCLUSION],
#               [38.141055, -76.526649, dialect.MAV_CMD_NAV_FENCE_POLYGON_VERTEX_EXCLUSION],  # bottom right
#               [38.141553, -76.527268, dialect.MAV_CMD_NAV_FENCE_POLYGON_VERTEX_EXCLUSION]
#               )

#east
# fence_list = ([38.143660, -76.523602, dialect.MAV_CMD_NAV_FENCE_POLYGON_VERTEX_EXCLUSION],
#                [38.143441, -76.523113, dialect.MAV_CMD_NAV_FENCE_POLYGON_VERTEX_EXCLUSION],
#                  [38.140821, -76.525738, dialect.MAV_CMD_NAV_FENCE_POLYGON_VERTEX_EXCLUSION], 
#                  [38.140964, -76.526389, dialect.MAV_CMD_NAV_FENCE_POLYGON_VERTEX_EXCLUSION])

# geofence
fence_list = ([38.1439, -76.525077, dialect.MAV_CMD_NAV_FENCE_POLYGON_VERTEX_EXCLUSION], # top left
              [38.143969, -76.524120, dialect.MAV_CMD_NAV_FENCE_POLYGON_VERTEX_EXCLUSION], 
              [38.143830, -76.523709, dialect.MAV_CMD_NAV_FENCE_POLYGON_VERTEX_EXCLUSION],
               [38.143567, -76.523026, dialect.MAV_CMD_NAV_FENCE_POLYGON_VERTEX_EXCLUSION],
               [38.143276, -76.523030, dialect.MAV_CMD_NAV_FENCE_POLYGON_VERTEX_EXCLUSION],
                 [38.140553, -76.525757, dialect.MAV_CMD_NAV_FENCE_POLYGON_VERTEX_EXCLUSION], 
                 [38.140753, -76.526599, dialect.MAV_CMD_NAV_FENCE_POLYGON_VERTEX_EXCLUSION],
              [38.140936, -76.526892, dialect.MAV_CMD_NAV_FENCE_POLYGON_VERTEX_EXCLUSION],  # bottom right
              [38.141681, -76.527589, dialect.MAV_CMD_NAV_FENCE_POLYGON_VERTEX_EXCLUSION]
              )

#both zones
# fence_list = ([38.143789, -76.524991, dialect.MAV_CMD_NAV_FENCE_POLYGON_VERTEX_EXCLUSION], # top left
#               [38.143895, -76.524179, dialect.MAV_CMD_NAV_FENCE_POLYGON_VERTEX_EXCLUSION], 
#               [38.143750, -76.523869, dialect.MAV_CMD_NAV_FENCE_POLYGON_VERTEX_EXCLUSION],
#               [38.143660, -76.523602, dialect.MAV_CMD_NAV_FENCE_POLYGON_VERTEX_EXCLUSION], # east
#                [38.143441, -76.523113, dialect.MAV_CMD_NAV_FENCE_POLYGON_VERTEX_EXCLUSION],
#                  [38.140821, -76.525738, dialect.MAV_CMD_NAV_FENCE_POLYGON_VERTEX_EXCLUSION], 
#                  [38.140964, -76.526389, dialect.MAV_CMD_NAV_FENCE_POLYGON_VERTEX_EXCLUSION],
#               [38.141055, -76.526649, dialect.MAV_CMD_NAV_FENCE_POLYGON_VERTEX_EXCLUSION],  # bottom right
#               [38.141553, -76.527268, dialect.MAV_CMD_NAV_FENCE_POLYGON_VERTEX_EXCLUSION]
#               )


class Fence():
    
    def add_polygon(self, polygon, command):
        fence_count = 0
        message = dialect.MAVLink_param_request_read_message(target_system=connection.target_system,
                                                         target_component=connection.target_component,
                                                         param_id=FENCE_TOTAL,
                                                         param_index=PARAM_INDEX)
        connection.mav.send(message)
        while True:
            message = connection.recv_match(type=dialect.MAVLink_param_value_message.msgname,
                                             blocking=True)
            # convert the message to dictionary
            message = message.to_dict()
            if message["param_id"] == "FENCE_TOTAL":
                print("FENCE_TOTAL parameter original:", message)
                fence_count = int(message["param_value"])
                break
            
        message = dialect.MAVLink_mission_count_message(target_system=connection.target_system,
                                                target_component=connection.target_component,
                                                count=len(polygon),
                                                mission_type=dialect.MAV_MISSION_TYPE_MISSION)
        connection.mav.send(message)
        message = connection.recv_match(blocking=True)
        message = message.to_dict()
        print(message)
            
        count = len(polygon)
        for i in range(0, count):
            coord = polygon[i]
            lat = coord[0]
            lon = coord[1]
            
            m = mavutil.mavlink.MAVLink_mission_item_int_message(
                    connection.target_system,
                    connection.target_component,
                    0,    # seq
                    mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,    # frame
                    command,    # command
                    0,    # current
                    0,    # autocontinue
                    count, # param1,
                    0.0,  # param2,
                    0.0,  # param3
                    0.0,  # param4
                    int(lat*1e7),  # x (latitude)
                    int(lon*1e7),  # y (longitude)
                    0,                     # z (altitude)
                    mavutil.mavlink.MAV_MISSION_TYPE_FENCE,
                )
            connection.mav.send(m)
        connection.param_set_send('FENCE_TOTAL', count, 3)
        
    def add_polygon_mission(self, target_locations):
        message = dialect.MAVLink_mission_count_message(target_system=connection.target_system,
                                                target_component=connection.target_component,
                                                count=len(target_locations),
                                                mission_type=dialect.MAV_MISSION_TYPE_FENCE)

        # send mission count message to the vehicle
        connection.mav.send(message)

        message = connection.recv_match(blocking=True)
        print(message)
        
        # this loop will run until receive a valid MISSION_ACK message
        while True:
        
            # catch a message
            message = connection.recv_match(blocking=True)
        
            # convert this message to dictionary
            message = message.to_dict()
        
            # check this message is MISSION_REQUEST
            if message["mavpackettype"] == dialect.MAVLink_mission_request_message.msgname:
        
                # check this request is for mission items
                if message["mission_type"] == dialect.MAV_MISSION_TYPE_FENCE:
        
                    # get the sequence number of requested mission item
                    seq = message["seq"]
                    
                    print(message)

                    # create mission item int message that contains the home location (0th mission item)
                    coord = target_locations[seq]
                    lat = coord[0]
                    lon = coord[1]
                    message = dialect.MAVLink_mission_item_int_message(target_system=connection.target_system,
                                                                       target_component=connection.target_component,
                                                                       seq=seq,
                                                                       frame=dialect.MAV_FRAME_GLOBAL,
                                                                       command=coord[2],
                                                                       current=0,
                                                                       autocontinue=0,
                                                                       param1=len(target_locations),
                                                                       param2=0,
                                                                       param3=0,
                                                                       param4=0,
                                                                       x=int(lat*1e7),  # x (latitude)
                                                                       y=int(lon*1e7),  # y (longitude)
                                                                       z=0,
                                                                       mission_type=dialect.MAV_MISSION_TYPE_FENCE)
        
                    # send the mission item int message to the vehicle
                    connection.mav.send(message)
        
            # check this message is MISSION_ACK
            elif message["mavpackettype"] == dialect.MAVLink_mission_ack_message.msgname:
        
                # check this acknowledgement is for mission and it is accepted
                if message["mission_type"] == dialect.MAV_MISSION_TYPE_FENCE and \
                        message["type"] == dialect.MAV_MISSION_ACCEPTED:
                    # break the loop since the upload is successful
                    print("Mission upload is successful")
                    break
                elif message["mission_type"] == dialect.MAV_MISSION_TYPE_FENCE:
                    print (message)
                    
# set geofence
fence = Fence()
fence.add_polygon_mission(fence_list)

# stream GLOBAL_POSITION_INT at 1 Hz
message = connection.mav.command_long_encode(
    connection.target_system,
    connection.target_component,
    mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
    0,
    mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT,
    1000000,
    0, 0, 0, 0, 0
)
connection.mav.send(message)


# ── Helpers ───────────────────────────────────────────────────────────────────

def await_ack(command, timeout=3):
    start = time.time()
    while time.time() - start < timeout:
        ack = connection.recv_match(type='COMMAND_ACK', blocking=False)
        if ack and ack.command == command:
            result = ack.result
            label = mavutil.mavlink.enums['MAV_RESULT'][result].name
            print(f"  ACK {command}: {label}")
            return result == mavutil.mavlink.MAV_RESULT_ACCEPTED
    print(f"  ACK timeout for command {command}")
    return False


def get_current_position(timeout=5):
    msg = connection.recv_match(type='GLOBAL_POSITION_INT',
                                blocking=True, timeout=timeout)
    if msg:
        return msg.lat / 1e7, msg.lon / 1e7, msg.alt / 1e3
    return None


# ── Safety Zone ───────────────────────────────────────────────────────────────
SAFETY_ZONE = {
    'lat1': min(38.143740, 38.141398),  # min lat
    'lon1': min(-76.52376, -76.528101), # min lon
    'alt1': 0,
    'lat2': max(38.143740, 38.141398),  # max lat
    'lon2': max(-76.52376, -76.528101), # max lon
    'alt2': 0
}


def in_safety_zone(lat, lon):
    return (SAFETY_ZONE['lat1'] <= lat <= SAFETY_ZONE['lat2'] and
            SAFETY_ZONE['lon1'] <= lon <= SAFETY_ZONE['lon2'])

def set_safety_zone():
    connection.mav.safety_set_allowed_area_send(
        connection.target_system,
        connection.target_component,
        frame=mavutil.mavlink.MAV_FRAME_GLOBAL_INT,
        p1x=SAFETY_ZONE['lat1'], p1y=SAFETY_ZONE['lon1'], p1z=SAFETY_ZONE['alt1'],
        p2x=SAFETY_ZONE['lat2'], p2y=SAFETY_ZONE['lon2'], p2z=SAFETY_ZONE['alt2']
    )
    print("Safety zone set.")

points = [(38.143789, -76.524991), 
      (38.143895, -76.524179), 
        (38.143750, -76.523869),
        (38.141055, -76.526649),
        (38.141553, -76.527268)]
fence = Polygon(points)
#plt.plot(fence.exterior.xy)

points2 = [(38.143660, -76.523602), (38.143441, -76.523113), (38.140821, -76.525738), (38.140964, -76.526389)]
fence2 = Polygon(points2)

# ── Commands ──────────────────────────────────────────────────────────────────

def set_target_pos(lat, lon, alt=0.0):
    connection.mav.set_position_target_global_int_send(
        int(1e3 * (time.time() - boot_time)),
        connection.target_system, connection.target_component,
        coordinate_frame=mavutil.mavlink.MAV_FRAME_GLOBAL_INT,
        type_mask=(
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_VX_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_VY_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_VZ_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
        ),
        lat_int=int(lat * 1e7),
        lon_int=int(lon * 1e7),
        alt=alt,
        vx=0, vy=0, vz=0,
        afx=0, afy=0, afz=0,
        yaw=0, yaw_rate=0
    )


def set_orbit(lat, lon):
    message = connection.mav.command_long_encode(
        connection.target_system,
        connection.target_component,
        mavutil.mavlink.MAV_CMD_DO_ORBIT,
        0,
        10.0,               # param1: radius (m)
        math.nan,             # param2: velocity (NaN = default)
        5,                    # param3: yaw behaviour (unchanged)
        2 * math.pi, # param4: radians (1 orbit = 2π)
        lat,                  # param5: latitude
        lon,                  # param6: longitude
        math.nan              # param7: altitude (NaN = current)
    )
    connection.mav.send(message)
    await_ack(mavutil.mavlink.MAV_CMD_DO_ORBIT)


def set_return():
    message = connection.mav.command_long_encode(
        connection.target_system,
        connection.target_component,
        mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH,
        0,
        0, 0, 0, 0, 0, 0, 0
    )
    connection.mav.send(message)
    await_ack(mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH)


def set_new_mission():
    message = connection.mav.command_long_encode(
        connection.target_system,
        connection.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_MISSION_CURRENT,
        0,
        -1, 0, 0, 0, 0, 0, 0 #second zero allows you to change mission in middle of auto
        # will stop current step and complete newly added waypoints: if 2 added at beginning, skips original 2 but will continue on from original 3
    )
    connection.mav.send(message)
    await_ack(mavutil.mavlink.MAV_CMD_DO_SET_MISSION_CURRENT)

def edit_current_mission():
    message = connection.mav.command_long_encode(
        connection.target_system,
        connection.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_MISSION_CURRENT,
        0,
        -1, 1, 0, 0, 0, 0, 0 #second zero allows you to change mission in middle of auto
        # will stop current step and complete newly added waypoints: if 2 added at beginning, skips original 2 but will continue on from original 3
    )
    connection.mav.send(message)
    await_ack(mavutil.mavlink.MAV_CMD_DO_SET_MISSION_CURRENT)


def disarm():
    connection.mav.command_long_send(
        connection.target_system,
        connection.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0,
        0, 0, 0, 0, 0, 0, 0
    )
    await_ack(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM)


# ── Mission Upload ────────────────────────────────────────────────────────────
def upload_mission(waypoints, crosspoints, returnpoints):
    # check all waypoints are inside the safety zone
    safe_waypoints = [(38.143700, -76.524090)] 
    for lat, lon in waypoints:
        point = Point(lat, lon)
        if fence.contains(point): #in_safety_zone(lat, lon) and edit to be wider zone
            safe_waypoints.append((lat, lon))
        # else:
        #     print("Outside safety zone: %u, %u" % (lat, lon))
    for lat, lon in crosspoints:
        safe_waypoints.append((lat, lon))
    for lat, lon in waypoints:
        point = Point(lat, lon)
        if fence2.contains(point): #in_safety_zone(lat, lon) and edit to be wider zone
            safe_waypoints.append((lat, lon))
    for lat, lon in returnpoints:
        safe_waypoints.append((lat, lon))

    # # clear existing mission
    connection.mav.mission_clear_all_send(
        connection.target_system,
        connection.target_component,
        mavutil.mavlink.MAV_MISSION_TYPE_MISSION
    )
    time.sleep(0.5)
    # Drain the MISSION_ACK the clear produces. Without this, the final ack
    # check below reads THIS stale ack instead of the upload's, so it printed
    # "accepted" even when the upload itself was rejected.
    while connection.recv_match(type='MISSION_ACK', blocking=False):
        pass

    # home position (item 0) + waypoints
    total = len(safe_waypoints)

    print("MISSION COUNT =", total)
    print("SAFE WAYPOINTS =", safe_waypoints)
    connection.mav.mission_count_send(
        connection.target_system,
        connection.target_component,
        total,
        mavutil.mavlink.MAV_MISSION_TYPE_MISSION
    )

    def send_item(seq, lat, lon, alt, current=0):
        req = connection.recv_match(
            type=['MISSION_REQUEST', 'MISSION_REQUEST_INT'],
            blocking=True, timeout=5
        )
        if req is None:
            print(f"Timeout waiting for request")
            return False
        
        print(f"  Got request for item {req.seq}")
        seq = req.seq  #took out for sim 
        connection.mav.mission_item_int_send(
            connection.target_system,
            connection.target_component,
            seq,
            mavutil.mavlink.MAV_FRAME_GLOBAL_INT,
            mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
            current, 1,        # current, autocontinue
            5, 1.0, 0,        # hold time !!! , acceptance radius, pass radius
            math.nan,          # yaw
            int(lat * 1e7),
            int(lon * 1e7),
            alt
        )
        print(f"  Sent item {seq}: ({lat}, {lon}, {alt})")
        return True

    # # item 0: home position (same as first waypoint)
    # send_item(waypoints[0][0], waypoints[0][1], waypoints[0][2])

    # items 1+: waypoints. Each item already carries a 5s hold (param1 in
    # send_item), so there is NO NAV_DELAY here. Sending a COMMAND_LONG
    # mid-handshake violates the mission-upload protocol and made ArduPilot
    # SITL silently drop the whole mission (ack'd but stored the old one).
    count = 0
    for lat, lon in safe_waypoints:
        send_item(count, lat, lon, 0.0)
        count += 1

    ack = connection.recv_match(type='MISSION_ACK', blocking=True, timeout=5)
    ack_name = (mavutil.mavlink.enums['MAV_MISSION_RESULT'][ack.type].name
                if ack else "no ack")
    print(f"Mission upload ack: {ack_name}")

    # Ground truth: read the mission back and confirm the vehicle actually
    # stored what we sent. The ack alone has lied before, so success depends
    # on the read-back count, not on the ack.
    connection.mav.mission_request_list_send(
        connection.target_system,
        connection.target_component,
        mavutil.mavlink.MAV_MISSION_TYPE_MISSION
    )
    stored = connection.recv_match(type='MISSION_COUNT', blocking=True, timeout=5)
    stored_count = stored.count if stored else None
    if stored_count == total:
        print(f"Mission upload verified: {stored_count} items on vehicle.")
        return True
    print(f"Mission upload FAILED: sent {total}, vehicle reports "
          f"{stored_count} (ack={ack_name}). "
          "Make sure QGC isn't re-pushing its own plan.")
    return False



# ── Main ──────────────────────────────────────────────────────────────────────

# Grid Waypoints
# WAYPOINTS = []
# lat_min = 38.1200
# lat_max = 38.1700
# lon_min = -76.5600
# lon_max = -76.5000
# step = 0.00025

# lat = lat_max
# while lat >= lat_min:
#     lon = lon_min
#     while lon <= lon_max:
#         WAYPOINTS.append((round(lat, 7), round(lon, 7)))
#         lon += step
#     lat -= step 

# WAYPOINTS = [
#     (38.143700, -76.524090), #ignores
#     (38.143700, -76.524090),
#     (38.143168, -76.524619),
#     (38.141632, -76.526641),
#     (38.141322, -76.526641),
#     (38.141905, -76.525008), # east                            
#     (38.142793, -76.524023)
#  ]


from contacts_coords import coords
WAYPOINTS = [(lat, lon) for lat,lon, *_ in coords]

CROSSPOINTS = [(38.141321, -76.526434), (38.141149, -76.526146)]

RETURNPOINTS = [(38.143542, -76.523686), (38.143670, -76.523895), (38.143749, -76.524022)]

# set_safety_zone()

# loiters for 10 secs before starting new mission
# upload_mission(WAYPOINTS) # starts new mission to new waypoints and ends 

#for manual upload 
# will start right away if already in auto mode
# if upload_mission(WAYPOINTS, CROSSPOINTS, RETURNPOINTS):  #msg.mission_state == 5 and
#         set_new_mission()

# # for automatic upload, remove the switch to manual & auto to not auto start
while True:
    msg = connection.recv_match(
            type=['MISSION_CURRENT'],
            blocking=True, timeout=5
        )
    if not msg:
        continue
    
    #check if mission complete and upload mission ready
    #if msg.mission_state == 5 and upload_mission(WAYPOINTS, CROSSPOINTS, RETURNPOINTS): 
    if upload_mission(WAYPOINTS, CROSSPOINTS, RETURNPOINTS):  
        set_new_mission()

        print("Starting new mission")

        # Manual -> Auto bounce is what actually starts an already-loaded
        # mission on ArduRover (custom mode 0 = MANUAL, 10 = AUTO). BOTH
        # halves must fire; the AUTO half used to be commented out, so it
        # printed "auto" but the boat never left MANUAL and never started.
        connection.mav.command_long_send(
            connection.target_system,
            connection.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_MODE,
            0,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            0,0,0,0,0,0
        )
        print("manual")
        await_ack(mavutil.mavlink.MAV_CMD_DO_SET_MODE)
        time.sleep(2)
        connection.mav.command_long_send(
            connection.target_system,
            connection.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_MODE,
            0,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            10,0,0,0,0,0
        )
        print("auto")
        await_ack(mavutil.mavlink.MAV_CMD_DO_SET_MODE)
        time.sleep(2)
        ######click download file to see waypoints on map

        break

