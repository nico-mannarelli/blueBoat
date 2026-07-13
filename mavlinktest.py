import os
os.environ['MAVLINK20'] = '1'

from pymavlink import mavutil
import time
import math
from shapely.geometry import Point, Polygon
from pymavlink import mavutil, mavwp
import pymavlink.dialects.v20.all as dialect

# for test
connection = mavutil.mavlink_connection('udp:0.0.0.0:14445')
# for boat
# connection = mavutil.mavlink_connection('tcp:192.168.2.2:5777')

connection.wait_heartbeat()
print("Heartbeat from system (system %u component %u)" % (connection.target_system, connection.target_component))

boot_time = time.time()

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
def upload_mission(waypoints):
    safe_waypoints = [] 
    for lat, lon in waypoints:
        safe_waypoints.append((lat, lon))

    # # clear existing mission
    connection.mav.mission_clear_all_send(
        connection.target_system,
        connection.target_component,
        mavutil.mavlink.MAV_MISSION_TYPE_MISSION
    )
    time.sleep(0.5)

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

    # change speed
    # connection.mav.command_long_send(
    #     connection.target_system,
    #     connection.target_component,
    #     mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED,
    #     0,
    #     1, 0.5, -1, 0, 0, 0, 0
    # )
    

    def send_item(seq, lat, lon, alt, current=0):
        req = connection.recv_match(
            type=['MISSION_REQUEST', 'MISSION_REQUEST_INT'],
            blocking=True, timeout=5
        )
        if req is None:
            print(f"Timeout waiting for request")
            return False
        
        print(f"  Got request for item {req.seq}")
        #seq = req.seq  #took out for sim 

        
        connection.mav.mission_item_int_send(
            connection.target_system,
            connection.target_component,
            seq,
            mavutil.mavlink.MAV_FRAME_GLOBAL_INT,
            mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
            current, 1,        # current, autocontinue
            5, 1.0, 0,        # hold time !!! #, acceptance radius, pass radius
            math.nan,          # yaw
            int(lat * 1e7),
            int(lon * 1e7),
            alt
        )
        print(f"  Sent item {seq}: ({lat}, {lon}, {alt})")
        return True

    # # item 0: home position (same as first waypoint)
    # send_item(waypoints[0][0], waypoints[0][1], waypoints[0][2])

    # items 1+: waypoints
    count = 0
    for lat, lon in safe_waypoints:
        send_item(count, lat, lon, 0.0)
        #wait for 3 secs at each waypoint
        connection.mav.command_long_send(
            connection.target_system,
            connection.target_component,
            mavutil.mavlink.MAV_CMD_NAV_DELAY,
            0,
            -1,
            -1,
            -1,
            1, #secs
            0,
            0,
            0
        )
        # connection.mav.command_long_send(
        #     connection.target_system,
        #     connection.target_component,
        #     mavutil.mavlink.MAV_CMD_NAV_LOITER_TIME,
        #     0,
        #     1,
        #     1,
        #     0,
        #     1,
        #     lat,
        #     lon,
        #     0
        # )

        count += 1
        

    ack = connection.recv_match(type='MISSION_ACK', blocking=True, timeout=5) 
    if ack and ack.type == mavutil.mavlink.MAV_MISSION_ACCEPTED:
        print("Mission upload accepted.")
        return True
    print(f"Mission upload failed: {ack}") # make sure qgroundcontrol connected and in fly or auto not plan
    return False



# ── Main ──────────────────────────────────────────────────────────────────────
from contacts import coords
WAYPOINTS = [
    (38.1437593, -76.5237917),
    (38.1437332, -76.5238217),
    (38.1438235, -76.5237891),
    (38.1437311, -76.5238627),
    (38.1434917, -76.5240546),
    (38.1434619, -76.5240873),
    (38.1423036, -76.5253288),
    (38.1422686, -76.5253383),
    (38.1422699, -76.5253972),
    (38.1422288, -76.5254735),
]

# loiters for 10 secs before starting new mission
# upload_mission(WAYPOINTS) # starts new mission to new waypoints and ends 

#for manual upload 
# will start right away if already in auto mode
# if upload_mission(WAYPOINTS): 
#         set_new_mission()


# # # for automatic upload
while True:
    msg = connection.recv_match(
            type=['MISSION_CURRENT'],
            blocking=True, timeout=5
        )
    if not msg:
        continue
    
    ###check if mission complete and upload mission ready
    if msg.mission_state == 5 and upload_mission(WAYPOINTS):  
        set_new_mission()

        print("Starting new mission")
         
        ### switch to manual then auto to start new mission
        connection.mav.command_long_send(
            connection.target_system,
            connection.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_MODE,
            0,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            0,0,0,0,0,0
        )
        print("manual")
        time.sleep(5)
        connection.mav.command_long_send(
            connection.target_system,
            connection.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_MODE,
            0,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            10,0,0,0,0,0
        )
        print("auto")
        time.sleep(5)
        
        break
        
    
# # return to launch at end of mission
while True:
    msg = connection.recv_match(
            type=['MISSION_CURRENT'],
            blocking=True, timeout=5
        )
    if not msg:
        continue
    
    #check if mission complete and upload mission ready
    if msg.mission_state == 5:  #msg.mission_state == 5 and
         ### comment out so wont automatically start 
         ##have to switch to manual then back to auto to start
        connection.mav.command_long_send(
            connection.target_system,
            connection.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_MODE,
            0,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            0,0,0,0,0,0
        )
        print("manual")
        time.sleep(5)
        connection.mav.command_long_send(
            connection.target_system,
            connection.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_MODE,
            0,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            10,0,0,0,0,0
        )
        print("auto")
        
        set_return()
        
        time.sleep(5)
        
        break
        
    

