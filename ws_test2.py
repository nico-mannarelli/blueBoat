import websocket
from brping.pingmessage import PingParser, PingMessage
from brping import definitions
import math 
from typing import List, Optional, Tuple
import numpy as np

parser = PingParser()

def on_message(ws, message):
    for byte in message if isinstance(message, (bytes,bytearray)) else message.encode():
        result = parser.parse_byte(byte)
        if result == PingParser.NEW_MESSAGE:
            msg = parser.rx_msg
            print(f"Got message ID: {msg.message_id} - {msg.name}")

            if msg.message_id == 150:
                    print(f"Raw message bytes: {bytes(msg.msg_data.hex())}")

                    
            if msg.message_id == definitions.OMNISCAN450_OS_MONO_PROFILE:
                print(f"Ping #{msg.ping_number}")
                print(f"Range: {msg.start_mm} to {msg.start_mm + msg.length_mm} mm")
                print(f"db range: {msg.min_pwr_db:.1f} to {msg.max_pwr_db:.1f}")
                print(f"Samples: {msg.num_results}")

            

            #     import struct
            #     n = len(msg.pwr_results) // 2
            #     samples = struct.unpack(f'<{n}H', msg.pwr_results)
            #     print(f'First 5 samples: {samples[:5]}')

                


def on_open(ws):
    print("Connected!")

def on_error(ws, error):
    print(f"Error: {error}")

def on_close(ws, close_status_code, close_msg):
    print("Connection closed")

ws = websocket.WebSocketApp(
    "ws://192.168.2.2:7077/connect_ws?session_id=2",
    on_message = on_message,
    on_open = on_open,
    on_error = on_error,
    on_close = on_close
)
ws.run_forever(skip_utf8_validation=True)











