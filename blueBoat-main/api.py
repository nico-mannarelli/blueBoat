import requests
from contacts_coords import coords

import argparse

import cv2
import numpy as np
import requests
BBOX = [90, 4, 120, 170]  

def synth_png():
    """A 224x224 test image: mid-grey field, a bright rectangle drawn at BBOX
    (as x,y,w,h), and 'TL' text top-left so orientation is unmistakable."""
    img = np.full((224, 224), 90, np.uint8)
    x, y, w, h = BBOX
    cv2.rectangle(img, (x, y), (x + w, y + h), 255, 2)
    cv2.putText(img, "TL", (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2)
    return cv2.imencode(".png", img)[1].tobytes()


def api_upload(lat, lon, cid, mission=1):
    #url = "http://10.107.30.63:30932/marker"
    url = "http://10.107.30.63:30932/marker/boat"
    BBOX = [90, 4, 120, 170]  

    # mission-1 images are 000001.png; mission-2 images are 2 + 5 digits
    # (see contact_export.py) — the name must match or open() fails.
    if int(mission) == 1:
        img_name = f'{cid:06d}.png'
    else:
        img_name = f'2{cid:05d}.png'

    data = { 'lat': lat,
            'lon': lon,
            "bbox": "[]"
    }

   

    with open(f'C:/Users/kbr_e/Downloads/blueBoat-main/blueBoat-main/sonar_web/data/images/{img_name}', 'rb') as fh:
        # files = [
        #     ('files', (img_name, fh, 'image/png'))
        # ]
        BBOX = [90, 4, 120, 170]  
        ap = argparse.ArgumentParser()
        ap.add_argument("--url", default=url)
        ap.add_argument("--low", help="low_res_img path (default: synthesized)")
        ap.add_argument("--high", help="high_res_img path (default: synthesized)")
        args = ap.parse_args()

        lo = open(args.low, "rb").read() if args.low else synth_png()
        hi = open(args.high, "rb").read() if args.high else synth_png()

        files = [
        ("low_res_img",  (img_name, fh, "image/png")),
        ("high_res_img", (img_name, fh, "image/png")),
    ]


        print(f"sent: {lat, lon}")
        response = requests.post(url, data=data, files=files)


    print(response.text)




# print("HHHH")
# COORDINATES = [(lat, lon, cid) for lat,lon,cid in coords]
# for lat,lon,cid in COORDINATES:
#     api_upload(lat, lon, cid)
#     print(cid)
