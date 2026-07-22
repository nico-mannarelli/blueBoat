import requests
from contacts_coords import coords

def api_upload(lat, lon, cid, mission=1):
    url = "http://10.107.30.63:30932/marker"

    # mission-1 images are 000001.png; mission-2 images are 2 + 5 digits
    # (see contact_export.py) — the name must match or open() fails.
    if int(mission) == 1:
        img_name = f'{cid:06d}.png'
    else:
        img_name = f'2{cid:05d}.png'

    payload = { 'lat': lat,
            'lon': lon,
            'mission': str(mission),
            'contents': '""'
    }

    headers = {}

    with open(f'C:/Users/kbr_e/Downloads/blueBoat-main/blueBoat-main/sonar_web/data/images/{img_name}', 'rb') as fh:
        files = [
            ('files', (img_name, fh, 'image/png'))
        ]

        print(f"sent: {lat, lon}")
        response = requests.request("POST", url, headers=headers, data=payload, files=files)

    print(response.text)
