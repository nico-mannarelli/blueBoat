import requests
from contacts_coords import coords

def api_upload(lat, lon, cid):
    url = "http://10.107.30.63:30932/marker"

    payload = { 'lat': lat,
            'lon': lon,
            'mission': '1',
            'contents': '""'
    }

    files = [
        ('files', (f'{cid:06d}.png', open(f'C:/Users/kbr_e/Downloads/blueBoat-main/blueBoat-main/sonar_web/data/images/{cid:06d}.png', 'rb'), 'image/png'))
    ]

    headers = {}

    print(f"sent: {lat, lon}")
    response = requests.request("POST", url, headers=headers, data=payload, files=files)

    print(response.text)
