import requests
import json
import os

with open(os.path.expanduser('~/.kaggle/kaggle.json'), 'r') as f:
    creds = json.load(f)
    KEY = creds['key']

H = {'Authorization': f'Bearer {KEY}'}
BASE = 'https://www.kaggle.com/api/v1'
COMP = 'birdclef-2026'

r = requests.get(f'{BASE}/competitions/submissions/list/{COMP}', headers=H)
if r.status_code == 200:
    subs = r.json()
    for s in subs[:5]:
        print(f"ID: {s.get('id')} - Score: {s.get('publicScore')} - Status: {s.get('status')} - Desc: {s.get('description')}")
else:
    print(f"Failed: {r.status_code} {r.text}")