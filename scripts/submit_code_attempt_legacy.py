import requests
import json
import os
import time

COMP = 'birdclef-2026'
SUB_DIR = os.path.expanduser('~/.openclaw/workspace-don/kaggle/birdclef-2026/v236-cooccurrence-matrix')
FILE = os.path.join(SUB_DIR, 'script.py')
MSG = 'v236: Co-occurrence matrix post-processing'

def submit():
    with open(os.path.expanduser('~/.kaggle/kaggle.json'), 'r') as f:
        creds = json.load(f)
        KEY = creds['key']

    H = {'Authorization': f'Bearer {KEY}'}
    BASE = 'https://www.kaggle.com/api/v1'

    try:
        # 1. Get signed URL
        file_size = os.path.getsize(FILE)
        epoch = int(time.time())
        r = requests.post(f'{BASE}/competitions/{COMP}/submissions/url/{file_size}/{epoch}', headers=H)
        r.raise_for_status()
        data = r.json()

        # 2. Upload to GCP
        with open(FILE, 'rb') as f:
            r2 = requests.put(data['createUrl'], data=f)
            r2.raise_for_status()

        # 3. Finalize
        r3 = requests.post(f'{BASE}/competitions/submissions/submit/{COMP}',
                           headers=H,
                           data={'blobFileTokens': data['token'], 'submissionDescription': MSG})
        r3.raise_for_status()
        print("Successfully submitted v236 to Kaggle!")
        
    except Exception as e:
        print(f"Submission failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(e.response.text)

if __name__ == '__main__':
    submit()