import json
import urllib.request

url = 'http://127.0.0.1:8000/api/analyze/text'
data = {
    'text': 'I have fatigue and weight loss and I feel dizzy.',
    'speaker': 'PATIENT'
}
req = urllib.request.Request(
    url,
    data=json.dumps(data).encode('utf-8'),
    headers={'Content-Type': 'application/json'}
)
with urllib.request.urlopen(req, timeout=20) as res:
    print(res.read().decode('utf-8'))
