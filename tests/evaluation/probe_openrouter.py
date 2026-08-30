import urllib.request
import json

req = urllib.request.Request('https://openrouter.ai/api/v1/models', headers={'User-Agent': 'SatQueryAI/1.0'})
with urllib.request.urlopen(req, timeout=15) as resp:
    data = json.loads(resp.read().decode('utf-8'))
    models = data.get('data', [])
    free_models = [m for m in models if ':free' in m['id']]
    print("ALL 18 FREE MODELS ON OPENROUTER:")
    for m in sorted(free_models, key=lambda x: x['id']):
        modalities = m.get('architecture', {}).get('modality', 'text')
        print(f"  - {m['id']} (modality: {modalities})")
