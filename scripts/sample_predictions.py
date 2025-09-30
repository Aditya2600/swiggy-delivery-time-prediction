import pandas as pd
import requests
from pathlib import Path
import os

# path for data
root_path = Path(__file__).parent.parent
data_path = root_path / "data" / "raw" / "swiggy.csv"

 # Prefer local FastAPI by default. Override with env var PREDICT_URL if hitting a remote server.
predict_url = os.getenv("PREDICT_URL", "http://127.0.0.1:8000/predict")
print(f"Using prediction endpoint: {predict_url}")

# sample row for testing the endpoint
sample_row = pd.read_csv(data_path).dropna().sample(1)
print("The target value is", sample_row.iloc[:,-1].values.item().replace("(min) ",""))
    
# remove the target column
data = sample_row.drop(columns=[sample_row.columns.tolist()[-1]]).squeeze().to_dict()
print(data)

 # get the response from API (with timeout and clear error handling)
try:
    response = requests.post(url=predict_url, json=data, timeout=10)
except requests.exceptions.ConnectTimeout:
    raise SystemExit("❌ Connection timed out. Is the API running and reachable? "
                     "If it's local, start it with `python app.py` and use the default URL. "
                     "If it's remote, export PREDICT_URL and ensure the server allows inbound traffic.")
except requests.exceptions.ConnectionError as e:
    raise SystemExit(f"❌ Connection error: {e}. "
                     "Check the URL/port, security group/firewall, and that the server process is up.")

print("The status code for response is", response.status_code)

if response.status_code == 200:
    try:
        # If the API returns JSON, prefer that; otherwise fall back to raw text.
        payload = response.json()
        print("Prediction response (JSON):", payload)
        # Try common keys, else fallback to raw text
        if isinstance(payload, dict) and "prediction" in payload:
            print(f"The prediction value by the API is {float(payload['prediction']):.2f} min")
        else:
            print(f"The prediction value by the API is {float(response.text):.2f} min")
    except ValueError:
        # Not JSON: treat as plain text
        print(f"The prediction value by the API is {float(response.text):.2f} min")
else:
    print("Error:", response.status_code, response.text)