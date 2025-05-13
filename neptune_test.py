import requests

# Replace with your actual Neptune endpoint (no "https://", no trailing slash)
NEPTUNE_HOST = "db-dev-neptune-1.cluster-cd0iemc4ay88.us-east-2.neptune.amazonaws.com"
NEPTUNE_HOST  = "db-dev-neptune-1-instance-1.cd0iemc4ay88.us-east-2.neptune.amazonaws.com"
PORT = 8182

def check_status():
    url = f"https://{NEPTUNE_HOST}:{PORT}/status"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        print("✅ Neptune status:")
        print(response.json())
    except requests.exceptions.RequestException as e:
        print("❌ Error connecting to Neptune:")
        print(e)

if __name__ == "__main__":
    check_status()
