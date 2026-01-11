import requests

# --- CONFIGURATION ---
# Replace this with the EXACT Direct URL you copied from the "Embed this space" menu
# Ensure NO spaces at the end, and NO '/predict' at the end yet (we add it below)
BASE_URL = "https://khanakshah-crime-detection-api.hf.space"

# Your File path
VIDEO_PATH = "/Users/khanakshah/Downloads/good1.mp4"

# Your Key
API_KEY = "KK2705709CR"

# --- THE TEST ---
url = f"{BASE_URL}/predict"
print(f"Testing URL: {url}")

try:
    # 1. Open the file
    with open(VIDEO_PATH, "rb") as video_file:
        files = {"file": video_file}
        headers = {"x-api-key": API_KEY}
        
        # 2. Send Request
        print("Sending request... (This might take 10-20 seconds)")
        response = requests.post(url, headers=headers, files=files)
        
        # 3. Print Result
        print(f"Status Code: {response.status_code}")
        
        try:
            print("Response JSON:", response.json())
        except:
            print("Response Text:", response.text)

except FileNotFoundError:
    print(f"Error: Could not find video file at {VIDEO_PATH}")
except Exception as e:
    print(f"An error occurred: {e}")