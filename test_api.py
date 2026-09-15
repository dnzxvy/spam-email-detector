import requests

email = "Congratulations! You have won £500! Click here to claim your prize now!"



response = requests.post(
    "http://127.0.0.1:5000/predict",
    json={
        "email": email
    }
)

print(response.json())