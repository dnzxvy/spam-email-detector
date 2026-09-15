import requests

email = "Congratulations! You have won £500! Click here to claim your prize now!"

email_ham = "Hey ELi, its me Amy, just wanted to know if you are still free \n for the night lol "


response = requests.post(
    "http://127.0.0.1:5000/predict",
    json={
        "email": email,
        "email_ham": email
    }
)

print(response.json())