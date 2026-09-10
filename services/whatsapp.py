import os
import requests
from core.config import settings

GRAPH_URL = "https://graph.facebook.com/v20.0"

def download_meta_media(media_id: str, output_filepath: str) -> bool:
    """Retrieves media download URL via Graph API and saves the file locally."""
    headers = {"Authorization": f"Bearer {settings.WHATSAPP_ACCESS_TOKEN}"}
    
    # 1. Fetch the temporary download URL from Meta
    meta_resp = requests.get(f"{GRAPH_URL}/{media_id}", headers=headers, timeout=10)
    if meta_resp.status_code != 200:
        print(f"[-] Meta Media URL Error: {meta_resp.text}")
        return False
        
    media_url = meta_resp.json().get("url")

    # 2. Download binary payload
    file_resp = requests.get(media_url, headers=headers, timeout=20)
    if file_resp.status_code == 200:
        with open(output_filepath, "wb") as f:
            f.write(file_resp.content)
        return True
    return False

def send_whatsapp_text(recipient_phone: str, text: str) -> None:
    """Sends a formatted text message to the user."""
    url = f"{GRAPH_URL}/{settings.WHATSAPP_PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {settings.WHATSAPP_ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": recipient_phone,
        "type": "text",
        "text": {"body": text}
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=10)
    if resp.status_code != 200:
        print(f"[-] WhatsApp Text Dispatch Error: {resp.text}")

def send_whatsapp_audio(recipient_phone: str, audio_url: str) -> None:
    """Sends an audio note via a publicly accessible HTTPS link."""
    url = f"{GRAPH_URL}/{settings.WHATSAPP_PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {settings.WHATSAPP_ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": recipient_phone,
        "type": "audio",
        "audio": {"link": audio_url}
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=10)
    if resp.status_code == 200:
        print(f"[+] Audio successfully dispatched to {recipient_phone}")
    else:
        print(f"[-] WhatsApp Audio Dispatch Error: {resp.text}")


def send_whatsapp_interactive_reply(recipient_phone: str, text: str):
    """Sends a quick processing message."""
    url = f"{GRAPH_URL}/{settings.WHATSAPP_PHONE_NUMBER_ID}/messages"
    headers = {"Authorization": f"Bearer {settings.WHATSAPP_ACCESS_TOKEN}", "Content-Type": "application/json"}
    payload = {"messaging_product": "whatsapp", "to": recipient_phone, "type": "text", "text": {"body": text}}
    requests.post(url, headers=headers, json=payload)

def mark_whatsapp_read(message_id: str):
    """Marks the incoming message as read."""
    url = f"{GRAPH_URL}/{settings.WHATSAPP_PHONE_NUMBER_ID}/messages"
    headers = {"Authorization": f"Bearer {settings.WHATSAPP_ACCESS_TOKEN}", "Content-Type": "application/json"}
    payload = {"messaging_product": "whatsapp", "status": "read", "message_id": message_id}
    requests.post(url, headers=headers, json=payload)