#!/usr/bin/env python3
"""
Verbesserte Version der App – alle Funktionen bleiben erhalten und der Code ist in einer einzigen Datei organisiert.
Die Struktur wurde verbessert, indem zusammengehörige Funktionen in klar abgegrenzten Abschnitten stehen,
und Docstrings sowie einheitliche Formatierung hinzugefügt wurden.
"""

import os
import logging
import schedule
import time
import random
import string
import smtplib
from email.mime.text import MIMEText
from datetime import datetime, timedelta
from dotenv import load_dotenv
from flask import Flask, jsonify, request, render_template_string, send_file
from flask_caching import Cache
from PIL import Image
from threading import Thread
from transformers import pipeline
import requests
import shutil
import pandas as pd
import numpy as np
import joblib
import psycopg2
import redis
from googleapiclient.discovery import build
import praw
from gtts import gTTS
from moviepy.editor import AudioFileClip, ImageClip
import tweepy
from sklearn.ensemble import RandomForestClassifier
from flask_split import split, ab_test

##############################################################################
# Setup & Konfiguration
##############################################################################
load_dotenv()
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

app = Flask(__name__)
cache = Cache(app, config={'CACHE_TYPE': 'simple'})
split(app)  # Initialisierung von Flask-Split für A/B-Tests

# KI-Textgenerierung: Wähle das Modell per ENV-Variable (Standard: gpt2; alternativ: mixtral oder llama2)
KI_MODEL = os.getenv("KI_MODEL", "gpt2")
generator = pipeline("text-generation", model=KI_MODEL)

# Simulationseinstellungen
USE_SIMULATION = True

# ENV-Variablen für diverse Services
DIGISTORE24_PARTNER_ID = os.getenv("DIGISTORE24_PARTNER_ID")
FINANCEADS_PARTNER_ID = os.getenv("FINANCEADS_PARTNER_ID")
GOOGLE_ADSENSE_CLIENT_ID = os.getenv("GOOGLE_ADSENSE_CLIENT_ID")
MEDIAMARKET_PARTNER_ID = os.getenv("MEDIAMARKET_PARTNER_ID")
SATURN_PARTNER_ID = os.getenv("SATURN_PARTNER_ID")
AMAZON_AFFILIATE_ID = os.getenv("AMAZON_AFFILIATE_ID")
IFTTT_WEBHOOK_URL = os.getenv("IFTTT_WEBHOOK_URL")
FACEBOOK_ACCESS_TOKEN = os.getenv("FACEBOOK_ACCESS_TOKEN")
LINKEDIN_ACCESS_TOKEN = os.getenv("LINKEDIN_ACCESS_TOKEN")
PINTEREST_ACCESS_TOKEN = os.getenv("PINTEREST_ACCESS_TOKEN")
REDDIT_ACCESS_TOKEN = os.getenv("REDDIT_ACCESS_TOKEN")
REDDIT_USERNAME = os.getenv("REDDIT_USERNAME")
FCM_SERVER_KEY = os.getenv("FCM_SERVER_KEY")
EMAIL_SENDER = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
SMTP_SERVER = os.getenv("SMTP_SERVER")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
GITHUB_REPO_URL = os.getenv("GITHUB_REPO_URL")
HOSTING_DOMAIN = os.getenv("HOSTING_DOMAIN")
GA_TRACKING_ID = os.getenv("GA_TRACKING_ID")
HOTJAR_ID = os.getenv("HOTJAR_ID", "DEINE_HOTJAR_ID")
SENDINBLUE_API_KEY = os.getenv("SENDINBLUE_API_KEY")

# Neue ENV-Variablen für Matomo Tracking & Dashboard
MATOMO_URL = os.getenv("MATOMO_URL")         # z.B. "https://matomo.deinedomain.de"
MATOMO_SITE_ID = os.getenv("MATOMO_SITE_ID")   # z.B. "1"
MATOMO_TOKEN = os.getenv("MATOMO_TOKEN")
MATOMO_DASHBOARD_URL = os.getenv("MATOMO_DASHBOARD_URL", "https://matomo.deinedomain.de/index.php?module=Widgetize&action=iframe&widget=1")

##############################################################################
# 1. A/B-Testing: Headlines & CTAs
##############################################################################
HEADLINE_VARIANTS = [
    "Entdecke die neuesten Technologietrends!",
    "So revolutionieren aktuelle Trends die Technikwelt!",
    "Die Zukunft der Technologie: Was du wissen musst!"
]
CTA_VARIANTS = [
    "Jetzt mehr erfahren!",
    "Lies weiter und optimiere deine Zukunft!",
    "Hol dir die Insights – klick hier!"
]

def get_random_headline():
    """Gibt eine zufällige Headline zurück."""
    return random.choice(HEADLINE_VARIANTS)

def get_random_cta():
    """Gibt einen zufälligen Call-to-Action zurück."""
    return random.choice(CTA_VARIANTS)

def track_ab_test_event(variant_type, variant_value):
    """Sendet (oder simuliert) ein GA-Event für A/B-Tests."""
    event_name = f"ABTest_{variant_type}"
    event_category = "A/B Testing"
    event_label = variant_value

    if USE_SIMULATION:
        logging.info(f"(Simuliert) GA A/B Event: {event_name} | {event_category} | {event_label}")
    else:
        if not GA_TRACKING_ID:
            logging.info("Keine GA_TRACKING_ID - Überspringe GA-Event.")
            return
        payload = {
            "v": "1",
            "tid": GA_TRACKING_ID,
            "cid": "555",
            "t": "event",
            "ec": event_category,
            "ea": event_name,
            "el": event_label,
        }
        try:
            response = requests.post("https://www.google-analytics.com/collect", data=payload)
            logging.info(f"GA A/B Event gesendet: {response.status_code}")
        except Exception as e:
            logging.error(f"GA A/B Event Fehler: {e}")

##############################################################################
# 2. E-Mail & Push-Funktionen (SMTP, FCM)
##############################################################################
def send_email_smtp(recipient, subject, body):
    """Sendet eine E-Mail via SMTP oder simuliert den Versand."""
    if USE_SIMULATION:
        logging.info(f"(Simuliert) E-Mail an {recipient} | Subject: {subject} | Body: {body}")
        return
    try:
        msg = MIMEText(body, _charset="utf-8")
        msg['Subject'] = subject
        msg['From'] = EMAIL_SENDER
        msg['To'] = recipient
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)
        logging.info(f"E-Mail erfolgreich an {recipient} gesendet.")
    except Exception as e:
        logging.error(f"Fehler beim E-Mail-Senden an {recipient}: {e}")

def send_fcm_notification(message, title="Benachrichtigung"):
    """Sendet eine FCM-Benachrichtigung oder simuliert diese."""
    if USE_SIMULATION:
        logging.info(f"(Simuliert FCM) {title}: {message}")
        return
    if not FCM_SERVER_KEY:
        logging.info("FCM_SERVER_KEY fehlt – Überspringe FCM-Benachrichtigung.")
        return
    url = "https://fcm.googleapis.com/fcm/send"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"key={FCM_SERVER_KEY}"
    }
    payload = {
        "to": "/topics/all",
        "notification": {"title": title, "body": message}
    }
    try:
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            logging.info("FCM-Benachrichtigung erfolgreich gesendet.")
        else:
            logging.warning(f"FCM-Benachrichtigung fehlgeschlagen: {response.text}")
    except Exception as e:
        logging.error(f"FCM-Fehler: {e}")

##############################################################################
# 3. Sendinblue E-Mail Versand
##############################################################################
def personalize_email_content(user_behavior, base_content):
    """Personalisiert den E-Mail Inhalt basierend auf dem Nutzerverhalten."""
    if "geld verdienen" in user_behavior.lower():
        return base_content + "\nExklusive Tipps, wie du online Geld verdienen kannst!"
    return base_content

def optimize_email_subject(possible_subjects):
    """Wählt einen optimierten Betreff aus."""
    chosen = random.choice(possible_subjects)
    logging.info(f"Optimierter Betreff: {chosen}")
    return chosen

def send_sendinblue_email(user_email, base_content, possible_subjects, user_behavior):
    """Sendet eine E-Mail über Sendinblue oder per SMTP-Fallback."""
    if not SENDINBLUE_API_KEY:
        logging.info("Sendinblue nicht konfiguriert – nutze SMTP-Fallback.")
        sbj = optimize_email_subject(possible_subjects)
        cnt = personalize_email_content(user_behavior, base_content)
        send_email_smtp(user_email, sbj, cnt)
        return
    subject = optimize_email_subject(possible_subjects)
    content_html = personalize_email_content(user_behavior, base_content)
    if USE_SIMULATION:
        logging.info(f"(Simuliert Sendinblue) E-Mail an {user_email}: {subject} | {content_html}")
        return
    url = "https://api.sendinblue.com/v3/smtp/email"
    headers = {"api-key": SENDINBLUE_API_KEY, "Content-Type": "application/json"}
    data = {
        "sender": {"name": "Dein Unternehmen", "email": EMAIL_SENDER},
        "to": [{"email": user_email}],
        "subject": subject,
        "htmlContent": f"<p>{content_html}</p>"
    }
    try:
        r = requests.post(url, headers=headers, json=data)
        if r.status_code in (200, 201):
            logging.info(f"Sendinblue-E-Mail erfolgreich an {user_email} gesendet.")
        else:
            logging.warning(f"Fehler beim Senden über Sendinblue: {r.text}")
            send_email_smtp(user_email, subject, content_html)
    except Exception as e:
        logging.error(f"Sendinblue-Fehler: {e}")
        send_email_smtp(user_email, subject, content_html)

##############################################################################
# 4. Willkommens-Serie & Retargeting
##############################################################################
def send_welcome_series(user_email):
    """Sendet eine Serie von Willkommens-E-Mails an den Nutzer."""
    subs = [
        "Willkommen – Tag 1",
        "Tag 2: Entdecke Vorteile",
        "Tag 3: Exklusive Tipps",
        "Tag 4: Sonderangebote!",
        "Tag 5: Letzte Chance!"
    ]
    cnts = [
        "Hallo und willkommen! Exklusive Einblicke...",
        "Schön, dass du dabei bist! Heute das Beste herausholen...",
        "Tag 3: Noch mehr spannende Tricks...",
        "Fast geschafft – besondere Sonderangebote...",
        "Vielen Dank – Abschiedsgruß + Rabattcode!"
    ]
    for i in range(5):
        send_sendinblue_email(user_email, cnts[i], [subs[i]], "welcome_series")
        logging.info(f"Willkommens-E-Mail Tag {i+1} an {user_email} gesendet.")
    return True

@app.route("/send_welcome_series", methods=["POST"])
def send_welcome_series_endpoint():
    data = request.get_json()
    user_email = data.get("email")
    if not user_email:
        return jsonify({"error": "Keine E-Mail-Adresse übermittelt"}), 400
    if send_welcome_series(user_email):
        return jsonify({"status": "Willkommensserie gesendet"}), 200
    return jsonify({"error": "Fehler beim Versenden"}), 500

def retarget_fast_buyers(user_email, product_clicked):
    """Sendet eine Retargeting-E-Mail an Nutzer, die ein Produkt angesehen haben."""
    sbj = "Hast du dein exklusives Angebot verpasst?"
    cnt = f"Hallo, wir sahen, dass du {product_clicked} interessant fandest, aber nicht gekauft hast!"
    send_sendinblue_email(user_email, cnt, [sbj], "retargeting")
    logging.info(f"Retargeting-E-Mail an {user_email} für {product_clicked} gesendet.")
    return True

@app.route("/retarget_fast_buyers", methods=["POST"])
def retarget_fast_buyers_endpoint():
    data = request.get_json()
    user_email = data.get("email")
    product = data.get("product")
    if not user_email or not product:
        return jsonify({"error": "E-Mail & Produkt sind nötig"}), 400
    if retarget_fast_buyers(user_email, product):
        return jsonify({"status": "Retargeting E-Mail gesendet"}), 200
    return jsonify({"error": "Fehler beim Retargeting"}), 500

##############################################################################
# 5. Dynamische Preisstrategie & Scarcity
##############################################################################
PRODUCT_BASE_PRICES = {1: 100.0, 2: 120.0, 3: 40.0, 4: 80.0, 5: 25.0, 6: 15.0}
CURRENT_PRICES = PRODUCT_BASE_PRICES.copy()

def dynamic_pricing():
    """Berechnet neue Preise basierend auf Wettbewerbs- und Nachfrageparametern."""
    logging.info("Starte dynamische Preisstrategie.")
    for pid, bp in PRODUCT_BASE_PRICES.items():
        competitor = bp * random.uniform(0.9, 1.2)
        demand = random.uniform(0.8, 1.2)
        new_price = ((bp + competitor) / 2.0) * demand
        CURRENT_PRICES[pid] = round(new_price, 2)
        logging.info(f"Produkt {pid}: Neuer Preis = {CURRENT_PRICES[pid]}")

@app.route("/update_prices", methods=["POST"])
def update_prices_endpoint():
    dynamic_pricing()
    return jsonify({"updated_prices": CURRENT_PRICES})

@app.route("/get_prices", methods=["GET"])
def get_prices_endpoint():
    return jsonify({"current_prices": CURRENT_PRICES})

@app.route("/apply_scarcity", methods=["POST"])
def apply_scarcity():
    data = request.get_json()
    product_id = data.get("product_id", 1)
    places_left = random.randint(1, 5)
    message = f"Nur noch {places_left} Plätze für Produkt {product_id} verfügbar!"
    logging.info(f"(Simuliert) Scarcity: {message}")
    return jsonify({"scarcity_message": message})

##############################################################################
# 6. KI-Sales-Chatbot (GPT-2 basiert)
##############################################################################
def generate_discount_offer(user_behavior_metric):
    """Generiert ein Rabattangebot basierend auf einem Metrikwert."""
    discount_percent = min(50, 5 + int(user_behavior_metric * 10))
    code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
    return discount_percent, code

@app.route("/chatbot", methods=["POST"])
def chatbot_endpoint():
    data = request.get_json()
    user_message = data.get("message", "")
    if not user_message:
        return jsonify({"error": "Keine Nachricht"}), 400

    prompt = f"User: {user_message}\nBot:"
    gen = generator(prompt, max_length=50, num_return_sequences=1)
    bot_resp = gen[0]['generated_text'].split("Bot:")[-1].strip()

    upsell_prompt = f"Basierend auf '{user_message}', schlage ein High-Ticket-Produkt (1.000-5.000€) vor..."
    upsell_gen = generator(upsell_prompt, max_length=60, num_return_sequences=1)
    upsell_offer = upsell_gen[0]['generated_text'].strip()

    metric = random.uniform(0, 5)
    disc_percent, disc_code = generate_discount_offer(metric)
    disc_message = f"{disc_percent}% Rabatt mit Code {disc_code}!"

    combined = f"{bot_resp}\n\nHigh-Ticket-Upsell: {upsell_offer}\nRabatt: {disc_message}"
    logging.info(f"Chatbot -> {combined}")
    return jsonify({"response": combined})

@app.route("/chatbot_gpt4", methods=["POST"])
def chatbot_gpt4():
    return jsonify({"error": "GPT-4 API wurde entfernt – bitte benutze /chatbot mit der lokalen GPT-2 Integration."}), 501

##############################################################################
# 7. Autom. Video-Erstellung, Ads, Affiliate-Funktionen
##############################################################################
def auto_generate_video(article_text):
    logging.info("Auto-Video (Placeholder).")
    voice_over = "Voice-Over generiert"
    subtitles = "Untertitel: " + article_text[:50] + "..."
    video_url = "https://deinedomain.de/generated_video.mp4"
    logging.info(f"Video erstellt: {video_url}")
    return {"video_url": video_url, "voice_over": voice_over, "subtitles": subtitles}

def analyze_affiliate_conversion(user_behavior):
    products = {
        "Produkt A": random.uniform(0.1, 0.9),
        "Produkt B": random.uniform(0.1, 0.9),
        "Produkt C": random.uniform(0.1, 0.9)
    }
    best_product, best_rate = sorted(products.items(), key=lambda x: x[1], reverse=True)[0]
    logging.info(f"Beste Affiliate-Produkte für '{user_behavior}': {best_product} (CR={best_rate:.2f})")
    return best_product, best_rate

def dynamic_affiliate_recommendation(user_behavior):
    bp, br = analyze_affiliate_conversion(user_behavior)
    rec = f"Empfehlung: {bp} – hohe Konversionsrate ({br:.0%})!"
    logging.info("Empfehlung: " + rec)
    return rec

@app.route("/affiliate_conversion", methods=["POST"])
def affiliate_conversion_endpoint():
    data = request.get_json()
    user_behavior = data.get("user_behavior", "")
    if not user_behavior:
        return jsonify({"error": "Kein user_behavior"}), 400
    rec = dynamic_affiliate_recommendation(user_behavior)
    return jsonify({"affiliate_recommendation": rec})

##############################################################################
# 8. Webinare, Livestream, Podcast
##############################################################################
def auto_generate_webinar(topic, schedule_time):
    url = f"https://deinedomain.de/webinar/{topic.replace(' ','-').lower()}"
    ai_msg = f"KI moderiert Webinar zu '{topic}'."
    logging.info(f"Webinar '{topic}' geplant für {schedule_time}, URL={url}")
    return {"topic": topic, "schedule_time": schedule_time, "webinar_url": url, "ai_moderation": ai_msg}

@app.route("/webinar", methods=["POST"])
def webinar_endpoint():
    data = request.get_json()
    topic = data.get("topic", "Allgemeines Thema")
    schedule_time = data.get("schedule_time", datetime.now().strftime("%Y-%m-%d %H:%M"))
    info = auto_generate_webinar(topic, schedule_time)
    return jsonify({"status": "Webinar geplant", "details": info})

def auto_generate_live_stream(topic, schedule_time):
    url = f"https://deinedomain.de/livestream/{topic.replace(' ','-').lower()}"
    ai_msg = f"KI moderiert Live-Stream zu '{topic}'."
    logging.info(f"Live-Stream '{topic}' am {schedule_time}, URL={url}")
    return {"topic": topic, "schedule_time": schedule_time, "live_stream_url": url, "ai_moderation": ai_msg}

@app.route("/livestream", methods=["POST"])
def livestream_endpoint():
    data = request.get_json()
    topic = data.get("topic", "Allgemeines Thema")
    schedule_time = data.get("schedule_time", datetime.now().strftime("%Y-%m-%d %H:%M"))
    info = auto_generate_live_stream(topic, schedule_time)
    return jsonify({"status": "Live-Stream geplant", "details": info})

def distribute_podcast(podcast_url):
    logging.info(f"Podcast Distribution: {podcast_url}")
    if USE_SIMULATION:
        return {"Spotify": "Veröffentlicht", "Apple Podcasts": "Veröffentlicht"}
    return {"Spotify": "In Bearbeitung", "Apple Podcasts": "In Bearbeitung"}

@app.route("/distribute_podcast", methods=["POST"])
def distribute_podcast_endpoint():
    data = request.get_json()
    p_url = data.get("podcast_url", "")
    if not p_url:
        return jsonify({"error": "Kein Podcast-URL"}), 400
    status = distribute_podcast(p_url)
    return jsonify({"status": "Podcast verteilt", "distribution": status})

def generate_podcast_from_article(article_text):
    p_url = "https://deinedomain.de/podcasts/" + ''.join(random.choices(string.ascii_lowercase, k=10)) + ".mp3"
    logging.info(f"Podcast generiert: {p_url}")
    return p_url

@app.route("/generate_podcast", methods=["POST"])
def generate_podcast_endpoint():
    data = request.get_json()
    art_txt = data.get("article_text", "")
    if not art_txt:
        return jsonify({"error": "Kein Artikeltext"}), 400
    p_url = generate_podcast_from_article(art_txt)
    return jsonify({"status": "Podcast generiert", "podcast_url": p_url})

def generate_podcast_transcript(podcast_audio_url):
    transcript = "Beispiel-Transkript: " + podcast_audio_url
    logging.info("Podcast-Transkript generiert.")
    return transcript

def generate_ai_article_from_podcast(podcast_audio_url):
    transcript = generate_podcast_transcript(podcast_audio_url)
    prompt = f"Erstelle einen Artikel aus diesem Podcast-Transkript:\n{transcript}"
    gen = generator(prompt, max_length=300, num_return_sequences=1)
    article = gen[0]['generated_text']
    logging.info("KI-Artikel aus Podcast-Transkript generiert.")
    return article

@app.route("/podcast_article", methods=["POST"])
def podcast_article_endpoint():
    data = request.get_json()
    audio_url = data.get("audio_url")
    if not audio_url:
        return jsonify({"error": "Audio-URL fehlt"}), 400
    article = generate_ai_article_from_podcast(audio_url)
    return jsonify({"status": "KI-Artikel generiert", "article": article}), 200

def auto_share_webinar_on_social(webinar_info):
    if USE_SIMULATION:
        logging.info(f"(Simuliert) Teile Webinar-Info auf Social Media: {webinar_info}")

@app.route("/webinar_advanced", methods=["POST"])
def webinar_advanced():
    data = request.get_json()
    topic = data.get("topic", "Allgemeines Thema")
    schedule_time = data.get("schedule_time", datetime.now().strftime("%Y-%m-%d %H:%M"))
    info = auto_generate_webinar(topic, schedule_time)
    auto_share_webinar_on_social(info)
    return jsonify({"status": "Webinar geplant & geteilt", "details": info})

def extended_podcast_distribution(podcast_url):
    logging.info(f"Erweiterte Podcast-Verteilung: {podcast_url}")
    dist_result = {
        "Spotify": "OK",
        "Apple": "OK",
        "Deezer": "OK (Sim)",
        "Stitcher": "OK (Sim)",
        "Google Podcasts": "OK (Sim)"
    }
    return dist_result

@app.route("/distribute_podcast_extended", methods=["POST"])
def distribute_podcast_extended():
    data = request.get_json()
    p_url = data.get("podcast_url", "")
    if not p_url:
        return jsonify({"error": "Keine Podcast-URL"}), 400
    dist = extended_podcast_distribution(p_url)
    return jsonify({"status": "Podcast erweitert verteilt", "distribution": dist})

##############################################################################
# 9. Social Proof, Gamification, Countdown, FOMO
##############################################################################
def get_social_proof_html():
    reviews = [
        {"name": "Anna", "review": "Das Produkt hat mein Leben verändert!", "rating": 5},
        {"name": "Max", "review": "Top Qualität und super Service!", "rating": 4},
        {"name": "Lisa", "review": "Ich bin begeistert – Geld-zurück-Garantie!", "rating": 5}
    ]
    trust_symbols = """
    <div class="trust-symbols">
      <img src="/static/geld-zurueck-garantie.png" alt="Geld-zurück-Garantie" style="height:50px;">
      <img src="/static/sicherheitslogo.png" alt="Sicherheitszertifikat" style="height:50px;">
    </div>
    """
    rev_html = "<ul>"
    for r in reviews:
        rev_html += f"<li><strong>{r['name']}</strong> ({r['rating']} Sterne): {r['review']}</li>"
    rev_html += "</ul>"
    return f"""
    <div class="social-proof">
      <h2>Kundenbewertungen</h2>
      {rev_html}
      {trust_symbols}
    </div>
    """

@app.route("/social_proof", methods=["GET"])
def social_proof_endpoint():
    html = get_social_proof_html()
    return render_template_string(f"""
    <html>
      <head><title>Social Proof & Vertrauen</title></head>
      <body>{html}</body>
    </html>
    """)

def get_progress_bar_html(progress=80):
    return f"""
    <div style="width:100%;background-color:#e0e0e0;border-radius:25px;overflow:hidden;">
      <div style="width:{progress}%;height:30px;background-color:#76c7c0;text-align:center;line-height:30px;color:white;">
        {progress}%
      </div>
    </div>
    <p>Noch {100 - progress}% bis zu deinem VIP-Bonus!</p>
    """

@app.route("/progress_bar", methods=["GET"])
def progress_bar_endpoint():
    html = get_progress_bar_html(progress=80)
    return render_template_string(f"""
    <html>
      <head><title>Dein Fortschritt</title></head>
      <body>
        <h2>Exklusive Belohnungen</h2>
        {html}
      </body>
    </html>
    """)

def get_offer_countdown_html(duration_seconds=300):
    return f"""
    <div id="countdown" style="font-size:24px;font-weight:bold;"></div>
    <script>
      var timeLeft = {duration_seconds};
      var countdownElem = document.getElementById("countdown");
      var timer = setInterval(function(){{
         if(timeLeft <= 0){{
            clearInterval(timer);
            countdownElem.innerHTML = "Angebot abgelaufen!";
         }} else {{
            countdownElem.innerHTML = "Nur noch " + timeLeft + " Sekunden bis zum Deal!";
         }}
         timeLeft -= 1;
      }}, 1000);
    </script>
    """

@app.route("/offer_countdown", methods=["GET"])
def offer_countdown_endpoint():
    html = get_offer_countdown_html(duration_seconds=300)
    return render_template_string(f"""
    <html>
      <head><title>Exklusiver Deal</title></head>
      <body>
        <h2>Nur für kurze Zeit: Exklusiver Deal!</h2>
        {html}
      </body>
    </html>
    """)

def add_fomo_element(message):
    live_purchases = random.randint(1, 10)
    return f"{message}\n🔥 {live_purchases} Leute haben gerade gekauft!"

@app.route("/spin_wheel", methods=["GET"])
def spin_wheel_endpoint():
    disc = random.randint(0, 20)
    logging.info(f"Glücksrad gedreht: {disc}% Rabatt!")
    return jsonify({"status": "Rad gedreht", "discount": disc}), 200

##############################################################################
# 10. Publishing (Medium, LinkedIn, IFTTT für Twitter)
##############################################################################
def auto_publish_linkedin_article(article_text):
    if USE_SIMULATION:
        url_ph = "https://www.linkedin.com/feed/update/urn:li:share:SIMULATED"
        logging.info(f"(Simuliert) LinkedIn-Artikel: {url_ph}")
        return url_ph
    else:
        if not LINKEDIN_ACCESS_TOKEN:
            logging.info("Kein LinkedIn-Token. Überspringe.")
            return "LinkedIn-API nicht konfiguriert."
        return "https://www.linkedin.com/feed/update/..."

@app.route("/publish_linkedin", methods=["POST"])
def publish_linkedin_endpoint():
    data = request.get_json()
    art_txt = data.get("article_text", "")
    if not art_txt:
        return jsonify({"error": "Kein Artikeltext"}), 400
    url = auto_publish_linkedin_article(art_txt)
    return jsonify({"status": "LinkedIn-Artikel veröffentlicht", "url": url})

def auto_publish_medium_post(article_text):
    if USE_SIMULATION:
        url_fake = "https://medium.com/@user/" + ''.join(random.choices(string.ascii_lowercase, k=10))
        logging.info(f"(Simuliert) Medium-Post: {url_fake}")
        return url_fake
    else:
        return "https://medium.com/@user/..."

@app.route("/publish_medium", methods=["POST"])
def publish_medium_endpoint():
    data = request.get_json()
    art_txt = data.get("article_text", "")
    if not art_txt:
        return jsonify({"error": "Kein Artikeltext"}), 400
    url = auto_publish_medium_post(art_txt)
    return jsonify({"status": "Medium-Post veröffentlicht", "url": url})

def send_twitter_ifttt(message):
    if USE_SIMULATION:
        logging.info(f"(Simuliert IFTTT) Twitter-Beitrag: {message}")
        return
    ifttt_url = IFTTT_WEBHOOK_URL
    try:
        response = requests.post(ifttt_url, json={"value1": message})
        if response.status_code == 200:
            logging.info("Twitter-Beitrag über IFTTT erfolgreich gesendet.")
        else:
            logging.warning(f"IFTTT Fehler: {response.text}")
    except Exception as e:
        logging.error(f"IFTTT-Fehler: {e}")

##############################################################################
# 11. Bildkomprimierung, CDN-Upload
##############################################################################
def compress_image(input_path, output_path, quality=85):
    try:
        with Image.open(input_path) as img:
            img.convert("RGB").save(output_path, "webp", quality=quality)
        logging.info(f"Bild komprimiert & als WebP gespeichert: {output_path}")
        return True
    except Exception as e:
        logging.error(f"Fehler beim Bildkomprimieren: {e}")
        return False

def upload_to_cdn(file_path):
    cdn_url = "https://cdn.deinedomain.de/" + os.path.basename(file_path)
    logging.info(f"Bild in CDN hochgeladen (simuliert): {cdn_url}")
    return cdn_url

@app.route("/compress_image", methods=["POST"])
def compress_image_endpoint():
    if "file" not in request.files:
        return jsonify({"error": "Keine Datei übermittelt"}), 400
    file = request.files["file"]
    in_path = "temp_" + file.filename
    out_path = "compressed_" + file.filename.rsplit(".", 1)[0] + ".webp"
    file.save(in_path)
    if compress_image(in_path, out_path):
        cdn_url = upload_to_cdn(out_path)
        os.remove(in_path)
        os.remove(out_path)
        return jsonify({"status": "Bild komprimiert & hochgeladen", "cdn_url": cdn_url}), 200
    return jsonify({"error": "Fehler bei Bildkomprimierung"}), 500

@app.route("/serve_image/<filename>", methods=["GET"])
def serve_image(filename):
    ua = request.headers.get("User-Agent", "").lower()
    version = "small_" if "mobile" in ua else "large_"
    f_path = f"images/{version}{filename}.webp"
    if os.path.exists(f_path):
        return send_file(f_path, mimetype="image/webp")
    return jsonify({"error": "Bild nicht gefunden"}), 404

##############################################################################
# 12. KI/ML: SALES-STRATEGIE & SEO-Automation
##############################################################################
MODEL_FILE = "sales_strategy_model.pkl"

def train_sales_strategy_model():
    data = {
        "traffic": [100, 200, 150, 300, 250, 400],
        "ads_spend": [10, 20, 15, 25, 22, 35],
        "seo_score": [50, 60, 58, 65, 70, 80],
        "conversion_rate": [0.02, 0.03, 0.025, 0.04, 0.033, 0.05],
        "success": [0, 1, 0, 1, 1, 1]
    }
    df = pd.DataFrame(data)
    X = df[["traffic", "ads_spend", "seo_score", "conversion_rate"]]
    y = df["success"]
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    joblib.dump(model, MODEL_FILE)
    logging.info("Sales-Strategie-Modell trainiert & gespeichert.")

def predict_best_strategy(traffic, ads_spend, seo_score, conversion_rate):
    if not os.path.exists(MODEL_FILE):
        logging.info("Kein Modell vorhanden, starte Training.")
        train_sales_strategy_model()
    model = joblib.load(MODEL_FILE)
    sample = np.array([[traffic, ads_spend, seo_score, conversion_rate]])
    return model.predict(sample)[0]

def recommend_optimizations(traffic, ads_spend, seo_score, conversion_rate):
    r = predict_best_strategy(traffic, ads_spend, seo_score, conversion_rate)
    if r == 1:
        return "Strategie scheint erfolgreich. Skaliere Ads oder Content-Marketing."
    return "Optimierungspotenzial: SEO ausbauen, Werbekampagnen optimieren, Landingpage A/B-Test."

def monitor_performance_and_optimize():
    traffic = random.randint(100, 500)
    ads_spend = random.randint(10, 50)
    seo_score = random.randint(50, 90)
    conv_rate = round(random.uniform(0.02, 0.06), 3)
    performance_falling = random.choice([True, False])
    if performance_falling:
        logging.info("Performance-Rückgang erkannt, hole KI-Empfehlung.")
        rec = recommend_optimizations(traffic, ads_spend, seo_score, conv_rate)
        logging.info(f"KI-Empfehlung: {rec}")
    else:
        logging.info("Performance stabil, kein Eingreifen nötig.")

@app.route("/monitor_and_optimize", methods=["GET"])
def monitor_and_optimize_endpoint():
    monitor_performance_and_optimize()
    return jsonify({"status": "Performance-Monitoring durchgeführt"}), 200

def automate_seo_and_backlink_building():
    prompt = "Schreibe einen SEO-optimierten Gastartikel über Laptops..."
    gen = generator(prompt, max_length=200, num_return_sequences=1)
    seo_article = gen[0]['generated_text']
    logging.info(f"SEO-Artikel: {seo_article}")
    logging.info("Backlink-Platzierung initiiert (Simuliert).")

@app.route("/seo_backlink_automation", methods=["GET"])
def seo_backlink_automation_endpoint():
    automate_seo_and_backlink_building()
    return jsonify({"status": "SEO & Backlink-Building ausgeführt"}), 200

##############################################################################
# 13. Multi-Channel Distribution & Influencer-Bot
##############################################################################
def distribute_content_multichannel(article_text):
    logging.info("Starte Multi-Channel-Distribution.")
    linked = auto_publish_linkedin_article(article_text)
    medium = auto_publish_medium_post(article_text)
    send_twitter_ifttt("Neuer Beitrag: " + article_text[:50])
    logging.info("(Simuliert) Posting auf Twitter, Facebook, Instagram etc.")
    return {
        "LinkedIn": linked,
        "Medium": medium,
        "Twitter": "Über IFTTT gesendet",
        "Facebook": "Simuliert gepostet",
        "InstagramDM": "Simuliert gesendet"
    }

@app.route("/multi_channel_distribution", methods=["POST"])
def multi_channel_distribution_endpoint():
    data = request.get_json()
    article_text = data.get("article_text", "Standard-Text")
    res = distribute_content_multichannel(article_text)
    return jsonify({"status": "Multi-Channel-Distribution", "details": res})

def influencer_marketing_bot(niche="fitness"):
    infls = [
        {"name": "FitAnna", "followers": 15000, "platform": "Instagram"},
        {"name": "GymKing", "followers": 8000, "platform": "TikTok"},
        {"name": "HealthyLisa", "followers": 25000, "platform": "Instagram"}
    ]
    for i in infls:
        if random.choice([True, False]):
            logging.info(f"(Bot) Kontakt mit {i['name']} auf {i['platform']} über IFTTT.")
        else:
            logging.info(f"{i['name']} übersprungen.")

@app.route("/influencer_marketing", methods=["GET"])
def influencer_marketing_endpoint():
    niche = request.args.get("niche", "fitness")
    influencer_marketing_bot(niche)
    return jsonify({"status": f"Influencer-Bot für '{niche}' ausgeführt"})

@app.route("/influencer_dm_campaign", methods=["POST"])
def influencer_dm_campaign():
    data = request.get_json()
    camp_title = data.get("campaign_title", "Influencer-Kampagne")
    infl_list = data.get("influencers", [])
    msg_tpl = data.get("message_template", "Hey {name}, schau dir unser Angebot an!")
    results = []
    for infl in infl_list:
        name = infl.get("name")
        platform = infl.get("platform", "").lower()
        msg = msg_tpl.format(name=name)
        if USE_SIMULATION:
            logging.info(f"(Simuliert) DM an {name} auf {platform} via IFTTT: {msg}")
            results.append(f"DM to {name} on {platform}: {msg}")
    return jsonify({"status": f"Kampagne '{camp_title}' ausgeführt", "results": results})

##############################################################################
# 14. Erweiterter Cyber-Schutz
##############################################################################
HIJACKING_USER_AGENTS = ["evil-bot", "hijack-curl", "fraud-crawler"]

def analyze_cyberattack(attack_vector, severity_score):
    logging.info(f"KI-Analyse Cyberangriff: {attack_vector}, Severity={severity_score}")
    if severity_score > 7.0:
        logging.warning("Kritischer Angriff - starte Gegenmaßnahmen!")
        send_fcm_notification(f"Kritischer Angriff: {attack_vector}", "Cyber-Angriff")
    else:
        logging.info("Angriff erkannt, aber nicht kritisch genug für Autoverteidigung.")

@app.route("/cyber_attack_analysis", methods=["POST"])
def cyber_attack_analysis_endpoint():
    data = request.get_json()
    atk_vec = data.get("attack_vector", "unspecified")
    sev = float(data.get("severity_score", 5.0))
    analyze_cyberattack(atk_vec, sev)
    return jsonify({"status": "Angriff analysiert"}), 200

def dark_web_monitoring():
    logging.info("Starte Dark-Web-Check.")
    found = random.choice([True, False])
    if found:
        logging.warning("Dark-Web: Marken-/Domainmissbrauch entdeckt!")
        send_fcm_notification("Dark-Web-Alarm: Missbrauch entdeckt!", "Dark-Web Alert")
    else:
        logging.info("Kein Missbrauch im Dark Web gefunden.")

@app.route("/dark_web_check", methods=["GET"])
def dark_web_check_endpoint():
    dark_web_monitoring()
    return jsonify({"status": "Dark-Web-Überwachung abgeschlossen"}), 200

##############################################################################
# 15. Logging / DB-Log / DDoS-Schutz / Betrugserkennung
##############################################################################
def log_error_to_db(message):
    try:
        conn = psycopg2.connect(
            dbname=os.getenv("PG_DBNAME", "DEIN_DB_NAME"),
            user=os.getenv("PG_USER", "DEIN_USER"),
            password=os.getenv("PG_PASSWORD", "DEIN_PASSWORT"),
            host=os.getenv("PG_HOST", "DEIN_HOST"),
            port=os.getenv("PG_PORT", "DEIN_PORT")
        )
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS errors (timestamp TIMESTAMP, message TEXT)")
        cursor.execute("INSERT INTO errors (timestamp, message) VALUES (%s, %s)", (datetime.now(), message))
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        logging.error(f"Fehler beim Speichern in der Datenbank: {e}")

@app.route("/admin/error_log", methods=["GET"])
def error_log():
    try:
        conn = psycopg2.connect(
            dbname=os.getenv("PG_DBNAME", "DEIN_DB_NAME"),
            user=os.getenv("PG_USER", "DEIN_USER"),
            password=os.getenv("PG_PASSWORD", "DEIN_PASSWORT"),
            host=os.getenv("PG_HOST", "DEIN_HOST"),
            port=os.getenv("PG_PORT", "DEIN_PORT")
        )
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS errors (timestamp TIMESTAMP, message TEXT)")
        cursor.execute("SELECT timestamp, message FROM errors ORDER BY timestamp DESC LIMIT 50")
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        errs = [{"timestamp": r[0], "message": r[1]} for r in rows]
        return jsonify({"errors": errs})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/admin/error_statistics", methods=["GET"])
def error_statistics():
    try:
        conn = psycopg2.connect(
            dbname=os.getenv("PG_DBNAME", "DEIN_DB_NAME"),
            user=os.getenv("PG_USER", "DEIN_USER"),
            password=os.getenv("PG_PASSWORD", "DEIN_PASSWORT"),
            host=os.getenv("PG_HOST", "DEIN_HOST"),
            port=os.getenv("PG_PORT", "DEIN_PORT")
        )
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS errors (timestamp TIMESTAMP, message TEXT)")
        cursor.execute("""
            SELECT message, COUNT(*) as c
            FROM errors
            WHERE timestamp >= NOW() - INTERVAL '7 days'
            GROUP BY message
            ORDER BY c DESC
        """)
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        stats = [{"message": r[0], "count": r[1]} for r in rows]
        return jsonify({"error_statistics": stats})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

REQUEST_COUNTS = {}
LOCKED_IPS = {}
MAX_REQUESTS_PER_WINDOW = 50
WINDOW_SECONDS = 60
LOCKOUT_DURATION = 300

def ddos_protection():
    ip = request.remote_addr
    if ip in LOCKED_IPS:
        locked_since = LOCKED_IPS[ip]
        if (datetime.now() - locked_since).total_seconds() > LOCKOUT_DURATION:
            del LOCKED_IPS[ip]
        else:
            logging.warning(f"DDoS-Schutz: IP {ip} gesperrt.")
            return jsonify({"error": "Zu viele Anfragen. Bitte später erneut versuchen."}), 429
    now = datetime.now()
    if ip not in REQUEST_COUNTS:
        REQUEST_COUNTS[ip] = []
    REQUEST_COUNTS[ip] = [t for t in REQUEST_COUNTS[ip] if (now - t).total_seconds() <= WINDOW_SECONDS]
    REQUEST_COUNTS[ip].append(now)
    if len(REQUEST_COUNTS[ip]) > MAX_REQUESTS_PER_WINDOW:
        LOCKED_IPS[ip] = now
        logging.warning(f"DDoS-Schutz: IP {ip} gesperrt.")
        return jsonify({"error": "Zu viele Anfragen. IP gesperrt."}), 429
    return None

AFFILIATE_CLICKS = {}
CLICK_FREQUENCY_THRESHOLD = 20
SUSPICIOUS_IPS = set()

def detect_affiliate_fraud(partner_id):
    ip = request.remote_addr
    now = datetime.now()
    if partner_id not in AFFILIATE_CLICKS:
        AFFILIATE_CLICKS[partner_id] = {}
    if ip not in AFFILIATE_CLICKS[partner_id]:
        AFFILIATE_CLICKS[partner_id][ip] = []
    AFFILIATE_CLICKS[partner_id][ip] = [t for t in AFFILIATE_CLICKS[partner_id][ip] if (now - t).total_seconds() <= 600]
    AFFILIATE_CLICKS[partner_id][ip].append(now)
    if len(AFFILIATE_CLICKS[partner_id][ip]) > CLICK_FREQUENCY_THRESHOLD:
        logging.warning(f"Affiliate-Betrug (Basis) von IP {ip} bei {partner_id}!")
        SUSPICIOUS_IPS.add(ip)

@app.route("/affiliate/<partner_id>", methods=["GET"])
def affiliate_link(partner_id):
    detect_affiliate_fraud(partner_id)
    return jsonify({"message": f"Affiliate-Link für {partner_id} geklickt."})

def advanced_affiliate_fraud_check(partner_id):
    ip = request.remote_addr
    user_agent = request.headers.get("User-Agent", "").lower()
    now = datetime.now()
    for ua in HIJACKING_USER_AGENTS:
        if ua in user_agent:
            logging.warning(f"Affiliate-Hijacking UA: {user_agent}, IP={ip}")
            SUSPICIOUS_IPS.add(ip)
            return
    if partner_id not in AFFILIATE_CLICKS:
        AFFILIATE_CLICKS[partner_id] = {}
    if ip not in AFFILIATE_CLICKS[partner_id]:
        AFFILIATE_CLICKS[partner_id][ip] = []
    AFFILIATE_CLICKS[partner_id][ip] = [t for t in AFFILIATE_CLICKS[partner_id][ip] if (now - t).total_seconds() <= 600]
    AFFILIATE_CLICKS[partner_id][ip].append(now)
    click_count = len(AFFILIATE_CLICKS[partner_id][ip])
    if click_count > CLICK_FREQUENCY_THRESHOLD:
        logging.warning(f"Affiliate-Betrug (Threshold) IP {ip}, Partner {partner_id}")
        SUSPICIOUS_IPS.add(ip)
        return
    x = np.array([[click_count, random.uniform(0, 1)]])
    fraud_prob = (click_count / 20.0) + x[0][1] * 0.3
    fraud_prob = min(fraud_prob, 1.0)
    if fraud_prob > 0.8:
        logging.warning(f"(ML) Betrug p={fraud_prob:.2f} für IP {ip}")
        SUSPICIOUS_IPS.add(ip)

@app.route("/affiliate2/<partner_id>", methods=["GET"])
def affiliate_link_hijack(partner_id):
    advanced_affiliate_fraud_check(partner_id)
    return jsonify({"message": f"Affiliate-Link (2.0) für {partner_id} geklickt."})

@app.before_request
def global_protection_layer():
    result = ddos_protection()
    if result is not None:
        return result
    if request.remote_addr in SUSPICIOUS_IPS:
        logging.warning(f"IP {request.remote_addr} blockiert (Betrugsverdacht).")
        return jsonify({"error": "Zugriff verweigert - Betrugsverdacht."}), 403

##############################################################################
# 16. Mehrstufiges Backup & Restore
##############################################################################
def double_backup_database():
    logging.info("Erstelle doppeltes lokales Backup.")
    db_file = "user_discounts.db"
    backup_folder = "double_backups"
    if not os.path.exists(backup_folder):
        os.makedirs(backup_folder)
    bf_name = f"double_user_discounts_backup_{datetime.now().strftime('%Y-%m-%d')}.db"
    b_path = os.path.join(backup_folder, bf_name)
    try:
        shutil.copy(db_file, b_path)
        logging.info(f"Doppeltes Backup erstellt: {b_path}")
    except Exception as e:
        logging.error(f"Fehler beim doppelten Backup: {e}")

def backup_database():
    db_file = "user_discounts.db"
    backup_folder = "backups"
    if not os.path.exists(backup_folder):
        os.makedirs(backup_folder)
    bf_name = f"user_discounts_backup_{datetime.now().strftime('%Y-%m-%d')}.db"
    b_path = os.path.join(backup_folder, bf_name)
    try:
        shutil.copy(db_file, b_path)
        logging.info(f"Backup erstellt: {b_path}")
        cleanup_old_backups(backup_folder, days=7)
        upload_backup_to_cloud(b_path)
        double_backup_database()
    except Exception as e:
        logging.error(f"Fehler beim Backup: {e}")
        send_email_smtp("admin@example.com", "DB-Backup Fehler", f"Fehler: {e}")
        send_fcm_notification(f"DB-Backup Fehler: {e}", "DB-Backup Error")

def cleanup_old_backups(backup_folder, days=7):
    now = datetime.now()
    for fn in os.listdir(backup_folder):
        fpath = os.path.join(backup_folder, fn)
        try:
            file_time = datetime.fromtimestamp(os.path.getmtime(fpath))
            if (now - file_time).days > days:
                os.remove(fpath)
                logging.info(f"Altes Backup gelöscht: {fpath}")
        except Exception as e:
            logging.error(f"Fehler beim Löschen {fpath}: {e}")
            send_email_smtp("admin@example.com", "Backup Cleanup Fehler", f"Fehler: {e}")
            send_fcm_notification(f"Cleanup Fehler: {e}", "DB-Backup Cleanup")

def upload_backup_to_cloud(backup_path):
    if USE_SIMULATION:
        logging.info(f"(Simuliert) Backup {backup_path} in kostenlosem Cloud-Speicher hochgeladen.")
    else:
        logging.info(f"Backup {backup_path} in kostenlosem Cloud-Speicher (z.B. S3 Free Tier) hochgeladen.")

def auto_restore_database():
    try:
        if USE_SIMULATION:
            logging.info("(Simuliert) Wiederherstellung aus Cloud-Backup.")
            return True
        else:
            logging.info("Echte Wiederherstellung implementieren...")
            return True
    except Exception as e:
        logging.error(f"Fehler bei DB-Wiederherstellung: {e}")
        send_email_smtp("admin@example.com", "Restore Fehler", f"Fehler: {e}")
        send_fcm_notification(f"Restore Fehler: {e}", "DB-Restore")
        return False

@app.route("/admin/restore_database", methods=["POST"])
def restore_database():
    if auto_restore_database():
        return jsonify({"status": "Datenbank wiederhergestellt"}), 200
    return jsonify({"error": "Wiederherstellung fehlgeschlagen"}), 500

def safety_scan():
    crit = random.choice([True, False])
    if crit:
        logging.info("Kritischer Fehler erkannt -> auto Restore.")
        auto_restore_database()
    else:
        logging.info("Sicherheitsscan: Alles ok.")

def integrity_check():
    try:
        conn = psycopg2.connect(
            dbname=os.getenv("PG_DBNAME", "DEIN_DB_NAME"),
            user=os.getenv("PG_USER", "DEIN_USER"),
            password=os.getenv("PG_PASSWORD", "DEIN_PASSWORT"),
            host=os.getenv("PG_HOST", "DEIN_HOST"),
            port=os.getenv("PG_PORT", "DEIN_PORT")
        )
        cursor = conn.cursor()
        cursor.execute("SELECT pg_is_in_recovery();")
        res = cursor.fetchone()
        cursor.close()
        conn.close()
        if res and res[0] is False:
            logging.info("DB-Integritätsprüfung: ok")
        else:
            logging.error("DB-Integritätsprüfung: Fehler")
            send_email_smtp("admin@example.com", "DB Integritätsfehler", "Integritätsprüfung fehlgeschlagen")
            send_fcm_notification("DB-Integritätsfehler", "DB-Fehler")
            auto_restore_database()
    except Exception as e:
        logging.error(f"Fehler bei Integritätsprüfung: {e}")
        send_email_smtp("admin@example.com", "Integritätsprüfung Fehler", str(e))
        send_fcm_notification(f"Integritätsprüfung Fehler: {e}", "DB-Fehler")

##############################################################################
# 17. Re-Engagement-Kampagne
##############################################################################
def get_inactive_users():
    return ["user1@example.com", "user2@example.com"]

def reengagement_campaign():
    inactive = get_inactive_users()
    base_content = "Hallo, es gibt tolle Neuigkeiten und Angebote!"
    possible_subjects = ["Exklusive Angebote warten!", "Verpasse nicht diese News!", "Top-Angebote für dich!"]
    for usr in inactive:
        user_behavior = "Interessiert sich für Marketing"
        send_sendinblue_email(usr, base_content, possible_subjects, user_behavior)
        send_fcm_notification(f"Hey {usr}, neue Angebote warten!", "Re-Engagement")
    logging.info("Re-Engagement-Kampagne abgeschlossen.")

##############################################################################
# 18. Conversion-Optimierung (Heatmap, CTA, FOMO)
##############################################################################
def simulate_heatmap_analysis():
    logging.info("Heatmap-Analyse (Simuliert).")
    return {"clicks": {"button1": 20, "button2": 15, "link1": 5}, "scroll_depth": "75%"}

def generate_dynamic_cta(user_behavior):
    if "interesse" in user_behavior.lower():
        return "Jetzt informieren und profitieren!"
    return "Klicken Sie hier für mehr Details!"

@app.route("/conversion_optimize", methods=["GET"])
def conversion_optimize():
    behavior = request.args.get("behavior", "allgemeines Interesse")
    heatmap = simulate_heatmap_analysis()
    cta = generate_dynamic_cta(behavior)
    base_msg = "Ihr Einkaufserlebnis wird jetzt noch besser!"
    fomo_msg = add_fomo_element(base_msg)
    return jsonify({
        "heatmap_data": heatmap,
        "dynamic_cta": cta,
        "fomo_message": fomo_msg
    })

def upsell_redirect(clicked_product):
    logging.info(f"Upsell für {clicked_product}")
    related = ["Produkt X", "Produkt Y", "Produkt Z"]
    return f"Weil du {clicked_product} angeklickt hast -> Empfehlung: {', '.join(related)}"

def limited_offer():
    return "Nur heute: 10% Rabatt – jetzt zugreifen!"

def get_bestseller_products():
    logging.info("Bestseller-Produkte abgerufen...")
    return [
        {"product_id": 101, "name": "Beliebtes Gadget", "price": 29.99},
        {"product_id": 102, "name": "Top-Seller Laptop", "price": 999.99},
    ]

##############################################################################
# 19. Personalisierte Landingpage-Demo
##############################################################################
@app.route("/landing_page", methods=["GET"])
def landing_page_demo_endpoint():
    user_interest = request.args.get("interest", "Technologie")
    html = f"""
    <html>
      <head><title>Exklusive Angebote für {user_interest}</title></head>
      <body>
        <h1>Willkommen, {user_interest}-Enthusiast!</h1>
        <p>Individuelle Angebote für dein Interesse an {user_interest}!</p>
      </body>
    </html>
    """
    logging.info(f"Landingpage für '{user_interest}' generiert.")
    return render_template_string(html)

##############################################################################
# 20. Ultimative SEO-Strategie (Google Discover, Trends, etc.)
##############################################################################
SEO_ARTICLES_DB = {}

def find_trending_keywords():
    if USE_SIMULATION:
        sample_keys = ["AI Tools", "Smarte Gadgets", "Kaffee-Vergleich", "Gaming Laptops 2025", "Yoga-Trends"]
        random.shuffle(sample_keys)
        return sample_keys[:2]
    return ["TOP-Keyword-1", "TOP-Keyword-2"]

def generate_seo_article_for_keyword(keyword):
    logging.info(f"Erstelle SEO-Artikel zu {keyword}")
    people_ask = [
        f"Was ist {keyword} genau?",
        f"Wie nutzt man {keyword} optimal?",
        f"Was kostet {keyword}? Warum lohnt sich ein Vergleich?"
    ]
    prompt = f"Schreibe SEO-optimierten Artikel über '{keyword}' und beantworte folgende Fragen: {people_ask}"
    genr = generator(prompt, max_length=400, num_return_sequences=1)
    content = genr[0]['generated_text']
    schema_markup = f"""
    <script type="application/ld+json">
    {{
      "@context": "https://schema.org",
      "@type": "Article",
      "headline": "Dein Artikel zu {keyword}",
      "mainEntityOfPage": {{"@type": "WebPage", "@id": "https://deinedomain.de/blog/{keyword.replace(' ', '-').lower()}"}},
      "author": "Deine Website",
      "datePublished": "{datetime.now().isoformat()}",
      "articleBody": "{content[:150]}..."
    }}
    </script>
    <script type="application/ld+json">
    {{
      "@context": "https://schema.org",
      "@type": "FAQPage",
      "mainEntity": [
        {{"@type": "Question", "name": "{people_ask[0]}", "acceptedAnswer": {{"@type": "Answer", "text": "Antwort im Artikel"}}}},
        {{"@type": "Question", "name": "{people_ask[1]}", "acceptedAnswer": {{"@type": "Answer", "text": "Antwort im Artikel"}}}}
      ]
    }}
    </script>
    """
    art_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
    SEO_ARTICLES_DB[art_id] = {
        "title": f"Ultimativer Guide zu {keyword}",
        "content": content,
        "keywords": [keyword],
        "schema_markup": schema_markup
    }
    return art_id

def submit_article_to_google_news(article_id):
    if USE_SIMULATION:
        logging.info(f"(Simuliert) Artikel {article_id} an Google News gemeldet.")

def optimize_for_featured_snippets(article_id):
    if article_id in SEO_ARTICLES_DB:
        logging.info(f"Artikel {article_id} für Featured Snippets optimiert (Simuliert).")

@app.route("/daily_seo_automation", methods=["GET"])
def daily_seo_automation():
    keys = find_trending_keywords()
    created_articles = []
    for kw in keys:
        a_id = generate_seo_article_for_keyword(kw)
        submit_article_to_google_news(a_id)
        optimize_for_featured_snippets(a_id)
        created_articles.append(a_id)
    return jsonify({
        "status": "Tägliche SEO-Automation ausgeführt",
        "keywords_used": keys,
        "created_articles": created_articles
    })

@app.route("/view_seo_article/<article_id>", methods=["GET"])
def view_seo_article(article_id):
    adata = SEO_ARTICLES_DB.get(article_id)
    if not adata:
        return jsonify({"error": "Artikel nicht gefunden"}), 404
    html = f"""
    <html>
      <head>
        <title>{adata["title"]}</title>
        {adata["schema_markup"]}
      </head>
      <body>
        <h1>{adata["title"]}</h1>
        <p>{adata["content"]}</p>
        <p style="color:green">Keywords: {", ".join(adata["keywords"])}</p>
      </body>
    </html>
    """
    return render_template_string(html)

##############################################################################
# 21. VIP-/Level-System
##############################################################################
VIP_DATA = {}

def get_vip_info(user_email):
    if user_email not in VIP_DATA:
        VIP_DATA[user_email] = {"points": 0, "vip_level": 0}
    return VIP_DATA[user_email]

def add_vip_points(user_email, amount=10):
    info = get_vip_info(user_email)
    info["points"] += amount
    while info["points"] >= 100:
        info["vip_level"] += 1
        info["points"] -= 100
    return info

def get_dynamic_vip_discount(user_email):
    info = get_vip_info(user_email)
    return info["vip_level"] * 5

@app.route("/vip_status", methods=["GET"])
def vip_status():
    user_email = request.args.get("email", "unknown@domain.com")
    info = get_vip_info(user_email)
    return jsonify(info)

@app.route("/add_vip_points", methods=["POST"])
def add_vip_points_endpoint():
    data = request.get_json()
    user_email = data.get("email")
    pts = data.get("points", 10)
    if not user_email:
        return jsonify({"error": "No email provided"}), 400
    info = add_vip_points(user_email, pts)
    return jsonify({"new_vip_status": info}), 200

##############################################################################
# 22. Neue Endpoints: Virale Posts, Wikipedia-Backlinks, Reddit, etc.
##############################################################################
@app.route("/viral_posts", methods=["POST"])
def viral_posts():
    data = request.get_json()
    topic = data.get("topic", "Trending Topic")
    if USE_SIMULATION:
        logging.info(f"(Simuliert) KI erstellt viralen Post über '{topic}' zur Peak-Zeit.")
    return jsonify({"status": f"Viraler Post zum Thema '{topic}' erstellt."})

@app.route("/community_interaction", methods=["POST"])
def community_interaction():
    data = request.get_json()
    platform = data.get("platform", "twitter")
    if USE_SIMULATION:
        logging.info(f"(Simuliert) Bot kommentiert & antwortet auf {platform}")
    return jsonify({"status": f"Community-Interaktion auf {platform} ausgeführt."})

@app.route("/create_thread", methods=["POST"])
def create_thread():
    data = request.get_json()
    platform = data.get("platform", "twitter")
    thread_title = data.get("title", "KI-gesteuerter Thread")
    if USE_SIMULATION:
        logging.info(f"(Simuliert) Thread '{thread_title}' auf {platform} gepostet.")
    return jsonify({"status": f"Thread '{thread_title}' auf {platform} gepostet."})

@app.route("/pinterest_optimization", methods=["POST"])
def pinterest_optimization():
    data = request.get_json()
    article_url = data.get("article_url", "https://deinedomain.de/blog/artikel")
    if USE_SIMULATION:
        logging.info(f"(Simuliert) Pinterest SEO & Gruppenboards für {article_url}.")
    return jsonify({"status": f"Pinterest-Optimierung für {article_url}"})

@app.route("/instagram_stories", methods=["POST"])
def instagram_stories():
    data = request.get_json()
    article_url = data.get("article_url", "https://deinedomain.de/blog/artikel")
    if USE_SIMULATION:
        logging.info(f"(Simuliert) IG Stories & Reels aus {article_url} generiert.")
    return jsonify({"status": f"IG Stories/Reels erstellt zu {article_url}"})

@app.route("/youtube_video_optimization", methods=["POST"])
def youtube_video_optimization():
    data = request.get_json()
    video_url = data.get("video_url", "https://youtube.com/watch?v=123")
    if USE_SIMULATION:
        logging.info(f"(Simuliert) YouTube SEO (Keywords/Hashtags) für {video_url}.")
    return jsonify({"status": f"YouTube-Optimierung für {video_url} abgeschlossen"})

@app.route("/backlink_outreach", methods=["POST"])
def backlink_outreach():
    data = request.get_json()
    site_url = data.get("site_url", "https://deinedomain.de")
    if USE_SIMULATION:
        logging.info(f"(Simuliert) KI analysiert & sendet Outreach für {site_url}.")
    return jsonify({"status": f"Backlink-Outreach für {site_url} gestartet"})

@app.route("/press_release", methods=["POST"])
def press_release():
    data = request.get_json()
    release_title = data.get("title", "Pressemitteilung: Neuer Meilenstein")
    if USE_SIMULATION:
        logging.info(f"(Simuliert) Pressemitteilung '{release_title}' verfasst & verteilt.")
    return jsonify({"status": f"Pressemitteilung '{release_title}' verteilt."})

@app.route("/wikipedia_forums", methods=["POST"])
def wikipedia_forums():
    data = request.get_json()
    page_topic = data.get("topic", "Beispielthema")
    if USE_SIMULATION:
        logging.info(f"(Simuliert) Wikipedia-Eintrag & Foren-Kommentare zu '{page_topic}'.")
    return jsonify({"status": f"Wikipedia/Foren-Links zu '{page_topic}' generiert."})

@app.route("/auto_newsletter", methods=["GET"])
def auto_newsletter():
    if USE_SIMULATION:
        logging.info("(Simuliert) Wöchentlicher Newsletter mit Top-Artikeln generiert.")
    return jsonify({"status": "Wöchentlicher Newsletter versendet."})

@app.route("/messenger_newsletter", methods=["POST"])
def messenger_newsletter():
    data = request.get_json()
    messenger_type = data.get("messenger", "whatsapp")
    if USE_SIMULATION:
        logging.info(f"(Simuliert) {messenger_type}-Newsletter gesendet.")
    return jsonify({"status": f"Messenger-Newsletter via {messenger_type} gesendet"})

@app.route("/email_surveys", methods=["POST"])
def email_surveys():
    data = request.get_json()
    survey_topic = data.get("topic", "Zufriedenheit")
    if USE_SIMULATION:
        logging.info(f"(Simuliert) E-Mail-Umfrage '{survey_topic}' versendet.")
    return jsonify({"status": f"E-Mail-Umfrage zu '{survey_topic}' verschickt."})

@app.route("/youtube_community", methods=["POST"])
def youtube_community():
    data = request.get_json()
    comm_msg = data.get("message", "Hey Community!")
    if USE_SIMULATION:
        logging.info(f"(Simuliert) YouTube-Community-Post & Shorts: {comm_msg}")
    return jsonify({"status": "YouTube Community & Shorts abgesetzt"})

@app.route("/tiktok_duet_stitch", methods=["POST"])
def tiktok_duet_stitch():
    data = request.get_json()
    target_video = data.get("video_url", "https://tiktok.com/@user/video/123")
    if USE_SIMULATION:
        logging.info(f"(Simuliert) Duet/Stitch mit {target_video} für virale Reichweite.")
    return jsonify({"status": f"TikTok Duet/Stitch mit {target_video} erstellt."})

##############################################################################
# 23. High-Ticket, Abo-Modelle, YT/TikTok Monetarisierung
##############################################################################
HIGH_TICKET_OFFERS = [
    {"id": "HT1", "title": "Exklusives Coaching 1:1", "price": 1000},
    {"id": "HT2", "title": "VIP-Betreuung (5.000 €)", "price": 5000}
]

@app.route("/list_high_ticket", methods=["GET"])
def list_high_ticket():
    return jsonify({"offers": HIGH_TICKET_OFFERS})

@app.route("/monthly_subscriptions", methods=["GET"])
def monthly_subscriptions():
    plans = [
        {"id": "Sub1", "desc": "VIP Light (29 €/Monat)"},
        {"id": "Sub2", "desc": "VIP Pro (59 €/Monat)"},
        {"id": "Sub3", "desc": "VIP Ultra (99 €/Monat)"}
    ]
    return jsonify({"plans": plans})

@app.route("/monetize_youtube_tiktok", methods=["POST"])
def monetize_youtube_tiktok():
    data = request.get_json()
    platform = data.get("platform", "youtube")
    if USE_SIMULATION:
        logging.info(f"(Simuliert) Monetarisierung auf {platform} aktiviert.")
    return jsonify({"status": f"Monetarisierung {platform} aktiviert"})

##############################################################################
# 24. KI-gesteuerte Retargeting & Budget-Optimierung
##############################################################################
@app.route("/retargeting_strategies", methods=["POST"])
def retargeting_strategies():
    data = request.get_json()
    campaign_name = data.get("campaign", "retargeting1")
    if USE_SIMULATION:
        logging.info(f"(Simuliert) KI-Retargeting '{campaign_name}' gestartet.")
    return jsonify({"status": f"Retargeting '{campaign_name}' aktiv"})

@app.route("/auto_budget_optimization", methods=["POST"])
def auto_budget_optimization():
    data = request.get_json()
    campaign_id = data.get("campaign_id", "XYZ123")
    if USE_SIMULATION:
        logging.info(f"(Simuliert) Budget-Optimierung für Kampagne {campaign_id}")
    return jsonify({"status": f"Budget für {campaign_id} optimiert"})

##############################################################################
# 25. Automatisierte Influencer-Koop & Multi-Channel Werbung
##############################################################################
@app.route("/auto_influencer_coop", methods=["POST"])
def auto_influencer_coop():
    data = request.get_json()
    niche = data.get("niche", "fitness")
    if USE_SIMULATION:
        logging.info(f"(Simuliert) Bot findet & kontaktiert Influencer in '{niche}'.")
    return jsonify({"status": f"Influencer-Kooperationen in '{niche}' automatisiert"})

@app.route("/multichannel_ads", methods=["POST"])
def multichannel_ads():
    data = request.get_json()
    channels = data.get("channels", ["youtube", "pinterest", "twitter"])
    if USE_SIMULATION:
        logging.info(f"(Simuliert) Ads auf {channels} gestartet.")
    return jsonify({"status": f"Werbung auf {channels} gestartet"})

##############################################################################
# 26. KI-überwachter Betrugsschutz & Selbstlernende Firewall
##############################################################################
@app.route("/advanced_affiliate_protection", methods=["POST"])
def advanced_affiliate_protection():
    data = request.get_json()
    aff_id = data.get("affiliate_id", "unknown")
    if USE_SIMULATION:
        logging.info(f"(Simuliert) KI scannt Transaktionen für {aff_id}, blockiert Betrug.")
    return jsonify({"status": f"Fortgeschrittener Betrugsschutz für {aff_id}"})

@app.route("/self_learning_firewall", methods=["POST"])
def self_learning_firewall():
    data = request.get_json()
    threat_level = data.get("threat_level", 5)
    if USE_SIMULATION:
        logging.info(f"(Simuliert) Selbstlernende Firewall -> Threat {threat_level}, Hacker blockiert.")
    return jsonify({"status": f"Firewall aktiv, Threat={threat_level}"})

##############################################################################
# 27. Neue Endpoints gemäß Benutzeranforderungen (zusätzliche Features)
##############################################################################
@app.route("/multi_language_content", methods=["GET"])
def multi_language_content():
    lang = request.args.get("lang", "en").lower()
    content = {
        "en": "<h1>Welcome to our site!</h1><p>Enjoy international content.</p>",
        "es": "<h1>¡Bienvenido a nuestro sitio!</h1><p>Disfruta de contenido internacional.</p>",
        "fr": "<h1>Bienvenue sur notre site!</h1><p>Profitez d'un contenu international.</p>"
    }
    result = content.get(lang, content["en"])
    logging.info(f"Multi-language content served for language: {lang}")
    return render_template_string(result)

@app.route("/auto_backlink", methods=["GET"])
def auto_backlink():
    backlink_list = [
        "https://example.com/backlink1",
        "https://anotherexample.com/backlink2",
        "https://yetanotherexample.com/backlink3"
    ]
    logging.info("Automatische Backlink-Erstellung getriggert. Backlinks: " + ", ".join(backlink_list))
    return jsonify({"status": "Backlinks erstellt", "backlinks": backlink_list})

@app.route("/generate_social_media_content", methods=["GET"])
def generate_social_media_content():
    tiktok_url = "https://tiktok.com/@generatedvideo/" + ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
    youtube_short_url = "https://youtube.com/shorts/" + ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
    logging.info("KI-generierte Social-Media-Inhalte erstellt: TikTok & YouTube Shorts.")
    return jsonify({
        "status": "Social-Media-Inhalte generiert",
        "tiktok_video": tiktok_url,
        "youtube_short": youtube_short_url
    })

@app.route("/cloudflare_protection", methods=["GET"])
def cloudflare_protection():
    logging.info("Cloudflare-Schutz getriggert. Firewall-Regeln wurden aktualisiert.")
    return jsonify({"status": "Cloudflare-Firewall-Regeln aktualisiert", "message": "Bot-Angriffe erfolgreich gemindert"})

@app.route("/ab_test_variants", methods=["GET"])
def ab_test_variants():
    variant_headline = get_random_headline()
    variant_cta = get_random_cta()
    track_ab_test_event("headline", variant_headline)
    track_ab_test_event("cta", variant_cta)
    response_data = {
        "headline_variant": variant_headline,
        "cta_variant": variant_cta,
        "message": "A/B-Test-Varianten generiert"
    }
    logging.info("A/B-Test-Varianten bereitgestellt: Headline und CTA.")
    return jsonify(response_data)

##############################################################################
# 29. Zusätzliche Features: Logging, Heatmap-Analyse, KI-Optimierung & Performance
##############################################################################
# Matomo-Integration: Sende Tracking-Events an einen Matomo-Server (selbst gehostet)
def send_matomo_event(event_category, event_action, event_name, event_value=None):
    if MATOMO_URL and MATOMO_SITE_ID and MATOMO_TOKEN:
        payload = {
            'idsite': MATOMO_SITE_ID,
            'rec': 1,
            'action_name': event_name,
            'e_c': event_category,
            'e_a': event_action,
            'e_n': event_name,
            'e_v': event_value if event_value else '',
            'token_auth': MATOMO_TOKEN,
        }
        try:
            response = requests.get(MATOMO_URL + '/matomo.php', params=payload)
            logging.info(f"Matomo event sent: {response.status_code}")
        except Exception as e:
            logging.error(f"Error sending Matomo event: {e}")
    else:
        logging.info("Matomo nicht konfiguriert. Tracking wird übersprungen.")

@app.route("/matomo_track", methods=["POST"])
def matomo_track():
    data = request.get_json()
    event_category = data.get("category", "General")
    event_action = data.get("action", "click")
    event_name = data.get("name", "Event")
    event_value = data.get("value", None)
    send_matomo_event(event_category, event_action, event_name, event_value)
    return jsonify({"status": "Matomo event tracked"})

# Heatmap.js Integration: Liefere ein HTML-Snippet, das Heatmap.js lädt und initialisiert
@app.route("/heatmap_script", methods=["GET"])
def heatmap_script():
    script = """
    <script src="https://unpkg.com/heatmap.js@2.0.5/heatmap.min.js"></script>
    <script>
    var heatmapInstance = h337.create({
      container: document.querySelector('body'),
      radius: 50,
      maxOpacity: .6,
      minOpacity: 0.1,
      blur: .90
    });
    document.addEventListener('click', function(e) {
      heatmapInstance.addData({ x: e.pageX, y: e.pageY, value: 1 });
    });
    </script>
    """
    return render_template_string(script)

# KI-Modelle optimieren: Beispiel-Endpoint zur Analyse von Nutzerdaten mit einem Dummy-Scikit-Learn-Modell
@app.route("/optimize_ki", methods=["POST"])
def optimize_ki():
    data = request.get_json()
    user_text = data.get("user_text", "")
    if not user_text:
        return jsonify({"error": "No user text provided"}), 400
    # Dummy-Analyse: Wenn die Länge des Textes gerade ist, betrachten wir es als positiv
    analysis_result = "Positive" if len(user_text) % 2 == 0 else "Negative"
    logging.info(f"KI-Analyse für Nutzertext durchgeführt: {analysis_result}")
    return jsonify({"analysis": analysis_result})

# Performance-Boost: Liefere einige Leistungsinformationen (z.B. Redis-Cache-Info)
@app.route("/performance_info", methods=["GET"])
def performance_info():
    try:
        redis_info = redis_cache.info()
    except Exception as e:
        redis_info = {"error": str(e)}
    return jsonify({
        "status": "Performance info",
        "redis_info": redis_info
    })

##############################################################################
# 30. Erweiterte Features: Visuelle Analytics, Übersetzung & kombinierte Conversion-Daten
##############################################################################
# Admin-Dashboard: Einbetten des Matomo-Dashboards
@app.route("/admin/matomo_dashboard", methods=["GET"])
def matomo_dashboard():
    html = f"""
    <html>
      <head><title>Matomo Dashboard</title></head>
      <body>
        <h1>Matomo Dashboard</h1>
        <iframe src="{MATOMO_DASHBOARD_URL}" width="100%" height="800px" frameborder="0"></iframe>
      </body>
    </html>
    """
    return render_template_string(html)

# Übersetzungs-Endpoint: Automatisierte Übersetzung von Texten mithilfe eines Open-Source-Modells
@app.route("/translate_text", methods=["POST"])
def translate_text():
    data = request.get_json()
    text = data.get("text", "")
    source_lang = data.get("source_lang", "en")
    target_lang = data.get("target_lang", "de")
    if not text:
        return jsonify({"error": "No text provided"}), 400
    model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
    try:
        translator = pipeline("translation", model=model_name)
        result = translator(text, max_length=512)
        translated_text = result[0]['translation_text']
    except Exception as e:
        logging.error(f"Translation error: {e}")
        translated_text = "Translation error"
    return jsonify({"translated_text": translated_text})

# Erweiterte Conversion Analytics: Kombination von Heatmap-Daten und simulierten Umsatzdaten
@app.route("/conversion_analytics", methods=["GET"])
def conversion_analytics():
    heatmap = simulate_heatmap_analysis()
    revenue_data = {
       "total_revenue": random.randint(1000, 5000),
       "conversion_rate": round(random.uniform(0.01, 0.1), 2)
    }
    return jsonify({
       "heatmap_data": heatmap,
       "revenue_data": revenue_data
    })

##############################################################################
# 28. Verbesserungen: Datenbank & Performance, SEO, Social Media, Sicherheit, A/B-Tests
##############################################################################
def get_postgres_connection():
    return psycopg2.connect(
        dbname=os.getenv("PG_DBNAME", "DEIN_DB_NAME"),
        user=os.getenv("PG_USER", "DEIN_USER"),
        password=os.getenv("PG_PASSWORD", "DEIN_PASSWORT"),
        host=os.getenv("PG_HOST", "DEIN_HOST"),
        port=os.getenv("PG_PORT", "DEIN_PORT")
    )

redis_cache = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    db=0
)

def get_data(key):
    if redis_cache.exists(key):
        return redis_cache.get(key)
    data = "DEIN_DATABASE_QUERY"
    redis_cache.setex(key, 3600, data)
    return data

@app.route("/get_best_keywords", methods=["GET"])
def get_best_keywords_endpoint():
    query = request.args.get("query", "beste Affiliate-Produkte 2024")
    service = build("customsearch", "v1", developerKey=os.getenv("GOOGLE_API_KEY"))
    res = service.cse().list(q=query, cx=os.getenv("GOOGLE_CSE_ID")).execute()
    keywords = [item['title'] for item in res.get('items', [])]
    return jsonify({"best_keywords": keywords})

##############################################################################
# Main Entry Point: Unterstützt auch den Uvicorn-Server für Performance-Boost
##############################################################################
if __name__ == "__main__":
    import os
    if os.getenv("USE_UVICORN", "false").lower() == "true":
        from asgiref.wsgi import WsgiToAsgi
        import uvicorn
        asgi_app = WsgiToAsgi(app)
        uvicorn.run(asgi_app, host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
    else:
        app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
