#!/usr/bin/env python3
"""Test script to verify NewsAPI connection and FinBERT sentiment analysis."""

import os
import requests
from app import finbert_sentiment_score, FINBERT_AVAILABLE, NEWSAPI_KEY

print("=" * 60)
print("NewsAPI + FinBERT Test")
print("=" * 60)

# Check NEWSAPI_KEY
print(f"\n1. NEWSAPI_KEY check:")
print(f"   Key is set: {bool(NEWSAPI_KEY)}")
if NEWSAPI_KEY:
    print(f"   Key length: {len(NEWSAPI_KEY)}")
    print(f"   Key preview: {NEWSAPI_KEY[:10]}...")
else:
    print("   ⚠️  NEWSAPI_KEY not set. Set it with: export NEWSAPI_KEY='your_key'")

# Check FinBERT
print(f"\n2. FinBERT check:")
print(f"   FinBERT available: {FINBERT_AVAILABLE}")

# Test NewsAPI connection
if NEWSAPI_KEY:
    print(f"\n3. Testing NewsAPI connection:")
    try:
        url = "https://newsapi.org/v2/top-headlines"
        params = {
            'country': 'in',
            'apiKey': NEWSAPI_KEY,
            'pageSize': 5
        }
        response = requests.get(url, params=params, timeout=10)
        print(f"   Status code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✓ Success! Found {data.get('totalResults', 0)} articles")
            articles = data.get('articles', [])
            if articles:
                print(f"   Sample article: {articles[0].get('title', 'N/A')[:60]}...")
        else:
            print(f"   ✗ Error: {response.status_code}")
            print(f"   Response: {response.text[:200]}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
else:
    print(f"\n3. Skipping NewsAPI test (no API key)")

# Test FinBERT sentiment
if FINBERT_AVAILABLE:
    print(f"\n4. Testing FinBERT sentiment:")
    test_texts = [
        "Stock prices surge as company reports record profits",
        "Market crashes amid economic uncertainty",
        "Company announces quarterly earnings meeting expectations"
    ]
    for text in test_texts:
        score = finbert_sentiment_score(text)
        print(f"   '{text[:40]}...' → {score:.3f}")
else:
    print(f"\n4. FinBERT not available, skipping sentiment test")

print("\n" + "=" * 60)
print("Test complete!")
print("=" * 60)

