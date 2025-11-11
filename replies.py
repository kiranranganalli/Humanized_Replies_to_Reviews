# GPT-based Auto Reply Generator for Yelp Reviews

import boto3
import pandas as pd
import openai
from io import StringIO
import urllib.parse
import os

# Configuration
api_key = "sk-..."  # Replace with your actual API key
bucket_name = "kiran-yelp-project-bucket"
s3_folder = "category_files/"
csv_files = [
    "american (traditional).csv",
    "bars.csv",
    "coffee & tea.csv",
    "Food.csv",
    "gyms.csv",
    "nightlife.csv",
    "restaurants.csv",
    "sandwiches.csv"
]

# GPT response function
def generate_reply(review_text, category, client):
    prompt = f"""
You're the owner of a business in the {category} category. Write a professional and polite response to this customer review:

"{review_text}"

Response:
"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content

# Main processing function
def process_files():
    s3 = boto3.client("s3")
    client = openai.OpenAI(api_key=api_key)

    for file_name in csv_files:
        category = os.path.splitext(file_name)[0]  # Get category name
        input_key = f"{s3_folder}{file_name}"
        output_key = f"{s3_folder}gpt_{file_name}"

        print(f"\n📥 Processing: {file_name}")

        # Read file from S3
        try:
            response = s3.get_object(Bucket=bucket_name, Key=input_key)
            df = pd.read_csv(response["Body"])
            print(f"✅ Loaded {len(df)} reviews")
        except Exception as e:
            print(f"❌ Failed to load {file_name}: {e}")
            continue

        # Generate GPT replies
        try:
            df["gpt_response"] = df["review_text"].astype(str).apply(lambda x: generate_reply(x, category, client))
        except Exception as e:
            print(f"❌ Error generating responses for {file_name}: {e}")
            continue

        # Save result to S3
        try:
            csv_buffer = StringIO()
            df.to_csv(csv_buffer, index=False)
            s3.put_object(Bucket=bucket_name, Key=output_key, Body=csv_buffer.getvalue())
            print(f"✅ Saved GPT responses to: {output_key}")
        except Exception as e:
            print(f"❌ Failed to save output for {file_name}: {e}")

# Run
if __name__ == "__main__":
    process_files()
