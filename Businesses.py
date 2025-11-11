# ✅ Optimized RAG: Dynamic Review Retriever & GPT Responder

import gzip
import pandas as pd
import boto3
import json
from io import BytesIO
from openai import OpenAI

# 🔐 OpenAI Setup
client = OpenAI(api_key="YOUR_API_KEY")  # Replace with actual key

# 🪣 AWS S3 Setup
s3 = boto3.client("s3")
bucket = "kiran-yelp-project-bucket"
prefix = "rag_combined/"

response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
all_files = [obj["Key"] for obj in response["Contents"] if obj["Key"].endswith(".gz")]

# 📦 Batch Loader for Large Yelp Files
def load_json_batches(file_keys, batch_size=1, max_lines=50000000):
    for i in range(0, len(file_keys), batch_size):
        batch = file_keys[i:i + batch_size]
        records = []
        for key in batch:
            obj = s3.get_object(Bucket=bucket, Key=key)
            with gzip.open(BytesIO(obj["Body"].read()), mode="rt", encoding="utf-8") as f:
                for j, line in enumerate(f):
                    if j >= max_lines:
                        break
                    try:
                        json_line = json.loads(line)
                        nested_record = json.loads(json_line["json_record"])
                        records.append({
                            "city": nested_record.get("city", ""),
                            "categories": nested_record.get("categories", ""),
                            "review_text": nested_record.get("review_text", ""),
                            "review_stars": nested_record.get("review_stars", 0)
                        })
                    except Exception:
                        pass
        df_batch = pd.DataFrame(records)
        yield df_batch

# 🏙️ Category + City Filter
def filter_reviews(df, city, category, limit=10):
    city = city.lower().strip()
    category = category.lower().strip()

    df = df.copy()
    df = df[df['city'].notnull() & df['categories'].notnull()]
    df['city_clean'] = df['city'].str.lower().str.strip()
    df['categories_clean'] = df['categories'].str.lower().str.strip()

    filtered = df[
        df['city_clean'].apply(lambda x: city in x or x in city) &
        df['categories_clean'].apply(lambda x: category in x)
    ]

    print(f"🔍 Searching for reviews in city: {city}, category: {category}...")
    print(f"✅ Found {len(filtered)} matching reviews.")
    return filtered.sort_values(by='review_stars', ascending=False).head(limit)

# ✍️ GPT Prompt Constructor
def build_prompt(question, reviews):
    bullet_reviews = "\n".join([f"- {r.lower()}" for r in reviews['review_text'].tolist()])
    return f"""
you're a helpful assistant. follow these rules while responding:

1. only respond to questions related to businesses present in the yelp dataset.
2. if the question is unrelated to yelp businesses, politely inform the user that only business-related questions can be answered.
3. use a simple, conversational, and human-like tone in all responses.
4. format recommendations in a clean and consistent format without bolding or starring names.
5. include emojis in bullet points when listing businesses (optional but preferred).
6. if the question is unclear or confusing, respond sarcastically but politely, asking the user to rephrase and giving example prompts.
7. do not provide information or speculation on non-business topics.
8. avoid using complex or overly technical language.
9. provide brief and clear answers, unless more detail is specifically requested.
10. end responses with a friendly and open-ended phrase encouraging further questions.
11. never make up data—only respond using information retrievable from the yelp dataset.
12. do not answer questions that involve personal opinions outside the dataset context.
13. keep answers unbiased and neutral—do not rank unless backed by ratings or relevant data.
14. do not answer questions related to politics, health, finance, or anything beyond the business scope of yelp.
15. maintain consistency in formatting across all responses.

customer asked: \"{question.lower()}\"

based on these reviews, provide a professional and insightful answer in bullet points:

{bullet_reviews}

answer:
"""

# 🤖 OpenAI API Caller
def query_gpt(prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content

# 🚀 Run End-to-End
if __name__ == "__main__":
    city = input("Enter city: ")
    category = input("Enter category: ")
    question = input("Ask your question: ")

    for df_batch in load_json_batches(all_files, batch_size=1, max_lines=25000):
        filtered = filter_reviews(df_batch, city, category)

        if not filtered.empty:
            prompt = build_prompt(question, filtered)
            print("🤖 Asking GPT...")
            response = query_gpt(prompt)
            print("\n📢 GPT Response:\n")
            print(response)
            break
    else:
        print("⚠️ No matching reviews found across any batch.")
