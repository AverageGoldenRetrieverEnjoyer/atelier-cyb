import json
from elasticsearch import Elasticsearch

# --- Configuration ---
# !! Replace with your elastic password
ELASTIC_PASSWORD = "Dhjzk76N3N0%3"

# The T-Pot host (running locally on the server)
ELASTIC_HOST = "10.0.0.1"

# The index pattern you want to dump (e.g., "logstash-*")
INDEX_NAME = "logstash-*"

# The file to save all the data to
OUTPUT_FILE = "tpot_data_dump.json"

# How many docs to get per batch (1000 is a good default)
BATCH_SIZE = 1000

# How long to keep the "scroll" open between requests
SCROLL_TIME = "2m"
# ---------------------

print(f"Connecting to http://{ELASTIC_HOST}:9200...")
try:
    # Authenticate with the 'elastic' user and your password
    es = Elasticsearch(
        [f"http://{ELASTIC_HOST}:9200"],
        basic_auth=("elastic", ELASTIC_PASSWORD)
    )
    
    # Verify connection
    if not es.ping():
        raise ValueError("Connection failed. Check credentials and host.")
        
    print("Connection successful!")

except Exception as e:
    print(f"Error connecting to Elasticsearch: {e}")
    print("Please check your ELASTIC_PASSWORD and network access.")
    exit()

# List to hold all the documents
all_docs = []

print(f"Starting data dump from index '{INDEX_NAME}'...")

try:
    # 1. Start the initial search and create the scroll
    resp = es.search(
        index=INDEX_NAME,
        query={"match_all": {}},
        size=BATCH_SIZE,
        scroll=SCROLL_TIME
    )

    # Get the scroll ID and the first batch of hits
    scroll_id = resp.get('_scroll_id')
    hits = resp['hits']['hits']
    
    if not scroll_id:
        print("Error: Could not start scroll. Are there documents in the index?")
        exit()

    doc_count = 0

    # 2. Loop as long as we have a scroll_id and are getting hits
    while scroll_id and hits:
        # Add the current batch of hits to our list
        all_docs.extend(hits)
        
        doc_count += len(hits)
        print(f"Fetched {doc_count} total documents...")

        # 3. Call the scroll API to get the NEXT batch of hits
        resp = es.scroll(
            scroll_id=scroll_id,
            scroll=SCROLL_TIME
        )

        # Update the scroll ID (it can change) and the hits
        scroll_id = resp.get('_scroll_id')
        hits = resp['hits']['hits']

    print(f"\nDump complete. Total documents retrieved: {doc_count}")

    # 4. Clean up the scroll context on the server
    if scroll_id:
        print("Clearing scroll context...")
        es.clear_scroll(scroll_id=scroll_id)

    # 5. Save all collected data to the output file
    print(f"Saving data to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        # Use a list comprehension to save *only* the '_source' field
        # This is what you'll use for machine learning
        source_data = [doc['_source'] for doc in all_docs]
        json.dump(source_data, f, indent=2)

    print("All done!")

except Exception as e:
    print(f"\nAn error occurred during the dump: {e}")
    if scroll_id:
        es.clear_scroll(scroll_id=scroll_id)
        print("Scroll context cleared.")