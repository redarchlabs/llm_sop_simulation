[query: "good email replies"]
       ↓
[search_vector_db → get 50+ email chunks]
       ↓
[for each chunk:]
  └─ check_is_reply(chunk)
       └─ if True → add to final list
       └─ if False → skip
       ↓
[summarize(filtered_chunks)]
       ↓
[output top 3 with explanation]
