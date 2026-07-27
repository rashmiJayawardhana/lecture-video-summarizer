import json

with open('src/module2_summarization/final_output/LecVideo_041_-_CS50x_2024_-_Artificial_Intelligence_module2_final.json') as f:
    data = json.load(f)

print(f'Total segments: {len(data)}')
print()
print('=== FIRST 3 SEGMENTS ===')

for seg in data[:3]:
    print(f'Segment    : {seg["segment_id"]}')
    print(f'Time       : {seg["timestamp_start"]}s to {seg["timestamp_end"]}s')
    print(f'Importance : {seg["importance_ratio_T"]}')
    print(f'Sentences  :')
    for s in seg["sentences"]:
        status = "IMPORTANT" if s["is_important"] else "not important"
        print(f'  [{status}]')
        print(f'  Text      : {s["sentence"][:80]}')
        print(f'  Confidence: {s["confidence"]}')
        print(f'  Keyword   : {s["keyword_boost"]}')
        print(f'  Definition: {s["definition_match"]}')
        print()
    print('-' * 50)