with open('src/module2_summarization/transcribe_all.py', 'r', encoding='utf-8') as f:
    content = f.read()

new_content = content.replace('"base"', '"large-v3"')

with open('src/module2_summarization/transcribe_all.py', 'w', encoding='utf-8') as f:
    f.write(new_content)

print("Done! Changed to large-v3!")

with open('src/module2_summarization/transcribe_all.py', 'r', encoding='utf-8') as f:
    check = f.read()

if 'large-v3' in check:
    print("Verified! large-v3 is set!")
else:
    print("Failed - try again!")