import json

mapping = {
           }



def translate_sentence(sentence: str) -> str:
    words = sentence.split()
    result_words = []
    for w in words:
        clean_w = w.strip(".,!?;:\"'()[]{}")
        if clean_w in mapping:
            result_words.append(mapping[clean_w])
        else:
         
            result_words.append("[UNK]")
    return " ".join(result_words)

try:
    with open("test_data.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)
except FileNotFoundError:
    test_data = []

answers = []
for item in test_data:
    source = item.get("source", "")
    translation = translate_sentence(source)
    answers.append({
        "source": source,
        "translation": translation
    })

with open("submission.json", "w", encoding="utf-8") as f:
    json.dump(answers, f, ensure_ascii=False, indent=2)
