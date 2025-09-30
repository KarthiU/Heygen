from transformers import MarianMTModel, MarianTokenizer

# Load SRT file and extract subtitle text
def read_srt_file(path):
    subtitles = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Ignore empty lines, index numbers, and timestamps
            if not line or line.isdigit() or "-->" in line:
                continue
            subtitles.append(line)
    return subtitles

# Example usage
srt_path = "samples/Tanzania-caption.srt"  # replace with your file
text = read_srt_file(srt_path)

# Translation setup
model_name = "Helsinki-NLP/opus-mt-en-de"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
translated = model.generate(**inputs)
outputs = tokenizer.batch_decode(translated, skip_special_tokens=True)

print(outputs)
