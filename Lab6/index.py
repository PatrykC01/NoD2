import pandas as pd
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import gzip
import json

DATA_URLS = {
    "Electronics_5.json.gz": "https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Electronics_5.json.gz",
    "Clothing_Shoes_and_Jewelry_5.json.gz": "https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Clothing_Shoes_and_Jewelry_5.json.gz",
    "Books_5.json.gz": "https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Books_5.json.gz"
}

print("--- SPRAWDZANIE PLIKÓW ---")
for filename, url in DATA_URLS.items():
    if not os.path.exists(filename):
        print(f"Pobieranie {filename}...")
        os.system(f"wget -q --no-check-certificate {url}")
    else:
        print(f"Plik {filename} gotowy.")

FILES_CONFIG = {
    "Electronics_5.json.gz": 0,                # Kategoria: Elektronika
    "Clothing_Shoes_and_Jewelry_5.json.gz": 1, # Kategoria: Odzież
    "Books_5.json.gz": 2                       # Kategoria: Książki
}

CLASS_NAMES = {0: "Elektronika", 1: "Odzież", 2: "Książki"}

SAMPLES_PER_CLASS = 6000 
MAX_VOCAB_SIZE = 10000
MAX_SEQ_LEN = 100
EMBEDDING_DIM = 64

# === KROK 1: Wczytywanie danych ===
def load_data_from_gzip(files_map, samples_n):
    texts = []
    labels = []
    
    for filename, label_id in files_map.items():
        print(f"Wczytywanie: {filename}...")
        count = 0
        try:
            with gzip.open(filename, 'rt', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        text_content = data.get('reviewText', data.get('summary', ''))
                        
                        if text_content and len(text_content) > 20: 
                            texts.append(text_content)
                            labels.append(label_id)
                            count += 1
                        
                        if count >= samples_n:
                            break
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            print(f"BŁĄD: Nie znaleziono pliku {filename}!")
            continue 

    if len(set(labels)) < 3:
        found = list(set(labels))
        raise ValueError(f"BŁĄD KRYTYCZNY: Znaleziono tylko kategorie: {found}. Wymagane są 3 (0, 1, 2). Sprawdź pliki!")
   
    return np.array(texts), np.array(labels)

print("\n--- GENEROWANIE DATASETU ---")
X_raw, y_raw = load_data_from_gzip(FILES_CONFIG, SAMPLES_PER_CLASS)

# === KROK 2: Preprocessing ===
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(X_raw)
sequences = tokenizer.texts_to_sequences(X_raw)
X_padded = pad_sequences(sequences, maxlen=MAX_SEQ_LEN, padding='post', truncating='post')

X_train, X_test, y_train, y_test = train_test_split(
    X_padded, y_raw, test_size=0.2, random_state=42, stratify=y_raw
)

# === KROK 3: Model BRNN ===
input_layer = Input(shape=(MAX_SEQ_LEN,))
x = Embedding(input_dim=MAX_VOCAB_SIZE, output_dim=EMBEDDING_DIM)(input_layer)

x = Bidirectional(LSTM(64))(x)

x = Dense(64, activation='relu')(x)
x = Dropout(0.5)(x)
output_layer = Dense(3, activation='softmax')(x) 

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# === KROK 4: Trening ===
print("\n--- ROZPOCZĘCIE TRENINGU ---")
history = model.fit(
    X_train, y_train, 
    epochs=15, 
    batch_size=32, 
    validation_data=(X_test, y_test), 
    callbacks=[early_stop]
)

# === KROK 5: Test zgodności ===
print("\n--- WERYFIKACJA NA WŁASNYCH ZDANIACH ---")
test_sentences = [
    "I bought this cotton hoodie because I needed something comfortable for cooler evenings, and it turned out better than expected. The inside is soft and warm without feeling too heavy, and the stitching looks surprisingly durable for the price. After two washes it hasn’t faded or lost its shape, which is rare for clothing ordered online. The only downside is that the sleeves are slightly longer than I’d prefer, but overall it’s a solid piece of clothing I’ve been wearing almost daily.",

    "This novel completely drew me in from the first chapter. The author has a talent for building emotional tension, and the characters feel incredibly lifelike, with believable flaws and motivations. Some sections move slowly, but they ultimately contribute to a much richer narrative. By the final pages I felt genuinely attached to the protagonist, and the ending left me thinking about the story long after I closed the book.",

    "I’ve been using this blender for a few weeks now, and it has genuinely simplified my morning routine. It handles frozen fruit, nuts, and ice without any trouble, producing smooth, consistent blends every time. Cleaning it is easy, thanks to the detachable parts that rinse quickly under warm water. The only issue is that it gets a bit loud at full power, but aside from that it has been a reliable addition to my kitchen.",

    "These running shoes looked great in the photos, but I was even more impressed once I tried them on. The cushioning is soft yet responsive, making long walks much more comfortable than with my previous pair. They provide good support without feeling tight, and the breathable material really helps during warmer days. I can’t speak for long-term durability yet, but so far they’ve exceeded my expectations.",

    "This fantasy book had an incredibly rich world with detailed lore and well-thought-out history. The pacing is uneven at times, but the writing style is smooth and engaging. I particularly loved how the author explores moral dilemmas through the plot rather than forcing them into dialogue. If you enjoy immersive worlds and complex political intrigue, this is definitely worth reading.",

    "The kitchen scale works exactly as advertised, providing precise measurements down to the gram. I use it daily for baking, and it has made my recipes far more consistent. The display is clear and bright, and the device automatically shuts off if left unused, saving battery life. It feels sturdy despite its sleek design and has become an essential tool for my cooking routine.",

    "I purchased this winter jacket hoping it would hold up against cold wind, and it absolutely delivered. The outer material blocks wind well, and the inner lining keeps warmth in without making the jacket feel bulky. I appreciate the deep pockets and adjustable hood, which are both highly functional. My only complaint is that the zipper feels a bit flimsy, but so far it hasn’t caused any problems.",

    "This mystery novel started off fairly ordinary, but by the halfway point I couldn’t put it down. Each chapter ends with just enough suspense to keep you reading, and the plot twists feel justified rather than forced. The protagonist is flawed but relatable, and the writing avoids clichés common in the genre. It’s been a long time since a book kept me up past midnight, but this one certainly did.",

    "This ceramic baking dish has quickly become one of my favorite kitchen items. It heats evenly, making casseroles and roasted vegetables come out perfectly each time. Cleaning it is simple, even when something sticks — a short soak is enough to remove residue. The handles are wide and easy to grip, which makes it much safer when removing hot dishes from the oven.",

    "The pair of jeans I received fits better than expected, especially around the waist and hips. The fabric has just the right amount of stretch, making them comfortable for long periods of sitting or walking. After wearing them for a few days, I noticed no sagging or loose threads, which is often a problem with cheaper denim. I’m genuinely impressed with the quality and will probably buy another pair soon.",

    "This vacuum cleaner surprised me with how quietly it operates compared to my old one. Despite the low noise, it has strong suction power and easily picks up pet hair from carpets and hardwood floors. The detachable handheld unit is incredibly convenient for stairs and corners. So far it feels like a huge upgrade that makes cleaning much less of a chore.",

    "I read this historical biography hoping for new insight into a figure I already admired, and the author delivered beautifully. The book balances personal stories with broader historical context, making it both informative and emotionally compelling. The pacing remains steady throughout, and the narrative never feels dry. It’s a great example of non-fiction writing that remains accessible without sacrificing depth.",

    "These leather gloves look stylish and provide excellent insulation. The interior lining feels soft, and the gloves stay warm even in freezing temperatures. They fit snugly without restricting movement, which is essential for driving. I’m pleasantly surprised by the craftsmanship and expect them to last for several seasons.",

    "This electric kettle boils water incredibly fast, saving me time during hectic mornings. The stainless steel body feels solid, and the handle stays cool even when the water reaches its boiling point. I appreciate the automatic shut-off feature for safety. Overall, it’s a practical, reliable appliance I now use multiple times a day.",

    "The sci-fi novel I finished yesterday had one of the most thought-provoking plots I’ve read in a long time. The futuristic technology is described in a way that feels plausible, and the character interactions are surprisingly grounded. While the middle section drags slightly, the final third is gripping and emotionally impactful. It’s the type of book you want to recommend immediately" 
]

test_labels = np.array([1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2, 1, 0, 2])

seq = tokenizer.texts_to_sequences(test_sentences)
pad = pad_sequences(seq, maxlen=MAX_SEQ_LEN, padding='post', truncating='post')
preds = model.predict(pad)

preds_class = np.argmax(preds, axis=1)
acc = np.mean(preds_class == test_labels)
print(f"\n>>> Dokładność na zbiorze testowym: {acc*100:.2f}% <<<\n")

for i, txt in enumerate(test_sentences):
    pred_idx = preds_class[i]
    true_idx = test_labels[i]
    confidence = np.max(preds[i]) * 100
    
    short_txt = txt[:180] + "..." if len(txt) > 180 else txt
    
    print(f"Tekst: {short_txt}")
    
    if pred_idx == true_idx:
        status = "✅ POPRAWNIE"
    else:
        status = f"❌ BŁĄD (Oczekiwano: {CLASS_NAMES[true_idx]})"
        
    print(f"Predykcja: {CLASS_NAMES[pred_idx]} ({confidence:.2f}%) | {status}")
    print("-" * 80)
