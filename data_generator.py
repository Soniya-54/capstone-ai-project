import pandas as pd
import random

# NEW: Categorical vocabularies are now totally isolated to prevent "Bleed"
vocab = {
    "Infrastructure": {
        "s": ["pothole on road", "street lights", "bridge support", "cracked sidewalk", "water pipe leak", "traffic signal"],
        "a": ["is crumbling", "needs urgent repair", "is breaking apart", "is dangerous for cars", "is gushing water"],
        "l": ["on the main highway", "at the intersection", "in the downtown area", "near the square"]
    },
    "Healthcare": {
        "s": ["blood pressure", "medical emergency", "heart rate", "insulin stock", "ambulance response", "clinic hygiene"],
        "a": ["is dangerously low", "is dropping fast", "needs a doctor immediately", "is critical condition", "is out of stock"],
        "l": ["at the city hospital", "in the medical ward", "near the health clinic", "at the ER"]
    },
    "Sanitation": {
        "s": ["trash accumulation", "garbage pile", "raw sewage", "dead animal carcass", "illegal dumping", "recycling bin"],
        "a": ["is rotting and stinking", "is overflowing onto street", "is leaking into drains", "is attracting rats"],
        "l": ["outside my house", "behind the school", "near the park", "in the residential alley"]
    },
    "Administration": {
        "s": ["passport processing", "birth certificate", "tax documents", "official website", "clerk behavior", "permit application"],
        "a": ["is stuck for months", "requested a bribe", "keeps crashing", "is full of errors", "is being delayed"],
        "l": ["at town hall", "in the records office", "on the government portal", "at city hall"]
    },
    "Public Safety": {
        "s": ["violent kidnapping", "gang fighting", "theft in progress", "gunshots fired", "suspicious loitering", "illegal racing"],
        "a": ["is a threat to life", "happened just now", "is out of control", "reported last night", "is extremely scary"],
        "l": ["near Birendra Chowk", "at the train station", "in the community park", "outside the mall"]
    },
    "General": {
        "s": ["The weather", "City gardens", "Website layout", "The parade", "Community festival"],
        "a": ["is very nice", "looks beautiful", "is helpful", "is interesting", "is a great event"],
        "l": ["today", "this morning", "in the park", "for everyone"]
    }
}

def build_refined_text(cat):
    comp = vocab[cat]
    # Keep the phrase structure tight to build strong features
    text = f"{random.choice(comp['s'])} {random.choice(comp['a'])} {random.choice(comp['l'])}."
    if random.random() > 0.7:
        text = f"{random.choice(['Urgent:', 'Attention:', 'Report:'])} {text}"
    return text

final_data = []
for cat in vocab.keys():
    for _ in range(500):
        # 2% Noise only - we need high signal now
        label = cat if random.random() > 0.02 else random.choice(list(vocab.keys()))
        final_data.append({"complaint_text": build_refined_text(cat), "category": label})

df = pd.DataFrame(final_rows if 'final_rows' in locals() else final_data)
df = df.sample(frac=1).reset_index(drop=True)
df.to_csv("robust_complaints_data.csv", index=False)
print("SUCCESS: High-Signal Dataset Created.")