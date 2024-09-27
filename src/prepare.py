import pandas as pd

# Exemple de texte annoté
texts = [
    "Je vais à <Arr>Madrid</Arr> depuis <Dep>Valencia</Dep>.",
    "J'irai de <Dep>Stockholm</Dep> à <Arr>Gothenburg</Arr>.",
    "Je vais de <Dep>Toronto</Dep> à <Arr>Montréal</Arr> la semaine prochaine.",
    # Ajoute les autres phrases ici...
]

# Fonction pour convertir le texte annoté en format tokens et labels
def convert_to_tokens_and_labels(text):
    tokens = []
    labels = []
    current_token = ""
    current_label = "O"

    for char in text:
        if char == "<":
            if current_token:
                tokens.append(current_token)
                labels.append(current_label)
                current_token = ""
            current_label = "O"
        elif char == ">":
            current_token = ""
        elif char == "/":
            current_label = "O"
        elif char.isalpha() or char.isspace():
            current_token += char
        else:
            if current_token:
                tokens.append(current_token)
                labels.append(current_label)
                current_token = ""
            tokens.append(char)
            labels.append("O")

        if "<Dep>" in current_token:
            current_label = "DEP"
            current_token = ""
        elif "<Arr>" in current_token:
            current_label = "ARR"
            current_token = ""

    if current_token:
        tokens.append(current_token)
        labels.append(current_label)

    return tokens, labels

if __name__ == "__main__":
    # Convertir les textes en format tokens et labels
    data = []
    for text in texts:
        tokens, labels = convert_to_tokens_and_labels(text)
        data.append((tokens, labels))

    # Convertir en DataFrame
    df = pd.DataFrame(data, columns=["tokens", "labels"])
    print(df)
