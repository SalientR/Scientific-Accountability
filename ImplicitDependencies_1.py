import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load the CSV file into a DataFrame
df = pd.read_csv('Citing_Cited.csv', encoding='latin1', sep=';')

# Initialize the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased').to('cuda' if torch.cuda.is_available() else 'cpu')


def get_embeddings(texts, tokenizer, model, batch_size=8):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
        inputs = {key: value.to(model.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.extend(batch_embeddings)
    return embeddings


# Compute embeddings for citing and cited methodology text
df['citing_embeddings'] = get_embeddings(df['citing_MaterialandMethod'].tolist(), tokenizer, model)
df['cited_embeddings'] = get_embeddings(df['cited_MaterialandMethod'].tolist(), tokenizer, model)


# Compute cosine similarity between citing and cited embeddings
def compute_cosine_similarity(row):
    citing_emb = np.array(row['citing_embeddings'], dtype=np.float32).reshape(1, -1)
    cited_emb = np.array(row['cited_embeddings'], dtype=np.float32).reshape(1, -1)
    return cosine_similarity(citing_emb, cited_emb)[0][0]


df['cosine_similarity'] = df.apply(compute_cosine_similarity, axis=1)

# Drop the embedding columns (optional, for cleanliness)
df = df.drop(columns=['citing_embeddings', 'cited_embeddings'])

# Save the DataFrame with the new column to a new CSV file
df.to_csv('data_with_similarity.csv', index=False)

print(df)
