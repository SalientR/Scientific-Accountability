import pandas as pd
from transformers import BertTokenizer, BertForNextSentencePrediction, BertModel
import torch
import spacy
import textstat
from TorturePhrases_Estimator import identify_tortured_phrases

# Read CSV file with error handling
try:
    df = pd.read_csv('PaperIV_Task1_data.csv', delimiter=',', on_bad_lines='skip', quoting=1)
except pd.errors.ParserError as e:
    print(f"Error reading CSV: {e}")
    exit()

# Initialize required models and tokenizers
tokenizer_nsp = BertTokenizer.from_pretrained('bert-base-uncased')
model_nsp = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
model_attention = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)
tokenizer_attention = BertTokenizer.from_pretrained('bert-base-uncased')
nlp = spacy.load('en_core_web_sm')

# Function to compute argument coherence
def get_nsp_score(sentence_a, sentence_b):
    encoding = tokenizer_nsp.encode_plus(sentence_a, sentence_b, return_tensors='pt')
    outputs = model_nsp(**encoding, labels=torch.LongTensor([1]))
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)
    return probabilities[0][0].item()

def compute_coherence_nsp(text):
    sentences = text.split('.')
    coherence_score = 0
    num_pairs = 0
    for i in range(len(sentences) - 1):
        if sentences[i].strip() and sentences[i + 1].strip():
            score = get_nsp_score(sentences[i].strip(), sentences[i + 1].strip())
            coherence_score += score
            num_pairs += 1
    return coherence_score / num_pairs if num_pairs > 0 else 0

def compute_coherence_cohmetrix(text):
    readability_score = textstat.flesch_reading_ease(text)
    doc = nlp(text)
    unique_words = set([token.text.lower() for token in doc if token.is_alpha])
    lexical_diversity = len(unique_words) / len(doc) if len(doc) > 0 else 0
    sentence_lengths = [len(sent) for sent in doc.sents]
    avg_sentence_length = sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0

    norm_readability = min(max((readability_score - 30) / 70, 0), 1)
    norm_lexical_diversity = min(lexical_diversity, 1)
    norm_sentence_length = min(avg_sentence_length / 20, 1)

    return (norm_readability + norm_lexical_diversity + norm_sentence_length) / 3

def extract_attention_weights(sentence_a, sentence_b):
    inputs = tokenizer_attention.encode_plus(sentence_a, sentence_b, return_tensors='pt')
    outputs = model_attention(**inputs)
    attentions = outputs.attentions
    attention_sum = torch.stack(attentions).sum(dim=0)
    avg_attention = attention_sum.mean(dim=1).detach().numpy()
    return avg_attention

def compute_attention_coherence(text):
    sentences = text.split('.')
    total_attention = 0
    num_pairs = 0
    for i in range(len(sentences) - 1):
        if sentences[i].strip() and sentences[i + 1].strip():
            attention_weights = extract_attention_weights(sentences[i].strip(), sentences[i + 1].strip())
            total_attention += attention_weights.mean()
            num_pairs += 1
    return total_attention / num_pairs if num_pairs > 0 else 0

def combined_coherence_score(text):
    if isinstance(text, str) and text.strip():  # Check if text is valid and not empty
        nsp_coherence = compute_coherence_nsp(text)
        cohmetrix_coherence = compute_coherence_cohmetrix(text)
        attention_coherence = compute_attention_coherence(text)
        attention_coherence_normalized = min(attention_coherence / 5, 1)

        final_coherence_score = (nsp_coherence + cohmetrix_coherence + attention_coherence_normalized) / 3
        return final_coherence_score
    else:
        return 0  # Return 0 for missing or invalid text

# Add the new columns to store the results

# Add the new columns to store the results
df['R_abstract_ArgumentCohere'] = df['R_abstract'].apply(combined_coherence_score)
df['S_abstract_ArgumentCohere'] = df['R_abstract'].apply(combined_coherence_score)


df['R_MM_ArgumentCohere'] = df['R_Materials_and_Methods'].apply(combined_coherence_score)
df['S_MM_ArgumentCohere'] = df['S_Materials_and_Methods'].apply(combined_coherence_score)

df['R_Results_ArgumentCohere'] = df['R_Results'].apply(combined_coherence_score)
df['S_Results_ArgumentCohere'] = df['S_Results'].apply(combined_coherence_score)

df['R_Conclusion_ArgumentCohere'] = df['R_Conclusion'].apply(combined_coherence_score)
df['S_Conclusion_ArgumentCohere'] = df['S_Conclusion'].apply(combined_coherence_score)


# Save the updated dataframe with new columns
df.to_csv('AllCoherence_PaperIV_Task1_data.csv', index=False)

print("New columns with coherence scores and tortured phrases count added.")
