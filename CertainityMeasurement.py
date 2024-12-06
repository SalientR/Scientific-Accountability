import pandas as pd
from certainty_estimator.predict_certainty import CertaintyEstimator
from tqdm import tqdm

# Load your data with abstracts in columns 'R_abstract' and 'S_abstract'
try:
    df = pd.read_csv('PaperIV_Task1_data.csv', delimiter=';', on_bad_lines='skip', quoting=1, encoding='latin1')
except pd.errors.ParserError as e:
    print(f"Error reading CSV: {e}")
    exit()

# Initialize CertaintyEstimator model for sentence-level certainty
sentence_estimator = CertaintyEstimator(task='sentence-level', use_auth_token=False)

# Define function to calculate sentence-level certainty for an abstract
def get_sentence_certainty(abstract):
    if isinstance(abstract, str) and abstract.strip():
        sentences = [sent.strip() for sent in abstract.split('.') if sent.strip()]  # Split abstract into sentences
        result = sentence_estimator.predict(sentences, tqdm=tqdm)
        return sum(result) / len(result) cif result else 0  # Average certainty across sentences
    return 0

# Apply sentence-level certainty estimation to each abstract in your dataset
df['R_Abstract_Certainty'] = df['R_abstract'].apply(get_sentence_certainty)
df['S_Abstract_Certainty'] = df['S_abstract'].apply(get_sentence_certainty)

# Apply sentence-level certainty estimation to each abstract in your dataset
df['R_MM_Certainty'] = df['R_Materials_and_Methods'].apply(get_sentence_certainty)
df['S_MM_Certainty'] = df['S_Materials_and_Methods'].apply(get_sentence_certainty)

# Apply sentence-level certainty estimation to each abstract in your dataset
df['R_Result_Certainty'] = df['R_Results'].apply(get_sentence_certainty)
df['S_Result_Certainty'] = df['S_Results'].apply(get_sentence_certainty)

# Apply sentence-level certainty estimation to each abstract in your dataset
df['R_Conclusion_Certainty'] = df['R_Conclusion'].apply(get_sentence_certainty)
df['S_Conclusion_Certainty'] = df['S_Conclusion'].apply(get_sentence_certainty)


# Save the updated dataframe with the new certainty columns
df.to_csv('AllCertainty_PaperIV_Task1.csv', index=False)

print("Sentence-level certainty scores and their normalized values for abstracts added to the CSV.")
