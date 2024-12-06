import pandas as pd
import re
from fuzzywuzzy import fuzz, process


def extract_steps(text):
    sentences = re.split(r'(?<!\w.\w.)(?<![A-Z][a-z].)(?<=\.|\?)\s', str(text))
    steps = [sentence.strip() for sentence in sentences if sentence.strip()]
    return steps


def find_overlapping_steps(steps1, steps2):
    overlapping_steps = [(step1, step2) for step1 in steps1 for step2 in steps2 if fuzz.ratio(step1, step2) > 90]
    return overlapping_steps


def find_citation_context(citation_sentence, steps):
    most_similar_step = max(steps, key=lambda step: fuzz.ratio(citation_sentence, step))
    return most_similar_step

def main():
    # Read the CSV file
    df = pd.read_csv('Citing_Cited.csv', encoding='latin1', sep=';')

    # Initialize a list to store detailed results
    detailed_logs = []

    # Loop over the DataFrame rows
    for index, row in df.iterrows():
        citing_steps = extract_steps(row['citing_MaterialandMethod'])
        cited_steps = extract_steps(row['cited_MaterialandMethod'])
        citation_sentence = row['CitationSentence']

        # Debug print for intermediate steps extraction
        print(f"Record {index + 1}")
        print(f"Citing Steps: {citing_steps}")
        print(f"Cited Steps: {cited_steps}")

        # Find overlapping steps
        overlapping_steps = find_overlapping_steps(citing_steps, cited_steps)
        overlap_count = len(overlapping_steps)

        # Debug print for overlapping steps and count
        print(f"Overlapping Steps: {overlapping_steps}")
        print(f"Overlap Count: {overlap_count}\n")

        # Add the overlap steps and count to the DataFrame
        df.at[index, 'overlap_steps'] = str(overlapping_steps)
        df.at[index, 'overlap_steps_count'] = overlap_count

        # Find the most similar step in cited method
        most_similar_step = find_citation_context(citation_sentence, cited_steps)

        # Prepare detailed log for the current record
        log = []
        log.append(f"Processing record {index + 1}")
        log.append(f"Citing PMC ID: {row['citing_pmcid']}")
        log.append(f"Cited PMC ID: {row['cited_pmcid']}")
        log.append("Overlapping Steps:")
        for cite_method, cited_method in overlapping_steps:
            log.append(f"Citing Method: {cite_method}")
            log.append(f"Cited Method: {cited_method}")
            log.append('-' * 20)

        log.append(f"\nCitation Sentence: {citation_sentence}")
        log.append(f"Most Similar Step in Cited Method: {most_similar_step}")
        log.append("\n" + "=" * 20)

        detailed_logs.append("\n".join(log))

    # Save the DataFrame with new columns to the same CSV file
    df.to_csv('Citing_Cited_with_overlaps.csv', index=False, encoding='utf-8')

    # Save the detailed logs to a txt file
    with open('overlap_steps.txt', 'w', encoding='utf-8') as file:
        file.write("\n\n".join(detailed_logs))


# Execute the main function
if __name__ == "__main__":
    main()
