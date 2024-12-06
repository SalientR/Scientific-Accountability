# Scientific Accountability: Detecting Salient Features of Retracted Articles

This repository provides the supplementory material for the paper titled **"Scientific Accountability: Detecting Salient Features of Retracted Articles."** The work focuses on analyzing retracted and non-retracted articles by detecting salient features and classifying articles based on various text and measurable features.







## File Descriptions
BertBasedClassification.py: Implements BERT-based classification models to distinguish between retracted and non-retracted articles based on textual features and calculates their performance metrics.

CertainityMeasurement.py: Contains functionality for measuring certainty levels in scientific reporting, assessing individual sentences, and aggregating scores across relevant sections.

ReadabilityMeasurement.py: Calculates readability scores for scientific articles using various metrics, providing insights into the clarity and accessibility of the text.

ConventionalModelClassification_Ranking.py: Applies conventional machine learning models to classify articles, comparing performance based on accuracy, precision, recall, and F1 score.

ImplicitDependencies_1.py: Identifies implicit dependencies between citing and cited retracted articles, highlighting similarities in methodological protocols.

ImplicitDependencies_2.py: Further investigates implicit dependencies using different methodologies and comparison approaches.

Retracted_NonRetracted Articles.csv: A dataset containing information on retracted and non-retracted articles, including relevant sections (Abstract, Methodology, Results, Conclusion), readability scores, certainty levels, and labels.


## Requirements

To run the provided code, ensure you have the following installed:

- **Python**: Version 3.6 or later
- **Libraries**: 
  - `pandas`
  - `numpy`
  - `torch` (PyTorch)
  - `transformers` (Hugging Face Transformers library)
  - `fuzzywuzzy`
  - `scikit-learn`
  - `textstat`
  - `spacy`

You can install the necessary libraries using pip:
```bash
pip install pandas numpy torch transformers fuzzywuzzy scikit-learn textstat spacy
```



## Instructions for Use

Clone this repository to your local machine using the following commands:

```bash
git clone https://github.com/SalientR/
cd Scientific-Accountability
```


https://github.com/SalientR/Scientific-Accountability/edit/main/README.md
