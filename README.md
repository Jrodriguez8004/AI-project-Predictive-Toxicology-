# AI-project-Predictive-Toxicology-
Project Overview
Early prediction of drug toxicity is a critical challenge in drug discovery and development. This project focuses on using machine learning algorithms to analyze molecular structures and predict potential toxic effects. The project is mentored by Johan Rodriguez.

Problem Statement
Determining a drug's toxicity traditionally requires extensive laboratory testing and clinical trials, which are resource-intensive and costly. This project aims to develop a predictive model that identifies potentially toxic compounds early in the development process, saving resources and reducing the likelihood of adverse reactions in later trials.

Objectives
Develop a machine learning pipeline for toxicity prediction.
Test various algorithms (Random Forest, Support Vector Machine, Neural Networks) for optimal predictive accuracy.
Utilize datasets from PubChem and Tox21 for training and testing.
Validate the model's performance against benchmark datasets.
Methodology
The project involves:

Data Collection: Gathering data from PubChem and Tox21 on compounds and their toxicology profiles.
Data Preprocessing: Cleaning, transforming, and feature extraction from chemical structures.
Model Training: Implementing various ML models to classify compounds as toxic or non-toxic.
Evaluation: Measuring model performance using metrics like accuracy, F1-score, precision, and recall.
Interpretation: Analyzing model output to provide insights into toxicological characteristics of compounds.
Data
Data sources for this project include PubChem and Tox21, which offer comprehensive chemical information and toxicity data for a wide range of compounds.

Dependencies
This project requires the following packages:

Python 3.x
NumPy
Pandas
Scikit-Learn
RDKit (for chemical structure processing)
Matplotlib / Seaborn (for visualizations)
Jupyter Notebook (for interactive experimentation)
To install the dependencies, you can run:


pip install -r requirements.txt
Usage
Clone the repository:

git clone https://github.com/0yeaman/AI-in-Predictive-Toxicology.git
Navigate to the project directory:

cd AI-in-Predictive-Toxicology
Run the Jupyter notebook to train models and analyze results:

jupyter notebook
Results
The model demonstrates strong predictive performance, showing its potential as a preclinical toxicity screening tool. Initial test results include:

Metric	Value
Accuracy	85%
Precision	82%
Recall	83%
F1-Score	82%
These metrics indicate the model’s reliability in detecting toxic compounds, with potential for further improvement as more data is integrated.

Future Work
Expand Dataset: Incorporate additional chemical and toxicology databases.
Feature Engineering: Experiment with advanced descriptors and feature extraction techniques.
Deep Learning: Implement and test neural networks for improved accuracy.
Web Interface: Develop a web-based tool for easy toxicity prediction by end-users.
Contributors
Amandeep Singh - Big Data Analytics student at Lambton College
Project Mentor - Johan Rodriguez
For any questions, please contact Amandeep Singh.
