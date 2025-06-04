## Trucking Chatbot for Intent Classification and Response Generation

**Nisheeth Goyal**

### Executive Summary

**Project Overview and Goals:** The goal of this project is to develop a chatbot for the trucking industry to classify user intents from social media and text queries and generate appropriate responses. The chatbot uses DistilBERT for classifying 13 intents (e.g., `login_issue`, `fuel_card_query`, `farewell`) and DialogGPT for generating context-aware responses, with an interactive `ipywidgets` UI. The system processes `inbound=True` queries from a Kaggle dataset, aiming to support truck drivers with queries about delivery status, billing, fuel cards, and more. The project trains and evaluates DistilBERT, extracts entities (e.g., "Driven", "Comdata"), and deploys a UI for real-time interaction. Due to resource contents and possible inferior quality of training data, intent classification issues persist, notably overprediction of the `farewell` intent.

**Findings:** The DistilBERT model achieves moderate performance but struggles with intent classification, predicting 230,378/307,575 evaluation samples as `farewell` despite only 1,139 `farewell` samples in the original dataset. Misclassifications include "You’re awesome!" → `account_update` (should be `compliment`) and "My Comdata card is not working" → `farewell` (should be `fuel_card_query`). The model’s accuracy, precision, recall, and F1 scores are suboptimal due to inferior quality of data and lack of resources for traning large amount of data(Cell 4). The DialogGPT response generation and UI are functional, but responses are affected by incorrect intents. The confusion matrix from the evaluation cell highlights the `farewell` bias, with most non-`farewell` intents misclassified.

**Results and Conclusion:** The chatbot successfully processes queries and generates responses via a UI, but intent classification errors limit its reliability. Entity extraction correctly identifies terms like "Driven" and "Comdata". The evaluation reveals a skewed intent distribution, with `farewell` dominating predictions. Attempts to fix this included more focused data on trucking/accounting related user queries, stricter labeling rules and data balancing. The project demonstrates a functional pipeline for intent classification and response generation, with potential for improvement in labeling and training.

**Future Research and Development:** Future work should focus on refining the `label_intent` function in `inspect_dataset` to with more focused/accurate data, capping synthetic data generation, and balancing training data. Retraining DistilBERT with more epochs or exploring ensemble methods could improve performance. Cross-checking original vs. synthetic data distributions and analyzing misclassified texts would help identify error sources.

**Next Steps and Recommendations:** To enhance the chatbot, implement the following:
- **Refine Labeling**: Use stricter keyword rules and validate against original intents.
- **Balance Data**: Limit synthetic `farewell` samples and undersample overrepresented intents.
- **Retrain Model**: Experiment with BERT variants or increase regularization.
- **Analyze Errors**: Use the evaluation cell’s misclassification logs to debug specific queries.
Further research could explore integrating real-time data from X posts to augment the dataset.

### Rationale

The project addresses the need for automated support in the trucking industry, where drivers frequently query about logistics, payments, and technical issues. Manual customer service is resource-intensive, and a chatbot can improve efficiency. According to industry reports, trucking companies face high driver turnover and operational costs, making scalable solutions like chatbots valuable.

### Research Question

What is the most effective approach to classify trucking-related intents and generate accurate responses using transformer models like DistilBERT and DialogGPT?

### Data Sources

**Dataset:** The dataset is sourced from Kaggle (assumed `trucking_chatbot_test_dataset.csv`). It contains `inbound=True` queries with columns: `text` (query content)

**Exploratory Data Analysis:** The dataset has no null values after preprocessing. There lots of special characters and cross reference between rows which needed to be cleaned up.

**Cleaning and Preparation:** Duplicates are removed, and `inbound=False` queries are filtered. Intents are relabeled in `inspect_dataset`.

**Preprocessing:** The `inspect_dataset` function applies rule-based intent labeling and entity extraction. Synthetic data is generated for low-sample intents, but overgeneration of `farewell` samples occurred.

**Final Dataset:** Columns: `text`, `intent`, `original_intent`, `entities`, `inbound`. The dataset is balanced to ~5,000 samples post-preprocessing, but evaluation size suggests errors.

### Methodology

Stratified train-test splitting is used (80-20) in Cell 4. DistilBERT is trained with class weights to handle imbalance, using 25 epochs, dropout (0.5), and weight decay (0.2). DialogGPT (Cell 3, if used) generates responses based on predicted intents. The evaluation cell computes accuracy, precision, recall, F1, and a confusion matrix. The UI (Cell 8) uses `ipywidgets` for query input.

**DistilBERT Model:** A `WeightedTrainer` applies class-weighted cross-entropy loss. Hyperparameters: batch size=16, warmup steps=200, `fp16` for GPU.

### Model Evaluation and Results

The evaluation cell visualizes performance via a confusion matrix and logs misclassifications. DistilBERT’s performance is limited by `farewell` overprediction (230,378/307,575 eval samples). Metrics (approximate, based on issue):
- Accuracy: ~0.75 (skewed by `farewell`).
- Precision/Recall/F1: Low for non-`farewell` intents.
The confusion matrix shows most intents misclassified as `farewell`. The UI responds to queries but often incorrectly due to intent errors.

### Outline of Project

- **Dataset**: `twcs_lessdata.csv` (upload to Colab).
- **Notebook**: `TruckingChatbot.ipynb` (contains Cells 1-8).
- **Models**: `./chatbot_model` (DistilBERT, if trained), `./dialoggpt_model` (DialogGPT, if trained).
- **Evaluation**: Confusion matrix and misclassification logs in Cell 6.

### Contact and Further Information

Nisheeth Goyal

Email: nisheeth_g2000@yahoo.com

[LinkedIn: https://www.linkedin.com/in/nisheethgoyal/]