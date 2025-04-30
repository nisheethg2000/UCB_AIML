### Project Title

# Chatbot Data Evaluation Notebook
**Author: Nisheeth Goyal**

#### Executive summary
The notebook processes the Twitter Customer Service Dataset, filters it for trucking/fuel card-related queries (e.g., “Why was my card declined at Pilot Flying J?”), and performs comprehensive data analysis. It uses advanced techniques to extract insights and generates visual charts to aid decision-making for chatbot design.


#### Rationale
This notebook is ideal for data scientists or developers preparing a dataset for a chatbot handling trucking/fuel card queries. It provides insights into query content (e.g., frequent keywords like “declined”), user sentiment (e.g., negative for declined card issues), and data quality (e.g., intent balance), guiding intent design and data preprocessing.

#### Research Question
The purpose of the customer service chatbot is to provide efficient, 24/7 support to clients of a trucking and fueling credit provider company, streamlining common inquiries and transactions related to fuel credit accounts, payment processing, and account management. The chatbot will enhance customer satisfaction by delivering quick, accurate responses, reducing wait times, and freeing up human agents for complex issues. It will also improve operational efficiency by automating repetitive tasks and integrating with the company’s existing systems (e.g., CRM, billing platforms).

#### Data Sources
- **Dataset**: Download `twcs.csv` from [Kaggle](https://www.kaggle.com/thoughtvector/customer-support-on-twitter) and place it in the same directory as the notebook.
- **Storage**: Approximately 500MB for the dataset.

#### Methodology
- **Text Analysis**:
  - **Keyword Frequency**: Identifies prevalent terms (e.g., “card,” “fuel”) using NLTK.
  - **Sentiment Analysis**: Gauges query tone (positive, negative, neutral) with TextBlob.
  - **Named Entity Recognition (NER)**: Extracts entities like station names (“Pilot Flying J”) and amounts (“$200”) using spaCy.
- **Data Quality Metrics**:
  - **Completeness**: Measures the proportion of non-missing tweets.
  - **Relevance**: Assesses the proportion of tweets relevant to trucking/fuel card queries.
  - **Diversity**: Evaluates the distribution of simulated intents (e.g., “check_balance,” “report_declined_card”) using normalized entropy.
- **Visual Charts**:
  - Bar plot of top 10 keywords.
  - Pie chart of sentiment distribution.
  - Bar plot of entity types (e.g., ORG, MONEY).
  - Bar plot of simulated intent distribution.
  - Bar plot of data quality metrics (completeness, relevance, diversity).
  - Histogram of tweet length distribution.
  - Word cloud of frequent terms.

#### Results
 **Test Dataset**:
  - File: `trucking_chatbot_test_dataset.csv`
  - Columns: `text`, `intent`, `entities`
  - Description: Contains filtered tweets with simulated intents (e.g., “report_declined_card”) and extracted entities (e.g., “[$200, MONEY]”).

- **Metrics** (printed to console):
  - **Completeness**: Proportion of non-missing tweets (e.g., ~0.9999).
  - **Relevance**: Proportion of relevant tweets (1.0, as filtered).
  - **Diversity**: Normalized entropy of intent distribution (e.g., ~0.85, indicating balanced intents).

- **Visual Charts**:
  - **Keyword Frequency**: Bar plot of top 10 words (e.g., “card,” “fuel”).
  - **Sentiment Distribution**: Pie chart (e.g., 60% Neutral, 25% Negative, 15% Positive).
  - **Entity Distribution**: Bar plot of entity types (e.g., ORG=50, MONEY=20).
  - **Intent Distribution**: Bar plot of intents (e.g., `general_inquiry`=200, `check_balance`=150).
  - **Data Quality Metrics**: Bar plot for completeness, relevance, diversity.
  - **Tweet Length Distribution**: Histogram (e.g., most tweets 50-100 characters).
  - **Word Cloud**: Visual of frequent terms (e.g., “card,” “station” prominent).

# Example Insights
 **Keywords**: High frequency of “declined” and “balance” suggests intents like “report_declined_card” and “check_balance” are critical.
- **Sentiment**: 25% Negative queries indicate a need for empathetic responses (e.g., for “card declined” issues).
- **Entities**: Frequent ORG entities (e.g., “Pilot Flying J”) highlight the importance of station-specific handling.
- **Data Quality**: High relevance (1.0) confirms the dataset’s focus, but moderate diversity (~0.85) suggests adding more varied queries.

#### Next steps
 Inspect `trucking_chatbot_test_dataset.csv` to verify simulated intents and entities.
- Manually label intents and entities for chatbot training (e.g., in Rasa or Dialogflow).
- Enhance the notebook with topic modeling, entity co-occurrence, or advanced sentiment analysis (see “Extending the Analysis”).
- Contact the repository maintainer for additional visualizations or integration with other datasets.

#### Outline of project

- [Link to notebook 1]()


