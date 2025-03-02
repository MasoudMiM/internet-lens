# Data Gathering and NLP Analysis

## Overview

This project includes two main components: a data gathering script that collects content and comments from various online sources (Reddit, Guardian [`not fully tested yet`], and Lemmy) and a Natural Language Processing (NLP) analysis script that analyzes the gathered data. The analysis includes sentiment analysis, emotion classification, keyword extraction, and visualizations of trends and distributions.

## Features

- **Data Gathering**: 
  - Collects posts and comments from Reddit and Lemmy.
  - Fetches articles from the Guardian API [`not fully tested yet`].
  - Supports keyword-based searches with time filters.

- **NLP Analysis**:
  - Performs sentiment analysis using NLTK and TextBlob.
  - Extracts keywords and summarizes text using LSA.
  - Classifies emotions using a pre-trained transformer model.
  - Generates visualizations for sentiment trends, emotion distributions, and more.

## Requirements

- Python 3.11
- Required Python packages:
  - `praw`
  - `requests`
  - `nltk`
  - `pandas`
  - `textblob`
  - `sumy`
  - `transformers`
  - `tqdm`
  - `matplotlib`
  - `seaborn`
  - `wordcloud`
  - `joypy`
  - `python-dotenv`

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/MasoudMiM/internet-lens.git
   cd internet-lens
   ```

2. **Create and Activate virtual environment** (optional):
   ```bash
   conda env create -f environment.yml
   conda activate data-gathering-nlp
   ```

3. **Set up environment variables**:
   Create a `.env` file in the root directory of the project and add the following variables:
   ```plaintext
   CLIENT_ID=your_reddit_client_id
   CLIENT_SECRET=your_reddit_client_secret
   USER_AGENT=your_user_agent
   LEMMY_USERNAME=your_lemmy_username
   LEMMY_PASSWORD=your_lemmy_password
   GUARDIAN_API_KEY=your_guardian_api_key
   ```

## Usage

1. **Run the Data Gathering Script**:
   Execute the first script to gather data from the specified sources:
   ```bash
   python data_gathering.py
   ```

2. **Run the NLP Analysis Script**:
   After gathering data, run the second script to perform NLP analysis:
   ```bash
   python nlp_analysis.py
   ```

3. **Output**:
   - The gathered data will be saved in the `data` directory.
   - The analysis results and visualizations will be saved in the `outputs` directory.

## Visualizations

The project generates various visualizations, including:
- Trends of posts and comments over time.
- Average sentiment of posts and comments.
- Distribution of emotion scores.
- Word clouds of keywords.
- Stance distribution and discourse complexity.

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](https://en.wikipedia.org/wiki/MIT_License) file for details.

## Contact

For any questions or inquiries, please contact [masoumi.masoud@gmail.com].
```