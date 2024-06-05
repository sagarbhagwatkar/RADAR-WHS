# RADAR-WHS

## Introduction
The purpose of this project is to scrape relevant links from web pages and analyze the content of those pages using Open source LLMs and Google Gemini Pro.

## Dependencies
- Download Ollama on your local machine from [here](https://ollama.com/download).
- Get Google Gemini Pro API from [here](https://aistudio.google.com/app/apikey).
- Required libraries: `pandas`, `requests`, `BeautifulSoup`, `urllib.parse`, `numpy`, `math`, `newspaper`, `nltk`, `sklearn`.
- Ensure that NLTK's 'punkt' tokenizer is downloaded (`nltk.download('punkt')`).
- Install LLM packages by running:
  ```sh
  pip install langchain
  pip install google-generativeai
  ```

## Prerequisite
Create a new column called “words for scraping relevant links” which has keywords like `news`, `media`, `media-release`, etc. This is to only scrape relevant URLs from the website. Skipping this step may result in scraping irrelevant URLs like `about us`, `contact us`, etc.

For example: If the URL looks like `https://www.safework.sa.gov.au/news-and-alerts/`, then the keyword will be `news-and-alerts`.

## Usage

### 1. Scraping Links
- Read the `Trusted_source.csv` file to get keywords present in the column “words for scraping relevant links”.
- Remove any null values and duplicates from the keyword list.
- Run the Scraping URLs code block. This Python script crawls a website starting from a given base URL and retrieves all links that match specified keywords.
- For websites that do not allow web scraping, use their API service if available.

**Explanation of each part of the code:**
- **Imports:**
  - `requests`: Library for making HTTP requests.
  - `BeautifulSoup`: Library for parsing HTML and XML documents.
  - `urljoin`, `urlparse`: Functions for handling URLs.
- **Function `get_links_from_page`:**
  - Takes three parameters: `url`, `base_domain`, and `keywords`.
  - Sends an HTTP GET request to the given URL and parses the response content using BeautifulSoup.
  - Finds all `<a>` tags (links) in the HTML content.
  - Constructs the absolute URL using `urljoin`.
  - Filters the links based on whether their domain matches the `base_domain` and whether any segment of their path matches any keyword in the `keywords` list.
  - Returns a list of absolute links that match the criteria.
- **Function `get_all_links`:**
  - Takes two parameters: `base_url` and `segments` (keywords).
  - Initializes an empty set `all_links` to store all unique links found.
  - Initializes a set `links_to_visit` with the base URL as its only element.
  - Enters a loop where it pops a URL from `links_to_visit`, retrieves all links from that page using `get_links_from_page`, and adds new unique links to `links_to_visit`.
  - The loop continues until there are no more links to visit.
  - Returns a set of all unique links found.
- **Main Execution:**
  - When the script is executed, it prompts the user to enter the base URL to crawl. For example, `Enter the base URL to crawl: https://www.safework.sa.gov.au/news-and-alerts/`.
  - Calls the `get_all_links` function with the base URL and keywords, then prints all the links found.

### 2. Extracting News Articles
- Run the Newspaper code block to extract `Headline`, `Content`, `Date`, `Reference URL` from each link extracted above.
- For links where dates cannot be extracted using the newspaper package, manually inspect the page to extract the dates for each article.

**Explanation of each part of the code:**
- **Imports:**
  - `pandas`: Library for data manipulation and analysis.
  - `newspaper`: Library for extracting and parsing articles from news websites.
  - `nltk`: Natural Language Toolkit library for text processing.
  - `dateparser`: Library for parsing date strings into datetime objects.
- **NLTK Download:**
  - Downloads the NLTK punkt tokenizer. This is necessary for some NLP tasks like tokenization.
- **Configuration:**
  - Sets up a custom user-agent and request timeout for the Newspaper library. The user-agent string helps identify the client to web servers, and the timeout specifies the maximum time to wait for a response from the server.
- **Article Extraction:**
  - Iterates over each link in the `all_links` list.
  - For each link, attempts to download the article content using the Newspaper library.
  - Parses the downloaded article content.
  - Applies natural language processing (NLP) techniques to the article content.
  - Extracts the headline, content, publish date, and URL of the article.
  - Appends this information as a list to the `result` list.
- **DataFrame Creation:**
  - Creates a pandas DataFrame (`df`) from the `result` list with columns for `Headline`, `Content`, `Date`, and `Reference URL`.
  - Removes duplicate rows based on both the `Headline` and `Date` columns to ensure unique articles.
  - Stores the resulting DataFrame in `unique_df`.

### 3. Date Range Filter
- Run the Date Range Filter code block to filter extracted articles based on a specified date range.

**Explanation of each part of the code:**
- **Date Conversion:**
  - Converts the `Date` column of the DataFrame `unique_df` to datetime format using `pd.to_datetime()`.
  - The `utc=True` parameter ensures that datetime objects are in the UTC timezone.
  - Uses `dt.strftime('%Y-%m-%d')` to format the dates as strings in 'YYYY-MM-DD' format.
- **Null Value Removal:**
  - Removes rows with null values in the `Date` column using `dropna(subset=['Date'])`.
  - Ensures that only rows with valid dates are retained for filtering.
- **User Input:**
  - Prompts the user to enter the start and end dates for the custom date range in 'YYYY-MM-DD' format.
- **Date Conversion (User Input):**
  - Converts the user inputs for start and end dates to datetime objects using `pd.to_datetime()`.
  - Formats the datetime objects as strings in 'YYYY-MM-DD' format with `strftime('%Y-%m-%d')`.
- **Date Range Filtering:**
  - Filters the DataFrame `unique_df` based on the custom date range using boolean indexing.
  - Uses `(unique_df['Date'] >= start_date) & (unique_df['Date'] <= end_date)` to create a boolean mask for rows within the specified date range.

### 4. Cosine Similarity Analysis
- Run the Cosine Similarity Analysis code block to calculate cosine similarity scores for a given keyword against the content of extracted articles.

**Explanation of each part of the code:**
- **Imports:**
  - `pandas`: Library for data manipulation and analysis.
  - `TfidfVectorizer`: Scikit-learn class for converting a collection of raw documents to a matrix of TF-IDF features.
  - `cosine_similarity`: Scikit-learn function for computing cosine similarity between samples in two matrices.
- **Input Phrase:**
  - Prompts the user to enter a keyword or phrase.
- **TF-IDF Vectorization:**
  - Creates a TF-IDF vectorizer object `tfidf_vectorizer`.
  - Fits the vectorizer to the content of the DataFrame `filtered_df` using `fit_transform()`.
  - Transforms the input phrase into a TF-IDF vector using `transform()`.
- **Cosine Similarity Calculation:**
  - Calculates the cosine similarity between the TF-IDF vector of the input phrase and the TF-IDF matrix of the content in the DataFrame.
  - The resulting similarity scores represent how similar each row of the DataFrame's content is to the input phrase.
- **DataFrame Manipulation:**
  - Adds the similarity scores as a new column `Similarity Score` to the DataFrame `df`.
  - Creates a new DataFrame `output_df` from the updated `df`.

### 5. Insight Generation using LLaMA2 7b
- Run the LLaMA2 code block to generate insights from the extracted article content.

**Explanation of each part of the code:**
- **Imports:**
  - `pandas`: Library for data manipulation and analysis.
  - `Ollama`: Class from the LangChain library for accessing the Ollama language model.
- **Ollama Initialization:**
  - Initializes an Ollama object `ollama` with the specified base URL and model (`'llama2'`).
  - The base URL indicates the endpoint where the Ollama service is hosted.
- **Insight Generation Function:**
  - Defines a function `generate_insights` that takes three parameters: `df` (DataFrame), `text_column` (column containing text data), and `insight_column` (column to store generated insights).
  - Iterates over each row in the DataFrame.
  - Retrieves the text from the specified column.
  - Generates insights from the text using the Ollama language model.
  - Saves the generated insight into the specified insight column in the DataFrame.
- **Function Invocation:**
  - Invokes the `generate_insights` function with the DataFrame `final_df`, specifying the `Content` column as the source of text data and `insight` as the column to store generated insights.
  - The result is stored in a new DataFrame named `

LLaMA2`.

### 6. Insight Generation using Google Gemini Pro
- Run the Gemini Pro code block to generate insights from the extracted article content.

**Explanation of each part of the code:**
- **Imports:**
  - `google.generativeai`: Module for accessing generative AI models by Google.
  - `HarmCategory`, `HarmBlockThreshold`: Types for specifying safety settings.
  - `os`: Module for interacting with the operating system.
- **Configuration:**
  - Configures the generative AI module `genai` with an API key generated from [here](https://aistudio.google.com/app/apikey).
  - Initializes a GenerativeModel object `model` with the `'gemini-pro'` model.
- **Insight Generation Function:**
  - Defines a function `generate_insights` that takes three parameters: `df` (DataFrame), `text_column` (column containing text data), and `insight_column` (column to store generated insights).
  - Iterates over each row in the DataFrame.
  - Retrieves the text from the specified column.
  - Generates insights from the text using the Gemini Pro generative model.
  - Specifies safety settings to block harmful content, such as hate speech, harassment, sexually explicit content, and dangerous content.
  - Saves the generated insight into the specified insight column in the DataFrame.
- **Function Invocation:**
  - Invokes the `generate_insights` function with the DataFrame `final_df`, specifying the `Content` column as the source of text data and `insight_Gemini` as the column to store generated insights.
  - The result is stored in a new DataFrame named `gemini`.

### 7. Insight Generation using LLaMA chat
- Run the LLaMA chat code block to generate insights from the extracted article content.

**Explanation of each part of the code:**
- **Imports:**
  - `ChatOllama`: Class for accessing the Ollama chat model from the LangChain community module.
  - `StrOutputParser`: Class for parsing string output from the LangChain core module.
  - `ChatPromptTemplate`: Class for creating chat prompt templates from the LangChain core module.
- **Ollama Initialization:**
  - Initializes a ChatOllama object `llm` with the specified model (`'llama2'`).
- **Chat Prompt Template:**
  - Creates a chat prompt template `prompt` for generating insights with the desired format and constraints.
  - The template includes instructions to generate insights in bullet points with correct statistical details, limited to a maximum of 100 words, based on the provided topic.
- **Chain Setup:**
  - Combines the chat prompt template, Ollama chat model, and string output parser into a processing chain `chain`.
  - The chain specifies the flow of data and processing steps: prompt -> Ollama -> string output parser.
- **Insight Generation Function:**
  - Defines a function `generate_insights` that takes three parameters: `df` (DataFrame), `text_column` (column containing text data), and `insight_column` (column to store generated insights).
  - Iterates over each row in the DataFrame.
  - Retrieves the text from the specified column.
  - Invokes the processing chain `chain` with the text as the topic to generate insights.
  - Saves the generated insight into the specified insight column in the DataFrame.
- **Function Invocation:**
  - Invokes the `generate_insights` function with the DataFrame `final_df`, specifying the `Content` column as the source of text data and `insight_llamachat` as the column to store generated insights.
  - The result is stored in a new DataFrame named `llamachat`.

### 8. Saving Results
- Save the final DataFrame or insights to a CSV file at the desired location.

## Notes
- For links where dates cannot be extracted using the newspaper package, manually inspect the page to extract the dates for each article.
- For websites that do not allow web scraping, use their API service if available.
- A low threshold for the similarity score will result in more content or rows, and vice versa. Choose the similarity score based on business needs.

## Conclusion
Please explore and adapt the script for your specific needs. For any help, please email [sagarbhagwatkar99@gmail.com](mailto:sagarbhagwatkar99@gmail.com).
