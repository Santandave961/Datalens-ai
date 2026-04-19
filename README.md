# DataLens AI - RAG Query Assistant + Text2SQL Platform

A combined AI-powered web application that lets you query Nigerian fintech documents using RAG (Retrieval-Augmented Generation) and query a financial database using plain English Text2SQL — both powered by Google Gemini.

Built with Python, Google Gemini API, and Streamlit, deployed on Streamlit Community Cloud.

---

## Live Demo

[Click here to view the app]https://datalens-ai-7gmybr6qqshgadon3rxvfc.streamlit.app/

---

## Overview

DataLens AI combines two powerful AI capabilities in one app:

1. **RAG Query Assistant** — Ask questions about Nigerian fintech documents and get answers grounded in retrieved context using Google Gemini
2. **Text2SQL Platform** — Ask questions about a Nigerian fintech database in plain English and Gemini automatically generates and runs the SQL query

---

## Features

**RAG Query Assistant:**
- 6 Nigerian fintech knowledge documents covering regulations, market data, digital lending, POS, crypto, and financial inclusion
- Keyword-based document retrieval
- Gemini generates answers grounded strictly in retrieved context
- Shows which source documents were used
- Sample questions to get started instantly

**Text2SQL Platform:**
- SQLite database with 4 tables: transactions, customers, loans, and agents
- Gemini converts plain English questions to valid SQL
- Executes the query and displays results as a table
- Auto-generates bar chart for numeric results
- Manual SQL editor for advanced queries
- Sample questions to get started instantly

---

## Knowledge Base Documents

| Document | Topic |
|----------|-------|
| Nigerian Fintech Market Overview 2024 | Market size, key players, payment stats |
| CBN Regulatory Framework for Fintechs | Licensing categories, regulations, compliance |
| Loan Default Risk in Nigerian Digital Lending | Default rates, risk factors, credit scoring |
| POS and Agency Banking in Nigeria | Terminal count, top players, transaction data |
| Cryptocurrency and Naira in Nigeria | Crypto adoption, eNaira, Naira depreciation |
| Financial Inclusion in Nigeria | Inclusion rate, barriers, key initiatives |

---

## Database Schema

| Table | Columns |
|-------|---------|
| transactions | id, date, customer_id, amount, type, category, state, bank, status |
| customers | customer_id, name, age, gender, state, account_type, balance, credit_score |
| loans | loan_id, customer_id, amount, interest_rate, tenure_months, status, disbursed_date, due_date |
| agents | agent_id, name, state, bank, total_transactions, total_volume, active |

---

## Tech Stack

- **Language:** Python 3
- **Framework:** Streamlit
- **AI Model:** Google Gemini 1.5 Flash
- **AI Library:** google-genai
- **Database:** SQLite (in-memory)
- **Data Processing:** pandas, NumPy
- **Visualisation:** Matplotlib

---

## Project Structure

```
datalens-ai/
    app.py                      # Main Streamlit application
    requirements.txt            # Python dependencies
    README.md                   # Project documentation
    .gitignore                  # Protects API key from GitHub
    .streamlit/
        secrets.toml            # Gemini API key (never push to GitHub)
```

---

## How to Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/Santandave961/datalens-ai.git
cd datalens-ai
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add your Gemini API key**

Create `.streamlit/secrets.toml` and add:
```toml
GEMINI_API_KEY = "your-gemini-api-key-here"
```

Get your free API key from: https://aistudio.google.com

**4. Run the app**
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## Sample RAG Questions

- What are the CBN licensing categories for fintechs?
- How many POS terminals does Nigeria have?
- What is the default rate for digital lenders in Nigeria?
- How has Naira depreciation affected crypto adoption?
- What is Nigeria's financial inclusion rate?

---

## Sample Text2SQL Questions

- Show me the top 5 states by total transaction volume
- How many loans are defaulted?
- What is the average credit score by gender?
- Which bank has the most active agents?
- How many customers have a balance above 100000?

---

## How It Works

**RAG Pipeline:**
1. User asks a question
2. Keyword search retrieves the top 2 most relevant documents
3. Retrieved context is passed to Gemini with a grounding prompt
4. Gemini generates an answer based strictly on the context
5. Source documents are displayed for transparency

**Text2SQL Pipeline:**
1. User asks a question in plain English
2. Gemini receives the database schema and question
3. Gemini generates a valid SQLite SQL query
4. Query is executed on the in-memory database
5. Results displayed as table with auto-generated chart

---

## Security Note

Never push your `.streamlit/secrets.toml` to GitHub. Your `.gitignore` should include:
```
.streamlit/secrets.toml
__pycache__/
*.pyc
```

---

## Author

**Okparaji Wisdom**
Data Science Student | Fintech Portfolio Builder | AI Engineer

- GitHub: [@Santandave961](https://github.com/Santandave961)
- LinkedIn: Okparaji Wisdom

---

## License

MIT License - feel free to use and modify this project.
