import streamlit as st
import numpy as np
import pandas as pd
import sqlite3
import os
import json
import re
from google import genai
from google.genai import types

st.set_page_config(page_title="DataLens AI", layout="centered")


# ── Gemini client ─────────────────────────────────────────────────────────────
def get_client():
    api_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
    if not api_key:
        return None
    return genai.Client(api_key=api_key)


# ── Sample fintech documents for RAG ─────────────────────────────────────────
DOCUMENTS = [
    {
        "id": "doc1",
        "title": "Nigerian Fintech Market Overview 2024",
        "content": """Nigeria's fintech sector is one of Africa's most vibrant, attracting over $1.5 billion 
        in investment in 2023. Key players include Flutterwave, Paystack, Kuda, Moniepoint, Carbon, and PiggyVest. 
        The sector is regulated by the Central Bank of Nigeria (CBN) and the Securities and Exchange Commission (SEC).
        Mobile money penetration has grown to 45% of the adult population. The USSD channel remains critical for 
        financial inclusion, serving over 60 million Nigerians without smartphones. Instant payment via NIP 
        processed over 8 billion transactions in 2023 worth over N600 trillion."""
    },
    {
        "id": "doc2",
        "title": "CBN Regulatory Framework for Fintechs",
        "content": """The Central Bank of Nigeria regulates fintechs through several licensing categories:
        Payment Service Banks (PSBs) can accept deposits but cannot grant loans. Mobile Money Operators (MMOs)
        facilitate mobile payments. Payment Solution Service Providers (PSSPs) handle payment processing.
        Super Agents provide agency banking services. The CBN's regulatory sandbox allows fintechs to test 
        innovative products. Key regulations include the PSB Guidelines 2020, the ePayment Policy, and the 
        Open Banking Policy 2021. Fintechs must comply with AML/CFT regulations and file Suspicious Transaction 
        Reports (STRs) with NFIU."""
    },
    {
        "id": "doc3",
        "title": "Loan Default Risk in Nigerian Digital Lending",
        "content": """Digital lenders in Nigeria face high default rates averaging 15-25%. Key risk factors 
        include income volatility, lack of credit history, multiple borrowing, and economic shocks like inflation.
        Alternative credit scoring uses mobile data, social signals, bank statement analysis, and psychometric 
        testing. Top digital lenders include Carbon, FairMoney, Branch, and Renmoney. The CRMS (Credit Risk 
        Management System) maintained by CBN tracks borrower credit history. Loan tenors typically range from 
        1-12 months with interest rates between 3-10% monthly. Default prediction models use XGBoost and 
        Random Forest with features like repayment history, loan amount, income ratio, and employment status."""
    },
    {
        "id": "doc4",
        "title": "POS and Agency Banking in Nigeria",
        "content": """Nigeria has over 1.8 million active POS terminals as of 2024, up from 300,000 in 2019.
        Moniepoint leads the agency banking space with over 300,000 agents. OPay and PalmPay have significant 
        presence in urban areas. The agency banking model allows unbanked Nigerians to access financial services 
        through local agents. POS transactions grew by 65% YoY in 2023 to N12 trillion. Common challenges 
        include network downtime, cash liquidity issues, and fraud. The average POS transaction size is N8,500. 
        Lagos, Rivers, and Kano are the top states by POS transaction volume."""
    },
    {
        "id": "doc5",
        "title": "Cryptocurrency and Naira in Nigeria",
        "content": """Nigeria is one of the world's largest crypto markets by volume. Despite CBN restrictions 
        in 2021, peer-to-peer crypto trading thrived. In 2023, CBN lifted the ban and allowed banks to service 
        crypto companies. Bitcoin and USDT are the most traded assets. The eNaira, Nigeria's CBDC launched in 
        2021, has seen slow adoption with under 1 million active wallets. Naira depreciation from N460/$ in 2022 
        to over N1,500/$ in 2024 drove crypto adoption as a hedge. Binance, Yellow Card, and Quidax are major 
        crypto platforms operating in Nigeria. P2P trading volumes exceed $400 million monthly."""
    },
    {
        "id": "doc6",
        "title": "Financial Inclusion in Nigeria",
        "content": """Nigeria's financial inclusion rate reached 64% in 2023, up from 56% in 2020. The EFInA 
        Access to Finance survey tracks inclusion metrics. Key barriers include distance to financial institutions,
        lack of documentation, low financial literacy, and poverty. SANEF (Shared Agent Network Expansion 
        Facilities) aims to reach the last mile. BVN (Bank Verification Number) enrollment exceeds 57 million.
        NIN (National Identity Number) is now required for account opening. Women and rural populations remain 
        the most financially excluded. Fintechs targeting the unbanked include OPay, PalmPay, and Bankly."""
    },
]


# ── Simple cosine similarity for RAG ─────────────────────────────────────────
def simple_search(query, documents, top_k=2):
    query_words = set(query.lower().split())
    scores = []
    for doc in documents:
        doc_words  = set((doc["title"] + " " + doc["content"]).lower().split())
        overlap    = len(query_words & doc_words)
        scores.append((overlap, doc))
    scores.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scores[:top_k]]


# ── SQLite database setup ─────────────────────────────────────────────────────
@st.cache_resource
def setup_database():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    cur  = conn.cursor()

    cur.executescript("""
    CREATE TABLE transactions (
        id INTEGER PRIMARY KEY,
        date TEXT,
        customer_id TEXT,
        amount REAL,
        type TEXT,
        category TEXT,
        state TEXT,
        bank TEXT,
        status TEXT
    );

    CREATE TABLE customers (
        customer_id TEXT PRIMARY KEY,
        name TEXT,
        age INTEGER,
        gender TEXT,
        state TEXT,
        account_type TEXT,
        balance REAL,
        credit_score INTEGER
    );

    CREATE TABLE loans (
        loan_id TEXT PRIMARY KEY,
        customer_id TEXT,
        amount REAL,
        interest_rate REAL,
        tenure_months INTEGER,
        status TEXT,
        disbursed_date TEXT,
        due_date TEXT
    );

    CREATE TABLE agents (
        agent_id TEXT PRIMARY KEY,
        name TEXT,
        state TEXT,
        bank TEXT,
        total_transactions INTEGER,
        total_volume REAL,
        active INTEGER
    );
    """)

    import random
    random.seed(42)
    states      = ["Lagos","Abuja","Kano","Rivers","Oyo","Kaduna","Enugu","Delta"]
    banks       = ["GTBank","Access Bank","Kuda","Moniepoint","OPay","Zenith Bank","UBA"]
    categories  = ["Food","Transport","Utilities","Shopping","Transfer","Salary","Healthcare"]
    loan_status = ["Active","Completed","Defaulted","Overdue"]
    acct_types  = ["Savings","Current","Wallet"]

    # Customers
    for i in range(50):
        cur.execute("INSERT INTO customers VALUES (?,?,?,?,?,?,?,?)", (
            f"CUST{i:04d}",
            f"Customer {i}",
            random.randint(18, 65),
            random.choice(["Male","Female"]),
            random.choice(states),
            random.choice(acct_types),
            round(random.uniform(1000, 500000), 2),
            random.randint(300, 850)
        ))

    # Transactions
    for i in range(200):
        cur.execute("INSERT INTO transactions VALUES (?,?,?,?,?,?,?,?,?)", (
            i,
            f"2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
            f"CUST{random.randint(0,49):04d}",
            round(random.uniform(500, 500000), 2),
            random.choice(["Credit","Debit"]),
            random.choice(categories),
            random.choice(states),
            random.choice(banks),
            random.choice(["Success","Failed","Pending"])
        ))

    # Loans
    for i in range(80):
        cur.execute("INSERT INTO loans VALUES (?,?,?,?,?,?,?,?)", (
            f"LOAN{i:04d}",
            f"CUST{random.randint(0,49):04d}",
            round(random.uniform(10000, 2000000), 2),
            round(random.uniform(3, 10), 2),
            random.choice([1,3,6,12]),
            random.choice(loan_status),
            f"2024-{random.randint(1,6):02d}-{random.randint(1,28):02d}",
            f"2024-{random.randint(7,12):02d}-{random.randint(1,28):02d}"
        ))

    # Agents
    for i in range(30):
        cur.execute("INSERT INTO agents VALUES (?,?,?,?,?,?,?)", (
            f"AGT{i:04d}",
            f"Agent {i}",
            random.choice(states),
            random.choice(banks),
            random.randint(100, 5000),
            round(random.uniform(500000, 50000000), 2),
            random.choice([1,1,1,0])
        ))

    conn.commit()
    return conn


conn = setup_database()

DB_SCHEMA = """
Tables:
1. transactions(id, date, customer_id, amount, type, category, state, bank, status)
2. customers(customer_id, name, age, gender, state, account_type, balance, credit_score)
3. loans(loan_id, customer_id, amount, interest_rate, tenure_months, status, disbursed_date, due_date)
4. agents(agent_id, name, state, bank, total_transactions, total_volume, active)
"""


# ── Header ────────────────────────────────────────────────────────────────────
st.title("DataLens AI")
st.caption("RAG Query Assistant + Text2SQL Platform — Powered by Google Gemini")
st.markdown("Ask questions about Nigerian fintech documents or query the financial database in plain English.")
st.divider()

# ── API Key ───────────────────────────────────────────────────────────────────
if "GEMINI_API_KEY" not in st.secrets and not os.environ.get("GEMINI_API_KEY"):
    st.warning("Add your Gemini API key to continue.")
    api_key_input = st.text_input("Enter Gemini API Key", type="password")
    if api_key_input:
        os.environ["GEMINI_API_KEY"] = api_key_input
        st.success("API key set!")
        st.rerun()
    st.stop()

client = get_client()
if not client:
    st.error("Could not initialize Gemini client. Check your API key.")
    st.stop()

# ── Mode selector ─────────────────────────────────────────────────────────────
mode = st.radio("Choose Mode", ["RAG Query Assistant", "Text2SQL Platform"], horizontal=True)
st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# ── MODE 1: RAG QUERY ASSISTANT ───────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
if mode == "RAG Query Assistant":
    st.subheader("RAG Query Assistant")
    st.markdown("Ask questions about Nigerian fintech — regulations, market data, lending, POS, crypto, and financial inclusion.")

    # Document browser
    with st.expander("Browse Knowledge Base"):
        for doc in DOCUMENTS:
            st.markdown(f"**{doc['title']}**")
            st.caption(doc["content"][:200] + "...")
            st.divider()

    # Sample questions
    st.markdown("**Sample questions:**")
    sample_qs = [
        "What are the CBN licensing categories for fintechs?",
        "How many POS terminals does Nigeria have?",
        "What is the default rate for digital lenders in Nigeria?",
        "How has Naira depreciation affected crypto adoption?",
        "What is Nigeria's financial inclusion rate?",
    ]
    for q in sample_qs:
        if st.button(q, key=q):
            st.session_state["rag_query"] = q

    st.markdown("---")
    query = st.text_area("Your Question",
                         value=st.session_state.get("rag_query", ""),
                         placeholder="e.g. What are the key risks in Nigerian digital lending?",
                         height=80)

    if st.button("Ask DataLens AI", key="rag_btn", use_container_width=True):
        if not query.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Searching knowledge base and generating answer..."):
                # Retrieve relevant docs
                retrieved = simple_search(query, DOCUMENTS, top_k=2)
                context   = "\n\n".join([f"**{d['title']}**\n{d['content']}" for d in retrieved])

                prompt = (
                    "You are a Nigerian fintech expert assistant. "
                    "Answer the user's question using ONLY the context provided below. "
                    "Be concise, accurate, and reference specific facts from the context.\n\n"
                    f"CONTEXT:\n{context}\n\n"
                    f"QUESTION: {query}\n\n"
                    "ANSWER:"
                )

                try:
                    response = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=prompt
                    )
                    answer = response.text

                    st.success("Answer")
                    st.markdown(answer)

                    st.divider()
                    st.markdown("**Sources used:**")
                    for doc in retrieved:
                        st.markdown(f"- {doc['title']}")

                except Exception as e:
                    st.error(f"Gemini API error: {str(e)}")

            if "rag_query" in st.session_state:
                del st.session_state["rag_query"]


# ══════════════════════════════════════════════════════════════════════════════
# ── MODE 2: TEXT2SQL PLATFORM ─────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.subheader("Text2SQL Platform")
    st.markdown("Ask questions about the Nigerian fintech database in plain English — Gemini converts them to SQL and runs them.")

    # Schema viewer
    with st.expander("View Database Schema"):
        st.code(DB_SCHEMA)
        st.markdown("**Sample data preview:**")
        t1, t2, t3, t4 = st.tabs(["Transactions","Customers","Loans","Agents"])
        with t1: st.dataframe(pd.read_sql("SELECT * FROM transactions LIMIT 5", conn), use_container_width=True)
        with t2: st.dataframe(pd.read_sql("SELECT * FROM customers LIMIT 5",    conn), use_container_width=True)
        with t3: st.dataframe(pd.read_sql("SELECT * FROM loans LIMIT 5",        conn), use_container_width=True)
        with t4: st.dataframe(pd.read_sql("SELECT * FROM agents LIMIT 5",       conn), use_container_width=True)

    # Sample questions
    st.markdown("**Sample questions:**")
    sql_samples = [
        "Show me the top 5 states by total transaction volume",
        "How many loans are defaulted?",
        "What is the average credit score by gender?",
        "Which bank has the most active agents?",
        "Show total transaction amount by category",
        "How many customers have a balance above 100000?",
    ]
    for q in sql_samples:
        if st.button(q, key="sql_"+q):
            st.session_state["sql_query"] = q

    st.markdown("---")
    nl_query = st.text_area("Your Question",
                             value=st.session_state.get("sql_query", ""),
                             placeholder="e.g. Show me total loan amount by status",
                             height=80)

    if st.button("Generate SQL and Run", key="sql_btn", use_container_width=True):
        if not nl_query.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Generating SQL with Gemini..."):

                sql_prompt = (
                    "You are a SQL expert. Convert the user's natural language question into a valid SQLite SQL query.\n\n"
                    f"DATABASE SCHEMA:\n{DB_SCHEMA}\n\n"
                    f"QUESTION: {nl_query}\n\n"
                    "Rules:\n"
                    "- Return ONLY the SQL query, nothing else\n"
                    "- No markdown, no backticks, no explanation\n"
                    "- Use proper SQLite syntax\n"
                    "- Limit results to 50 rows max\n"
                    "SQL:"
                )

                try:
                    response = client.models.generate_content(
                        model="gemini-2.0-flash",
                        contents=sql_prompt
                    )
                    sql_raw = response.text.strip()
                    sql_clean = re.sub(r"```sql|```", "", sql_raw).strip()

                    st.markdown("**Generated SQL:**")
                    st.code(sql_clean, language="sql")

                    try:
                        result_df = pd.read_sql(sql_clean, conn)

                        st.success(f"Query returned {len(result_df)} rows")
                        st.dataframe(result_df, use_container_width=True)

                        # Auto chart if numeric result
                        numeric_cols = result_df.select_dtypes(include=np.number).columns.tolist()
                        text_cols    = result_df.select_dtypes(exclude=np.number).columns.tolist()

                        if len(numeric_cols) >= 1 and len(text_cols) >= 1 and len(result_df) > 1:
                            st.markdown("**Chart:**")
                            fig, ax = plt.subplots(figsize=(7, 4))
                            ax.bar(result_df[text_cols[0]].astype(str),
                                   result_df[numeric_cols[0]],
                                   color="#3498db", width=0.6)
                            ax.set_xlabel(text_cols[0])
                            ax.set_ylabel(numeric_cols[0])
                            ax.set_title(nl_query[:60])
                            plt.xticks(rotation=45, ha="right", fontsize=8)
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()

                    except Exception as db_err:
                        st.error(f"SQL execution error: {str(db_err)}")
                        st.markdown("The generated SQL could not be executed. Try rephrasing your question.")

                except Exception as e:
                    st.error(f"Gemini API error: {str(e)}")

            if "sql_query" in st.session_state:
                del st.session_state["sql_query"]

    st.divider()

    # Manual SQL editor
    with st.expander("Manual SQL Editor"):
        manual_sql = st.text_area("Write your own SQL query", height=100,
                                   placeholder="SELECT * FROM transactions LIMIT 10")
        if st.button("Run SQL", key="manual_sql"):
            try:
                df = pd.read_sql(manual_sql, conn)
                st.dataframe(df, use_container_width=True)
            except Exception as e:
                st.error(f"Error: {str(e)}")

st.divider()
st.caption("DataLens AI · RAG + Text2SQL · Powered by Google Gemini · Built by Okparaji Wisdom")