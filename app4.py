import os
import re
import json
import pandas as pd
import streamlit as st
from openai import AzureOpenAI

# ===============================
# Streamlit Secrets Helper
# ===============================
def get_secret(key: str):
    """
    Fetch secret from Streamlit secrets or environment variables.
    Works on Streamlit Cloud, Azure App Service, Docker, local.
    """
    return st.secrets.get(key) or os.getenv(key)


# ===============================
# Validate secrets
# ===============================
required_keys = [
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_DEPLOYMENT"
]

for key in required_keys:
    if not get_secret(key):
        st.error(f"❌ Missing {key} in Streamlit secrets / App Settings")
        st.stop()


# ===============================
# Initialize Azure OpenAI client
# ===============================
try:
    client = AzureOpenAI(
        azure_endpoint=get_secret("AZURE_OPENAI_ENDPOINT"),
        api_key=get_secret("AZURE_OPENAI_API_KEY"),
        api_version="2024-02-01"
    )

    DEPLOYMENT = get_secret("AZURE_OPENAI_DEPLOYMENT")
    if not DEPLOYMENT:
        st.error("❌ Missing AZURE_OPENAI_DEPLOYMENT in Streamlit secrets / App Settings")
        st.stop()

except Exception as e:
    st.error(f"❌ Error initializing Azure OpenAI client: {str(e)}")
    st.stop()


# ===============================
# Constants
# ===============================
RTI_REFERENCE = "TRAOI/R/E/25/00652"
RTI_COMPLAINT_DATE = "12/05/2025"
RTI_APPLICATION_DATE = "09.09.2025"


# ===============================
# URL Extraction
# ===============================
URL_REGEX = re.compile(
    r"""(?ix)\b(https?://[^\s<>\)\]\}]+)"""
)

def extract_first_url(text: str) -> str:
    if not text or str(text).strip() in {"—", "nan", "", "1", "None"}:
        return ""

    s = str(text).strip()
    if s.startswith("blob:"):
        return ""

    match = URL_REGEX.search(s)
    if not match:
        return ""

    url = match.group(1).rstrip(".,);]}>\"'")
    if not url.startswith(("http://", "https://")):
        return ""

    return url


# ===============================
# Load Knowledge Base
# ===============================
def load_knowledge_base():
    try:
        sheet1 = pd.read_excel("RTI-Query and Response.xlsx", sheet_name="Sheet1")
        sheet2 = pd.read_excel("RTI-Query and Response.xlsx", sheet_name="Sheet2")

        combined_df = pd.concat([sheet1, sheet2], ignore_index=True)

        combined_df.columns = [
            str(col).strip()
            .replace('\xa0', ' ')
            .replace('`', "'")
            .replace('\u2019', "'")
            for col in combined_df.columns
        ]

        theme_col = "Specific type of query (RTI theme)"
        response_col = "TRAI's response (as per letter)"
        url_col = "URLs mentioned in the reply"

        for col in [theme_col, response_col, url_col]:
            if col not in combined_df.columns:
                st.error(f"❌ Required column missing: {col}")
                st.stop()

        combined_df = combined_df[
            ~combined_df[url_col].astype(str).str.contains("^1$", na=False)
        ]
        combined_df = combined_df[
            ~combined_df[url_col].astype(str).str.startswith("blob:")
        ]
        combined_df = combined_df.drop_duplicates(subset=[theme_col])
        combined_df = combined_df.dropna(subset=[theme_col, response_col])
        combined_df.reset_index(drop=True, inplace=True)

        st.session_state.column_mapping = {
            "theme": theme_col,
            "response": response_col,
            "url": url_col
        }

        return combined_df

    except Exception as e:
        st.error(f"❌ Error loading Excel knowledge base: {str(e)}")
        st.stop()


# ===============================
# Split Query into Questions
# ===============================
def split_into_questions(query):
    cleaned_query = query.replace('"', '\\"').replace('\n', ' ').strip()

    prompt = f"""
You are an expert assistant for analyzing RTI queries.

USER QUERY:
{cleaned_query}

Return ONLY a JSON array of questions.
"""

    try:
        response = client.chat.completions.create(
            model=DEPLOYMENT,
            messages=[
                {"role": "system", "content": "Return only valid JSON arrays."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=500
        )

        result = response.choices[0].message.content.strip()

        if "```" in result:
            result = result.split("```")[1].strip()

        questions = json.loads(result)
        if not isinstance(questions, list):
            return [query]

        return [str(q).strip() for q in questions if q]

    except json.JSONDecodeError:
        st.warning("⚠️ Invalid JSON returned. Treating input as single question.")
        return [query]

    except Exception:
        return [query]


# ===============================
# Match Questions to KB
# ===============================
def find_best_match_with_llm(query, knowledge_base):
    theme_col = st.session_state.column_mapping["theme"]
    response_col = st.session_state.column_mapping["response"]

    kb_summary = ""
    for idx, row in knowledge_base.iterrows():
        kb_summary += f"ID:{idx}\nTheme:{row[theme_col]}\nResponse:{row[response_col]}\n\n"

    prompt = f"""
USER QUERY: "{query}"
KNOWLEDGE BASE:
{kb_summary}

Return ONLY the matching ID or -1.
"""

    try:
        response = client.chat.completions.create(
            model=DEPLOYMENT,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=10
        )

        match_id = int(response.choices[0].message.content.strip())
        return knowledge_base.iloc[match_id] if match_id >= 0 else None

    except Exception:
        return None


# ===============================
# Summarize Response
# ===============================
def summarize_response_with_llm(question, response_text, url):
    prompt = f"""
QUESTION: {question}
KNOWLEDGE BASE ENTRY: {response_text}
"""

    try:
        response = client.chat.completions.create(
            model=DEPLOYMENT,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300
        )

        text = response.choices[0].message.content.strip()
        if url and url not in text:
            text += f" Relevant details are available at: {url}"
        return text

    except Exception:
        return response_text


# ===============================
# Generate Final Response
# ===============================
def generate_multi_response(questions_and_matches):
    response_col = st.session_state.column_mapping["response"]
    url_col = st.session_state.column_mapping["url"]

    response = f"""Sir,

Please refer to your RTI application dated {RTI_APPLICATION_DATE} vide Reg. No. {RTI_REFERENCE}.

**Information Sought:**
"""

    for idx, (q, _) in enumerate(questions_and_matches, 1):
        response += f"\n{idx}. {q}\n"

    response += "\n| S.No. | Reply |\n|---|---|\n"

    for idx, (q, match) in enumerate(questions_and_matches, 1):
        if match is not None:
            url = extract_first_url(str(match[url_col]))
            reply = summarize_response_with_llm(q, match[response_col], url)
        else:
            reply = "Information not available in the knowledge base."

        response += f"| {idx} | {reply.replace(chr(10), '<br>')} |\n"

    return response


# ===============================
# MAIN APP
# ===============================
def main():
    st.set_page_config("TRAI RTI Response Generator", "📄", layout="wide")

    st.title("📄 TRAI RTI Response Generator")

    knowledge_base = load_knowledge_base()

    user_query = st.text_area("Enter RTI Query:", height=200)

    if st.button("Generate Response", disabled=not user_query):
        questions = split_into_questions(user_query)
        matches = [(q, find_best_match_with_llm(q, knowledge_base)) for q in questions]
        response = generate_multi_response(matches)
        st.markdown(response, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
