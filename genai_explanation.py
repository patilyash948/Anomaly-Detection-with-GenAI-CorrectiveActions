import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
import certifi
import ssl

# -------------------------------
# 1️⃣ Load API key
# -------------------------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("❌ OPENAI_API_KEY not found in .env file")
    st.stop()

# Fix SSL issues on Windows
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ['SSL_CERT_FILE'] = certifi.where()

# -------------------------------
# 2️⃣ Load anomaly CSV
# -------------------------------
df = pd.read_csv("ai4i2020_with_anomalies.csv")
st.title("Industrial Anomaly Chatbot 🤖")
st.write("Ask any question about detected anomalies in industrial sensor data!")

# Only anomalies
anomaly_df = df[df['Anomaly'] == 1]
total_anomalies = len(anomaly_df)

if anomaly_df.empty:
    st.warning("No anomalies found in the dataset!")
    st.stop()

# -------------------------------
# 3️⃣ Prompt Template
# -------------------------------
template = """
You are an industrial engineer assistant.

Dataset info:
- Total anomalies detected: {total_anomalies}
- Anomaly sensor readings (top 10 by anomaly score):
{top_anomalies}

User question: "{user_question}"

Explain in simple terms:
First give the total anonly count 
which are top most anomly s by score and also why top most 
- Why top most nomalies might have occurred-reason
- Also suggsest corrective actions
- Potential impact on the machine(s)
- Maintenance or troubleshooting suggestions
- Include any stats if relevant

Provide a concise, practical answer for engineers.
"""

prompt = PromptTemplate(
    input_variables=["total_anomalies", "top_anomalies", "user_question"],
    template=template
)

# -------------------------------
# 4️⃣ Setup LangChain LLM
# -------------------------------
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.3,
    openai_api_key=api_key
)

# -------------------------------
# 5️⃣ User input
# -------------------------------
user_input = st.text_input("Ask your question about anomalies:")

if st.button("Get Answer") and user_input.strip() != "":
    # Prepare top 10 anomalies data
    top_rows = anomaly_df.sort_values(by="Anomaly_Score", ascending=False).head(10)
    top_anomalies_text = ""
    for idx, row in top_rows.iterrows():
        top_anomalies_text += (
            f"- Index {idx}: Air_temperature={row['Air temperature [K]']}, "
            f"Process_temperature={row['Process temperature [K]']}, "
            f"Rotational_speed={row['Rotational speed [rpm]']}, "
            f"Torque={row['Torque [Nm]']}, "
            f"Tool_wear={row['Tool wear [min]']}, "
            f"Score={row['Anomaly_Score']}\n"
        )

    # Format prompt
    formatted_prompt = prompt.format(
        total_anomalies=total_anomalies,
        top_anomalies=top_anomalies_text,
        user_question=user_input
    )

    try:
        response = llm.invoke(formatted_prompt)
        st.success("💡 Answer:")
        st.write(response.content)
    except Exception as e:
        st.error(f"❌ Error generating answer: {e}")
