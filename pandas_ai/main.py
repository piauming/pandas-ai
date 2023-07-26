from dotenv import load_dotenv
import os
import streamlit as st
import pandas as pd
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI
from pandasai.middlewares.base import Middleware
from pandasai.middlewares.streamlit import StreamlitMiddleware

class CodeMiddleware(Middleware):
    def run(self, code: str) -> str:
        global my_code
        my_code = code
        return code

load_dotenv()
API_KEY = os.environ['OPENAI_API_KEY']

llm = OpenAI(api_token=API_KEY)
pandas_ai = PandasAI(llm, enable_cache=False, middlewares=[StreamlitMiddleware(), CodeMiddleware()])

st.title("Prompt-driven analysis with PandasAI")
uploaded_file = st.file_uploader("Upload a CSV file for analysis", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df.head(3))

    prompt = st.text_area("Enter your prompt:")
    # Count the number of passengers in each age group [table]
    # (Chart) Plot the number of passengers in each age group
    # Plot the number of survivors in each age group (None)
    # Count the number of survivors in each age group (None)
    # List the attributes for a higher chance of survival (Table)

    if st.button("Generate"):
        if prompt:
            st.write("PandasAI is generating an answer, please wait...")
            output = pandas_ai.run(df, prompt=prompt)
            st.code(my_code, language='python')
            st.write(output)
        else:
            st.warning("Please enter a prompt.")
