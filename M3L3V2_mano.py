import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from snowflake.cortex import complete
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark import Session

# Create Snowflake connection
@st.cache_resource
def get_snowflake_session():
    connection_parameters = {
        "account": st.secrets["connections"]["snowflake"]["account"],
        "user": st.secrets["connections"]["snowflake"]["user"],
        "password": st.secrets["connections"]["snowflake"]["password"],
        "role": st.secrets["connections"]["snowflake"]["role"],
        "warehouse": st.secrets["connections"]["snowflake"]["warehouse"],
        "database": st.secrets["connections"]["snowflake"]["database"],
        "schema": st.secrets["connections"]["snowflake"]["schema"]
    }
    return Session.builder.configs(connection_parameters).create()

session = get_snowflake_session()

# Load data from Snowflake
query = """
SELECT
    *
FROM
    reviews_sentiment_big
"""
df_reviews = session.sql(query).to_pandas()

# Convert date columns to datetime
df_reviews['REVIEW_DATE'] = pd.to_datetime(df_reviews['REVIEW_DATE'])
df_reviews['SHIPPING_DATE'] = pd.to_datetime(df_reviews['SHIPPING_DATE'])

df_string = df_reviews.to_string(index=False)

def create_avalanche_prompt(user_question: str, dataframe_context: str) -> str:
    """Creates the prompt for the LLM."""
    prompt = f"""
You are a helpful AI chat assistant. Answer the user's question based on the provided
context data from customer reviews provided below.

Use the data in the <context> section to inform your answer about customer reviews or sentiments
if the question relates to it. If the question is general and not answerable from the context, answer naturally. Do not explicitly mention "based on the context" unless necessary for clarity.

<context>
{dataframe_context}
</context>

<question>
{user_question}
</question>
"""
    return prompt


# Initialize the Streamlit app
st.title("Avalanche Streamlit App")

# Visualization: Average Sentiment by Carrier
st.subheader("Average Sentiment by Carrier")
product_sentiment = df_reviews.groupby("CARRIER")["SENTIMENT_SCORE"].mean().sort_values()

fig, ax = plt.subplots()
product_sentiment.plot(kind="barh", ax=ax, title="Average Sentiment by Carrier")
ax.set_xlabel("Sentiment Score")
plt.tight_layout()
st.pyplot(fig)

# Region filter on the main page
st.subheader("Filter by Region")
regions = df_reviews['REGION'].unique()
selected_regions = st.multiselect(
    "Select Regions to View:",
    options=regions,
    default=regions  # Default to showing all regions
)

# Filter data based on selected regions
filtered_data = df_reviews[df_reviews['REGION'].isin(selected_regions)]

# Display the filtered data as a table
st.subheader("Filtered Data Table")
st.dataframe(filtered_data)

# Visualization: Sentiment Distribution for Selected Regions
st.subheader("Sentiment Distribution for Selected Regions")
fig, ax = plt.subplots()
filtered_data['SENTIMENT_SCORE'].hist(ax=ax, bins=20)
ax.set_title("Distribution of Sentiment Scores")
ax.set_xlabel("Sentiment Score")
ax.set_ylabel("Frequency")
st.pyplot(fig)

# Chatbot for Q&A
st.subheader("Ask Questions About Your Data")
# Cache the completion function
@st.cache_data
def get_cached_completion(question: str, context: str):
    return complete(
        model="claude-3-5-sonnet", 
        prompt=create_avalanche_prompt(question, context), 
        session=session
    )

# Chatbot Q&A section
st.subheader("Ask Questions About Your Data")
user_question = st.text_input("Enter your question here:")

if user_question:
    with st.spinner('Generating response...'):
        response = get_cached_completion(user_question, df_string)
    st.write(response)


