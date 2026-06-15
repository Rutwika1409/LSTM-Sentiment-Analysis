import streamlit as st

from src.predict import predict_sentiment

st.set_page_config(
    page_title="IMDb LSTM Sentiment Analysis",
    page_icon="🎬",
    layout="wide"
)

st.markdown(
    """
    <h1 style='text-align:center;'>
    🎬 IMDb LSTM Sentiment Analysis
    </h1>
    """,
    unsafe_allow_html=True
)

st.write(
    "Analyze movie reviews using an LSTM Neural Network."
)

review = st.text_area(
    "Enter Movie Review",
    height=250
)

if st.button(
    "Predict Sentiment",
    use_container_width=True
):

    if review.strip():

        sentiment, score = predict_sentiment(
            review
        )

        if sentiment == "Positive":
            st.success(
                f"😊 {sentiment}"
            )
            st.progress(
                float(score)
            )

        else:
            st.error(
                f"😞 {sentiment}"
            )
            st.progress(
                float(1 - score)
            )