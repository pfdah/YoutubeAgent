import streamlit as st
from utils import load_video_data_vector, get_response_from_query

st.title('Youtube Video Answerer')
input = st.text_input('Enter the Youtube video link')

if input:
    db = load_video_data_vector(input)
    st.write("Successfully loaded the video information")

video = st.text_input('Questions about the video')
if video:
    response, docs = get_response_from_query(db,video)
    st.write("Model response : ", response)
