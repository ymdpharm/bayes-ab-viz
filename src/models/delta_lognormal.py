import streamlit as st

from src.models.model import Model


class DeltaLognormal(Model):
    def show_sidebar(self):
        ...

    def show_page(self):
        st.text("wip")
