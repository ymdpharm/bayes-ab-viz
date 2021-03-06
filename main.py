import matplotlib.pyplot as plt
import streamlit as st

from src.models.beta_binomial import BetaBinomial
from src.models.model import Model
from src.models.normal_normal import NormalNormal

plt.style.use("seaborn-dark")

st.set_page_config(layout="wide")
st.title("Bayes-AB-Viz")

BETA_BINOMIAL = "Beta-Binomial"
NORMAL_NORMAL = "Normal-Normal"
DELTA_LOGNORMAL = "Delta-Lognormal"

models = [
    BETA_BINOMIAL,
    NORMAL_NORMAL,
]

selected_model = st.sidebar.selectbox("Select Model", models)

if selected_model == BETA_BINOMIAL:
    model: Model = BetaBinomial()
elif selected_model == NORMAL_NORMAL:
    model: Model = NormalNormal()

model.show_sidebar()
model.show_page()
