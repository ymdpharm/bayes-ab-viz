import matplotlib.pyplot as plt
import streamlit as st

from src.models.beta_binomial import BetaBinomial
from src.models.delta_lognormal import DeltaLognormal
from src.models.model import Model
from src.models.normal import Normal

plt.style.use("seaborn-dark")

st.set_page_config(layout="wide")
st.title("Bayes-AB-Viz")

BETA_BINOMIAL = "Beta-Binomial"
NORMAL = "Normal"
DELTA_LOGNORMAL = "Delta-Lognormal"

models = [
    BETA_BINOMIAL,
    NORMAL,
    DELTA_LOGNORMAL,
]

selected_model = st.sidebar.selectbox("Select Model", models)

if selected_model == BETA_BINOMIAL:
    model: Model = BetaBinomial()
elif selected_model == NORMAL:
    model: Model = Normal()
elif selected_model == DELTA_LOGNORMAL:
    model: Model = DeltaLognormal()

model.show_sidebar()
model.show_page()
