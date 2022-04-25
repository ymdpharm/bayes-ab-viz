import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from scipy.stats import beta

from src.models.model import Model


class BetaBinomial(Model):
    _prior_alpha: float
    _prior_beta: float
    _x_a: int
    _x_b: int
    _n_a: int
    _n_b: int

    UNIFORM_PRIOR = "Uniform"
    JEFFREYS_PRIOR = "Jeffrey's"
    CUSTOM_PRIOR = "Custom"

    N_TRIAL = 10000

    def _cb_enforce_larger_or_equal(self, key_in: str, key_target):
        if key_target not in st.session_state:
            st.session_state[key_target] = st.session_state[key_in]
        else:
            st.session_state[key_target] = max(
                st.session_state[key_target], st.session_state[key_in]
            )

    def show_sidebar(self):
        st.sidebar.subheader("Prior Settings")
        prior_model = st.sidebar.radio(
            "Prior Model", [self.UNIFORM_PRIOR, self.JEFFREYS_PRIOR, self.CUSTOM_PRIOR]
        )

        if prior_model == self.UNIFORM_PRIOR:
            self._prior_alpha, self._prior_beta = 1, 1
        elif prior_model == self.JEFFREYS_PRIOR:
            self._prior_alpha, self._prior_beta = 0.5, 0.5
        elif prior_model == self.CUSTOM_PRIOR:
            self._prior_alpha = st.sidebar.number_input("alpha", min_value=0.01)
            self._prior_beta = st.sidebar.number_input("beta", min_value=0.01)

        st.sidebar.subheader("Bucket A Observation")
        self._x_a = st.sidebar.number_input(
            "x",
            key="x_a",
            min_value=0,
            on_change=self._cb_enforce_larger_or_equal,
            args=("x_a", "n_a"),
        )
        self._n_a = st.sidebar.number_input("n", key="n_a", min_value=0)

        st.sidebar.subheader("Bucket B Observation")
        self._x_b = st.sidebar.number_input(
            "x",
            key="x_b",
            min_value=0,
            on_change=self._cb_enforce_larger_or_equal,
            args=("x_b", "n_b"),
        )
        self._n_b = st.sidebar.number_input("n", key="n_b", min_value=0)

    def show_page(self):
        posterior_alpha_a = self._prior_alpha + self._x_a
        posterior_beta_a = self._prior_beta + self._n_a - self._x_a
        posterior_alpha_b = self._prior_alpha + self._x_b
        posterior_beta_b = self._prior_beta + self._n_b - self._x_b

        dist_a = beta(posterior_alpha_a, posterior_beta_a)
        dist_b = beta(posterior_alpha_b, posterior_beta_b)

        col1, col2 = st.columns(2)

        with col1:
            fig = plt.figure()
            ax = plt.axes()
            x = np.linspace(0, 1, 1000)
            ax.plot(x, dist_a.pdf(x), lw=2, label="Bucket A", color="red")
            ax.plot(x, dist_b.pdf(x), lw=2, label="Bucket B", color="blue")
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)

        def _win_rate():
            win_a = 0
            for _ in range(self.N_TRIAL):
                rand_a = dist_a.rvs()
                rand_b = dist_b.rvs()
                if rand_a > rand_b:
                    win_a += 1
            win_rate_a = win_a / self.N_TRIAL

            return win_rate_a, (1 - win_rate_a)

        with col2:
            exp_a = dist_a.expect()
            lower_a, upper_a = dist_a.interval(0.95)
            exp_b = dist_b.expect()
            lower_b, upper_b = dist_b.interval(0.95)
            win_rate_a, win_rate_b = _win_rate()

            st.header("Report")
            st.subheader(f"Bucket A Win Rate: {win_rate_a:.2f}")
            text = f"""
            - Expected Value    : {exp_a:.3f}
            - 95% Interval   : [{lower_a:.3f}, {upper_a:.3f}]
            """
            st.markdown(text)

            st.subheader(f"Bucket B Win Rate: {win_rate_b:.2f}")
            text = f"""
            - Expected Value    : {exp_b:.3f}
            - 95% Interval   : [{lower_b:.3f}, {upper_b:.3f}]
            """
            st.markdown(text)

            st.caption(f"Based on {self.N_TRIAL:,} sampling results.")
