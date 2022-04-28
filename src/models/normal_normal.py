import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from scipy.stats import norm

from src.models.model import Model


class NormalNormal(Model):
    _prior_mu: float
    _prior_sigma: float
    _known_sigma: float
    _sum_a: float
    _sum_b: float
    _n_a: int
    _n_b: int

    N_TRIAL = 10000

    def _cb_enforce_at_least_one(self, key_target: str):
        st.session_state[key_target] = max(1, st.session_state[key_target])

    def show_sidebar(self):
        st.sidebar.subheader("Prior Settings")

        self._prior_mu = st.sidebar.number_input("mu", step=1.0)
        if "sigma" not in st.session_state:
            st.session_state["sigma"] = 1

        self._prior_sigma = st.sidebar.number_input(
            "sigma", key="sigma", min_value=0.001, step=1.0
        )

        st.sidebar.subheader("Known Variance of data")

        if "known_sigma" not in st.session_state:
            st.session_state["known_sigma"] = 1

        self._known_sigma = st.sidebar.number_input(
            "sigma", key="known_sigma", min_value=0.001, step=1.0
        )

        st.sidebar.subheader("Bucket A Observation")
        self._sum_a = st.sidebar.number_input(
            "sum",
            key="normal_sum_a",
            min_value=0.0,
            on_change=self._cb_enforce_at_least_one,
            kwargs={"key_target": "normal_n_a"},
            step=1.0,
        )
        self._n_a = st.sidebar.number_input("n", key="normal_n_a", min_value=0)

        st.sidebar.subheader("Bucket B Observation")
        self._sum_b = st.sidebar.number_input(
            "sum",
            key="normal_sum_b",
            min_value=0.0,
            on_change=self._cb_enforce_at_least_one,
            kwargs={"key_target": "normal_n_b"},
            step=1.0,
        )
        self._n_b = st.sidebar.number_input("n", key="normal_n_b", min_value=0)

    def show_page(self):
        posterior_mu_a = (
            self._prior_mu / self._prior_sigma**2
            + self._sum_a / self._known_sigma**2
        ) / (1 / self._prior_sigma**2 + self._n_a / self._known_sigma**2)
        posterior_mu_b = (
            self._prior_mu / self._prior_sigma**2
            + self._sum_b / self._known_sigma**2
        ) / (1 / self._prior_sigma**2 + self._n_b / self._known_sigma**2)
        posterior_sigma_a = 1 / (
            1 / self._prior_sigma**2 + self._n_a / self._known_sigma**2
        )
        posterior_sigma_b = 1 / (
            1 / self._prior_sigma**2 + self._n_b / self._known_sigma**2
        )

        dist_a = norm(posterior_mu_a, posterior_sigma_a)
        dist_b = norm(posterior_mu_b, posterior_sigma_b)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Posterior dist of $\mu$. ")
            fig = plt.figure()
            ax = plt.axes()
            x = np.linspace(
                min(
                    posterior_mu_a - 3 * posterior_sigma_a,
                    posterior_mu_b - 3 * posterior_sigma_b,
                ),
                max(
                    posterior_mu_a + 3 * posterior_sigma_a,
                    posterior_mu_b + 3 * posterior_sigma_b,
                ),
                1000,
            )
            ax.plot(x, dist_a.pdf(x), lw=2, label="Bucket A", color="red")
            ax.plot(x, dist_b.pdf(x), lw=2, label="Bucket B", color="blue")
            ax.legend(loc="upper right")
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

            text = f"""
            ### Bucket A Win Rate: {win_rate_a:.2f}

            - Expected Value    : {exp_a:.3f}
            - 95% Interval   : [{lower_a:.3f}, {upper_a:.3f}]
            
            ### Bucket B Win Rate: {win_rate_b:.2f}

            - Expected Value    : {exp_b:.3f}
            - 95% Interval   : [{lower_b:.3f}, {upper_b:.3f}]
            """
            st.markdown(text)

        st.caption(f"Based on {self.N_TRIAL:,} sampling results.")
