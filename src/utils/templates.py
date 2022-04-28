import streamlit as st


def github():
    st.sidebar.markdown("---")
    st.sidebar._iframe(
        src="https://ghbtns.com/github-btn.html?user=ymdpharm&repo=bayes-ab-viz&type=star&count=true"
    )


def result(
    win_rate_a: float,
    exp_a: float,
    lower_a: float,
    upper_a: float,
    win_rate_b: float,
    exp_b: float,
    lower_b: float,
    upper_b: float,
):
    text = f"""
    ### Bucket A Win Rate: {win_rate_a:.2f}

    - Expected Value    : {exp_a:.3f}
    - 95% Interval   : [{lower_a:.3f}, {upper_a:.3f}]
    
    ### Bucket B Win Rate: {win_rate_b:.2f}

    - Expected Value    : {exp_b:.3f}
    - 95% Interval   : [{lower_b:.3f}, {upper_b:.3f}]
    """
    st.markdown(text)
