import os

import streamlit as st

NEW_COUNTER_URL = "https://ryokusui-counter.web.app/"
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    st.set_page_config(layout="centered", page_title=st.secrets["NAME"], page_icon="img/r.ico")
    images = [os.path.join(HERE, "img", name) for name in ("logo.png", "main.jpg", "sub.jpg")]
    if all(os.path.exists(p) for p in images):
        for p in images:
            st.image(p, width="stretch")
    else:
        st.warning(st.secrets["PIC_ERR"])
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.link_button(st.secrets["USE_BUTTON"], NEW_COUNTER_URL, type="primary")


if __name__ == "__main__":
    main()
