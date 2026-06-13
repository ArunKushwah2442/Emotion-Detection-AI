import streamlit as st
import pickle

# ---------------- PAGE ----------------

st.set_page_config(
    page_title="Emotion Detection AI",
    page_icon="🧠",
    layout="centered"
)

# ---------------- MODEL ----------------

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ---------------- EMOTIONS ----------------

emotion = {
    0: ("😢", "Sadness", "#3498db",
        "Your text reflects sadness or emotional pain."),
    1: ("😠", "Anger", "#e74c3c",
        "Your text expresses frustration or anger."),
    2: ("❤️", "Love", "#ff4d88",
        "Your text expresses affection and care."),
    3: ("😲", "Surprise", "#9b59b6",
        "Your text reflects surprise or amazement."),
    4: ("😨", "Fear", "#f39c12",
        "Your text indicates fear or uncertainty."),
    5: ("😊", "Joy", "#2ecc71",
        "Your text expresses happiness and positivity.")
}

# ---------------- CSS ----------------

st.markdown("""
<style>

.block-container{
padding-top:2rem;
padding-bottom:2rem;
max-width:850px;
}

.title{
text-align:center;
font-size:52px;
font-weight:700;
color:white;
}

.subtitle{
text-align:center;
font-size:20px;
color:#B8B8B8;
margin-bottom:30px;
}

.footer{
text-align:center;
color:gray;
margin-top:50px;
font-size:14px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------

st.markdown(
    '<div class="title">🧠 Emotion Detection AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Analyze human emotions from text using Natural Language Processing.</div>',
    unsafe_allow_html=True
)

# ---------------- INPUT ----------------

text = st.text_area(
    "✍️ Enter your text",
    height=170,
    placeholder="Example: Today was one of the happiest days of my life..."
)

# ---------------- BUTTON ----------------

if st.button("🚀 Detect Emotion", use_container_width=True):

    if text.strip():

        clean = text.lower()

        data = vectorizer.transform([clean])

        pred = model.predict(data)[0]

        emoji, name, color, desc = emotion[pred]

        confidence = None

        try:
            confidence = max(model.predict_proba(data)[0]) * 100
        except:
            pass

        st.markdown("")

        st.markdown(
            f"""
            <div style="
            background:{color};
            padding:35px;
            border-radius:18px;
            text-align:center;
            color:white;">
            
            <div style="font-size:70px;">
            {emoji}
            </div>

            <div style="font-size:36px;
            font-weight:bold;">
            {name}
            </div>

            <br>

            <div style="font-size:18px;">
            {desc}
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        if confidence:

            st.write("")
            st.progress(int(confidence))
            st.caption(
                f"Confidence Score : {confidence:.2f}%"
            )

    else:

        st.warning("Please enter some text.")

# ---------------- EXAMPLES ----------------

with st.expander("💡 Try Example Texts"):

    st.write(
    """
😊 Joy:
I achieved my dream and I cannot stop smiling.

😢 Sadness:
I feel alone and disappointed with my life.

😠 Anger:
Someone betrayed my trust and I am furious.

❤️ Love:
My family means everything to me.

😨 Fear:
I am scared about what will happen tomorrow.

😲 Surprise:
I never expected to win this competition.
"""
    )

# ---------------- ABOUT ----------------

with st.expander("ℹ️ About this Project"):

    st.write(
    """
This project uses Natural Language Processing (NLP)
to detect human emotions from text.

Technologies Used:

• Python

• Scikit-learn

• TF-IDF Vectorization

• Logistic Regression

• Streamlit
"""
    )

# ---------------- FOOTER ----------------

st.markdown("---")

st.markdown("""
<div class="footer">

Made with ❤️ by <b>Arun Kushwah</b>

</div>
""", unsafe_allow_html=True)