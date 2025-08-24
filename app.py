import streamlit as st

#Main about page
About_page=st.Page(
    page="Pages/about.py",
    title="About",
    icon="😉",
    default=True
)

#Backtesing Strategy page
Main_page=st.Page(
    page="Pages/main.py",
    title="Health Engine",
    icon="📊"
)


#for navigation
pg=st.navigation({
    "Info":[About_page],
    "Health Diagnose":[Main_page],
})

st.logo("Asset/Sidebar.png")
st.sidebar.text("Analysis with 💗 Sumit")

pg.run()