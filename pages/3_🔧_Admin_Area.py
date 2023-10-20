
# Import necessary libraries
import streamlit as st
import sqlite3

if not st.session_state["authentication_status"]:
    st.warning("You must log-in to see the content of this sensitive page! Reload page to login.")
    st.stop()  # App won't run anything after this line

#Database connection 
conn = sqlite3.connect('OPTcallsAnalytics.db')
c = conn.cursor()

st.markdown(
    """
<style>
button {
    height: auto;
    width: 200px !important;
    
}
</style>
""",
    unsafe_allow_html=True,
)
col1, col2 = st.columns([9,3]) 

employee_name = col1.text_input("Employee",placeholder="Employee Name",label_visibility='collapsed')
if col2.button("Add Employee"):
    if employee_name:
        c.execute("INSERT OR IGNORE INTO Employees (name) VALUES (?)",(employee_name,))
        conn.commit()
    else: 
        st.warning("Enter an Employee Name")
        

Client = col1.text_input("Client",placeholder="Client Name",label_visibility='collapsed')
if col2.button("Add Client"):
    print("Add")

Phrase = col1.text_input("Sentencex",placeholder="Sentence/keyword",label_visibility='collapsed')
if col2.button("Add Sentence/Keyword"):
    print("Add")

st.subheader("Select to review database")

col1_b, col2_b, col3_b = st.columns(3) 

if col1_b.button("Employees"):
    print("Add")

if col2_b.button("Clients"):
    print("Add")

if col3_b.button("Sentences/Words"):
    print("Add")