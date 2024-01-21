
# Import necessary libraries
import streamlit as st
import sqlite3
import pyodbc
import pandas as pd
from authetication.add_user import *
from streamlit_extras.switch_page_button import switch_page


st.set_page_config(page_icon=':bar_chart:')
if not st.session_state["authentication_status"]:
    st.warning("You must log-in to see the content of this sensitive page! Reload page to login.")
    st.stop()  # App won't run anything after this line

if 'employees' not in st.session_state:
        st.session_state.employees = False
if 'clients' not in st.session_state:
        st.session_state.clients = False
if 'phrases' not in st.session_state:
        st.session_state.phrases = False

def local_css(file_name):
    with open(file_name) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)


#Database connection 
#conn = sqlite3.connect('OPTcallsAnalytics.db')
Driver="Driver={ODBC Driver 18 for SQL Server};Server=tcp:opt-call-analyzer-server.database.windows.net,1433;Database=OPTCallsAnalytics;Uid="+str(SQLUSER)+";Pwd={"+str(SQLPASS)+"};Encrypt=yes;TrustServerCertificate=no;Connection Timeout=250;";
conn = pyodbc.connect(Driver)
c = conn.cursor()
# Using the container's class for styling
local_css("style.css")

if st.button("Sign Up new user"):
    st.session_state.login = False
    st.session_state.qrSecurity = False
    st.session_state.success_signup = False
    st.session_state.secureLogin = False
    st.session_state.access = False
    st.session_state.tempKey = ""
    st.session_state["authentication_status"] = False
    st.cache_resource.clear()
    st.cache_data.clear()
    switch_page("Home")


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
#Function to add employees data
def add_employee_data():
    st.session_state.clients = False
    st.session_state.employees = True
    st.session_state.phrases = False
    st.session_state.employee_name = st.session_state.widget
    st.session_state.widget = ""
    if st.session_state.employee_name:
        c.execute("SELECT * FROM Employees WHERE cast(name as nvarchar(max))=?", (st.session_state.employee_name,))
        existing_employee = c.fetchone()
        if existing_employee:
            st.warning("Employee already exists in the database.")
        else: 
            c.execute("INSERT INTO Employees (name) VALUES (?)",(st.session_state.employee_name,))
            conn.commit()
            st.success("Employee Added to database")
    else: 
        st.warning("Enter an Employee Name")

if 'employee_name' not in st.session_state:
        st.session_state.employee_name = ""


employee_name = st.text_input("Press enter to add the record to the database",placeholder="Employee Name",key="widget",on_change=add_employee_data)


#Function to add clients data
def add_client_data():
    st.session_state.clients = True
    st.session_state.employees = False
    st.session_state.phrases = False
    st.session_state.client_name = st.session_state.widget_client
    st.session_state.widget_client = ""
    if st.session_state.employee_name:
        c.execute("SELECT * FROM Customers WHERE cast(name as nvarchar(max))=?", (st.session_state.client_name,))
        existing_employee = c.fetchone()
        if existing_employee:
            st.warning("Client already exists in the database.")
        else: 
            c.execute("INSERT INTO Customers (name) VALUES (?)",(st.session_state.client_name,))
            conn.commit()
            st.success("Client Added to database")
    else: 
        st.warning("Enter a Client Name")

if 'client_name' not in st.session_state:
        st.session_state.client_name = ""
Client = st.text_input("Press enter to add the record to the database",placeholder="Client Name", key="widget_client",on_change=add_client_data)

#Function to add clients data
def add_phrase_data():
    st.session_state.clients = False
    st.session_state.employees = False
    st.session_state.phrases = True
    st.session_state.phrase_name = st.session_state.widget_phrase
    st.session_state.widget_phrase = ""
    if st.session_state.phrase_name:
        c.execute("SELECT * FROM Phrases WHERE cast(phrase as nvarchar(max))=?", (st.session_state.phrase_name,))
        existing_employee = c.fetchone()
        if existing_employee:
            st.warning("Phrase already exists in the database.")
        else: 
            c.execute("INSERT INTO Phrases (phrase) VALUES (?)",(st.session_state.phrase_name,))
            conn.commit()
            st.success("Phrase Added to database")
    else: 
        st.warning("Enter a new phrase or sentence")

if 'phrase_name' not in st.session_state:
        st.session_state.phrase_name = ""


Phrase = st.text_input("Press enter to add the record to the database",placeholder="Sentence/keyword", key="widget_phrase",on_change=add_phrase_data)


st.subheader("Select to review database")

col1_b, col2_b, col3_b = st.columns(3) 

if col1_b.button("Employees"):
    st.session_state.employees = True
    st.session_state.clients = False
    st.session_state.phrases = False

if col2_b.button("Clients"):
    st.session_state.employees = False
    st.session_state.clients = True
    st.session_state.phrases = False

if col3_b.button("Sentences/Words"):
    st.session_state.employees = False
    st.session_state.clients = False
    st.session_state.phrases = True


if st.session_state.employees: 
    data = pd.read_sql("select * from Employees",conn)
    st.dataframe(data, hide_index=True,use_container_width=True)

if st.session_state.clients: 
    data_clients = pd.read_sql("select * from Customers",conn)
    st.dataframe(data_clients, hide_index=True,use_container_width=True)


if st.session_state.phrases: 
    data_phrase = pd.read_sql("select * from Phrases",conn)
    st.dataframe(data_phrase, hide_index=True,use_container_width=True)
