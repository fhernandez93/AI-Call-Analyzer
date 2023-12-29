import streamlit as st
import streamlit_authenticator as stauth
import sqlite3
import pyodbc
import re
import pyotp
import qrcode
import os
import numpy as np
import base64
from dotenv import load_dotenv
from cryptography.fernet import Fernet
from variables import *

load_dotenv()



if 'userInfo' not in st.session_state:
        st.session_state.userInfo = {}



def encrypt_key(key, plaintext):
    """
    Encrypt the plaintext using the provided key.

    :param key: The encryption key as a string.
    :param plaintext: The plaintext to encrypt.
    :return: The encrypted text as a string.
    """
    f = Fernet(key)
    ciphertext = f.encrypt(plaintext.encode('utf-8'))
    return base64.urlsafe_b64encode(ciphertext).decode('utf-8')


def generate_base32_key(length=20):
    """
    Generate a random base32 encoded key.

    :param length: The number of random bytes to generate. Default is 20, which will generate a 32 character base32 encoded string.
    :return: The base32 encoded key as a string.
    """
    random_bytes = os.urandom(length)
    return base64.b32encode(random_bytes).decode('utf-8')

def check_user(mail, users):
    mails = [item[1] for item in users]
    if mail in mails: 
        return True
    else: 
        return False

regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b'
 
# Define a function for
# for validating an Email
def check_mail(email):
    # pass the regular expression
    # and the string into the fullmatch() method
    if(re.fullmatch(regex, email)):
        return True
 
    else:
        False
 


def runAddUser():
    
    # Connect to the database. This will create a new file named 'mydatabase.db' if it doesn't exist.
    #conn = sqlite3.connect('OPTcallsAnalytics.db')
    # Driver="DRIVER={ODBC Driver 18 for SQL Server};Server=tcp:opt-calls-analytics.database.windows.net,1433;Database=OPTCallsAnalytics;Uid="+str(SQLUSER)+";Pwd={"+str(SQLPASS)+"};Encrypt=yes;TrustServerCertificate=no;Connection Timeout=100";
    Driver="DRIVER={ODBC Driver 18 for SQL Server};Server=tcp:opt-calls-analytics.database.windows.net,1433;Database=OPTCallsAnalytics;Uid="+str(SQLUSER)+";Pwd={"+str(SQLPASS)+"};Encrypt=yes;TrustServerCertificate=no;Connection Timeout=100";
    conn = pyodbc.connect(Driver)
    c = conn.cursor()
    st.title("Sign Up")
    #with st.form(key='signup',clear_on_submit=True):
    #Function to hash passwords
    st.session_state.access = False
    name= st.text_input("Full Name:")
    user = st.text_input("email:")
    password = st.text_input("Password:",type='password')

    col1,col2,col3,col4,col5,col6,col7= st.columns(7)
    if col2.button("Cancel"):
        st.session_state.login = True
        st.experimental_rerun()

    if col1.button("Sign Up"):
        if user and name and password:
            c.execute("SELECT * FROM Users")
            users = c.fetchall()
            if check_user(user,users):
                st.warning("This email already exist")
            else:
                if check_mail(user):
                    hash_pass = stauth.Hasher([password]).generate()[0]
                    st.session_state.userInfo = {'name':name,
                                                  'user':user,
                                                  'password':hash_pass}                   
                    st.session_state.qrSecurity = True
                    st.experimental_rerun()
                else: 
                    st.warning("Insert a valid email address!")
        else: 
            st.warning("Please complete all fields!")



        #st.form_submit_button('Sign Up')

        

    # Commit the transaction and close the connection
    conn.commit()
    conn.close()


##### 2FA setup #####

def addQR():
   
    #conn = sqlite3.connect('OPTcallsAnalytics.db')
    Driver="DRIVER={ODBC Driver 18 for SQL Server};Server=tcp:opt-calls-analytics.database.windows.net,1433;Database=OPTCallsAnalytics;Uid="+str(SQLUSER)+";Pwd={"+str(SQLPASS)+"};Encrypt=yes;TrustServerCertificate=no;Connection Timeout=100";
    conn = pyodbc.connect(Driver)
    c = conn.cursor()
    userInfo = st.session_state.userInfo

    if st.session_state.tempKey =="":
        st.session_state.tempKey = generate_base32_key()

    uri = pyotp.totp.TOTP(st.session_state.tempKey).provisioning_uri(name=userInfo['user'], issuer_name="Optumus")


    qr = qrcode.QRCode(
    version=1,
    box_size=5,  # controls the size of the QR Code, (box_size=10 means each box in the QR code will be 10x10 pixels)
    border=5,
            )
    qr.add_data(uri)
    qr.make(fit=True)
    img = qr.make_image(fill='black', back_color='white')
    qr_np = np.array(img.convert("RGB"))


    st.text("Please scan the following QR code using an app like Google Authenticator or Authy,\nand insert the generated code below.")
    st.image(qr_np)


    code = st.text_input("Generated Code:")

    if st.button("Submmit"):
        totp = pyotp.TOTP(st.session_state.tempKey)
        if(totp.verify(code)):
            encrypted_key = encrypt_key(TWOFA_KEY, st.session_state.tempKey)
            c.execute(f"""INSERT INTO Users (Name, UserName, Password, KeySecure) VALUES ('{st.session_state.userInfo['name']}',
                       '{st.session_state.userInfo['user']}','{st.session_state.userInfo['password']}',
                       '{encrypted_key}')""") 
            conn.commit()
            st.session_state.success_signup = True
            st.session_state.userInfo = {}
            st.experimental_rerun()

        else: 
            st.error("Try Again")




    conn.close()

def successQR():
    st.success("Successful Sign Up!")
    st.balloons()
    if st.button("Continue"):
        st.session_state.access = False
        st.session_state.login = True
        st.session_state.qrSecurity = False
        st.session_state.success_signup = False
        st.session_state.tempKey = ""
        st.experimental_rerun()
