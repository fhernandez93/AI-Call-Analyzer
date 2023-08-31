import streamlit as st
import streamlit_authenticator as stauth
import sqlite3
import os
from dotenv import load_dotenv
from cryptography.fernet import Fernet
import base64
import pyotp

load_dotenv()


def decrypt_key(key, ciphertext):
    """
    Decrypt the ciphertext using the provided key.

    :param key: The encryption key as a string.
    :param ciphertext: The encrypted text to decrypt.
    :return: The decrypted text as a string.
    """
    f = Fernet(key)
    ciphertext_bytes = base64.urlsafe_b64decode(ciphertext.encode('utf-8'))
    plaintext = f.decrypt(ciphertext_bytes)
    return plaintext.decode('utf-8')

def login_screen():
 
    conn = sqlite3.connect('OPTcallsAnalytics.db')
    c = conn.cursor()
    
    col1,col2,col3,col4,col5,col6,col7= st.columns(7)
    if col7.button("Sign Up"):
        st.session_state.login = False
        st.experimental_rerun()
    
    c.execute("SELECT * FROM Users")
    users = c.fetchall()
    emails = []
    userids = []
    passwords = []

    for user in users: 
        userids.append(user[0])
        emails.append(user[1])
        passwords.append(user[2])


    #Let's create a credentials dictionary to mimic the YAML file required by streamlit_authenticator
    credentials = {'usernames':{}}

    for index in range(len(emails)):
        credentials['usernames'][emails[index]] = {'email':emails[index],'name':userids[index],'password':passwords[index]}


    authenticator = stauth.Authenticate(credentials, cookie_name='Streamlit2',  key = os.getenv('HASH_KEY'), cookie_expiry_days=0)


    name, authentication_status, username =  authenticator.login('Login','main')
    if username:
        if authentication_status:
            #st.session_state.access = True
            st.session_state.secureLogin = True
        elif authentication_status is False:
            st.error('Username/password is incorrect')
        elif authentication_status is None:
            st.warning('Please enter your username and password')
    
   
    
    #Commit the transaction and close the connection
    conn.commit()
    conn.close()

def secure_login():
    st.markdown(
        """
    <style>
        [data-testid="collapsedControl"] {
            display: none
        }
    </style>
    """,
        unsafe_allow_html=True,
    )
    st.text("Insert code generated by your 2FA app:")
    conn = sqlite3.connect('OPTcallsAnalytics.db')
    c = conn.cursor()
    with st.form(key='secure',clear_on_submit=True):

        code = st.text_input("Code")
        c.execute(f"SELECT * FROM Users where User = '{st.session_state.username}'")
        userKey = c.fetchone()[3]
        unEncryptedKey = decrypt_key(os.getenv('2fa_KEY'),userKey)
        totp = pyotp.TOTP(unEncryptedKey)
        if (totp.verify(code)):
            conn.commit()
            conn.close()
            st.session_state.access = True
            st.session_state.login = False
            st.session_state.qrSecurity = False
            st.session_state.success_signup = False
            st.session_state.secureLogin = False
            st.session_state.tempKey = ""
            st.experimental_rerun()
        else: 
            if code:
                st.warning("Try Again")

        st.form_submit_button('Continue')
        

    conn.commit()
    conn.close()