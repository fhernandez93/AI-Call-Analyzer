import sqlite3

# Connect to the database. This will create a new file named 'mydatabase.db' if it doesn't exist.
conn = sqlite3.connect('OPTcallsAnalytics.db')
cursor = conn.cursor()

# Create a new table

cursor.execute('''
CREATE TABLE IF NOT EXISTS Customers (
    clientID TEXT,
    name TEXT NOT NULL,
               
    PRIMARY KEY (clientID)
)
''')
               
cursor.execute('''
CREATE TABLE IF NOT EXISTS Users (
    Name TEXT,
    User TEXT NOT NULL,
    Password TEXT NOT NULL,
    KeySecure TEXT,
    PRIMARY KEY (User)
)
''')
               

cursor.execute('''
CREATE TABLE IF NOT EXISTS customersCalls (
    clientID TEXT,
    name TEXT NOT NULL,
    date DATE,
    EmployeeName TEXT NOT NULL,
    callType TEXT NOT NULL,
    recordingID TEXT,
    fileName,
    cleanTranscription,
    Summary,           
               
    PRIMARY KEY (clientID, recordingID)
)
''')
               
cursor.execute('''
CREATE TABLE IF NOT EXISTS callsRecords (
    clientID TEXT,
    recordingID TEXT,
    date DATE,
    speaker TEXT,
    confidence TEXT,
    text TEXT,
    sentiment_score_neg TEXT,
    sentiment_score_neu TEXT,
    sentiment_score_pos TEXT,
    sentiment_score_overall TEXT
)
''')
               
# Commit the transaction and close the connection
conn.commit()
conn.close()