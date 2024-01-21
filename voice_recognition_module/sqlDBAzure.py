import pyodbc
from variables import *
# Connect to the database. This will create a new file named 'mydatabase.db' if it doesn't exist.
Driver="Driver={ODBC Driver 18 for SQL Server};Server=tcp:opt-call-analyzer-server.database.windows.net,1433;Database=OPTCallsAnalytics;Uid="+str(SQLUSER)+";Pwd={"+str(SQLPASS)+"};Encrypt=yes;TrustServerCertificate=no;Connection Timeout=250;"
conn = pyodbc.connect(Driver)
cursor = conn.cursor()
# Create a new table

cursor.execute('''
    if not exists (select * from sysobjects where name='Customers' and xtype='U')
    create table Customers (
        clientID INT IDENTITY(1,1) PRIMARY KEY,
        name varchar(max) NOT NULL
    )
''')

cursor.execute('''
    if not exists (select * from sysobjects where name='Employees' and xtype='U')
    create table Employees (
    EmployeeId INT IDENTITY(1,1) PRIMARY KEY,
    name varchar(max) NOT NULL
    )
''')

cursor.execute('''
    if not exists (select * from sysobjects where name='Phrases' and xtype='U')
    create table Phrases (
    PhraseId INT IDENTITY(1,1) PRIMARY KEY,
    phrase varchar(max) NOT NULL
    )
''')


cursor.execute('''
    if not exists (select * from sysobjects where name='Users' and xtype='U')
    create table Users (
    UserID int IDENTITY(1,1) PRIMARY KEY,
    Name TEXT,
    UserName TEXT NOT NULL,
    Password TEXT NOT NULL,
    KeySecure TEXT,
    Role varchar(max),
    )
''')
               
cursor.execute('''
    if not exists (select * from sysobjects where name='customersCalls' and xtype='U')
    create table customersCalls (
    clientID varchar(max),
    name varchar(max) NOT NULL,
    date DATE,
    EmployeeName varchar(max) NOT NULL,
    callType varchar(max) NOT NULL,
    recordingID varchar(max) NOT NULL,
    fileName varchar(max),
    cleanTranscription varchar(max),
    Summary varchar(max)      
    )
''')   

cursor.execute('''
    if not exists (select * from sysobjects where name='callsRecords' and xtype='U')
    create table callsRecords (
    clientID varchar(max),
    recordingID varchar(max),
    date DATE,
    speaker varchar(max),
    confidence varchar(max),
    text varchar(max),
    sentiment_score_neg varchar(max),
    sentiment_score_neu varchar(max),
    sentiment_score_pos varchar(max),
    sentiment_score_overall varchar(max),
    toxicity varchar(max),
    severe_toxicity varchar(max),
    obscene varchar(max),
    threat varchar(max),
    insult varchar(max),
    identity_attack varchar(max)
    )
''')
               
# Commit the transaction and close the connection
conn.commit()
conn.close()