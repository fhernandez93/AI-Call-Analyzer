from python:3.9.6
cmd mkdir -p /app
WORKDIR /app
EXPOSE 8501
copy requirements.txt ./requirements.txt
run apt-get install gcc
# Install gcc, update apt-get, and install unixODBC development package
RUN apt-get update 
RUN apt-get install -y gcc 
RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - 
RUN curl https://packages.microsoft.com/config/debian/11/prod.list  > /etc/apt/sources.list.d/mssql-release.list 

RUN curl -sL https://aka.ms/InstallAzureCLIDeb | bash
RUN apt-get update 
RUN ACCEPT_EULA=Y apt-get install -y msodbcsql18 
RUN apt-get install -y unixodbc-dev
run pip3 install pandas
run pip3 install matplotlib
run pip3 install plotly
run pip3 install spacy
run pip3 install torch==2.0.1
run pip3 install credential==0.0.2
run pip3 install python-dotenv==1.0.0
run pip3 install --no-binary :all: pyodbc
run python3 -m spacy download en_core_web_sm
run pip3 install -r requirements.txt
copy . .
EXPOSE 8501
run az login
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health
ENTRYPOINT ["streamlit", "run"]
CMD ["Home.py"]
