FROM python:3.9-slim

WORKDIR /frontend
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY . .

EXPOSE 8500
CMD ["streamlit", "run", "streamlit.py", "--server.address", "0.0.0.0", "--server.port", "8500"]
