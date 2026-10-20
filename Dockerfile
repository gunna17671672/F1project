FROM python:3.11-slim

WORKDIR /app

# deps first so this layer gets cached and doesn't reinstall every time
# the app code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

# streamlit binds to localhost by default, which isn't reachable from
# outside the container
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
