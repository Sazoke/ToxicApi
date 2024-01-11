FROM python:3.9-slim
#
COPY ./requirements.txt /requirements.txt
#
RUN pip install --no-cache-dir --upgrade -r /requirements.txt
#
COPY ./main.py main.py
#
EXPOSE 80
#
CMD ["python3", "main.py"]

