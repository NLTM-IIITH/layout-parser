FROM python:3.11.6
WORKDIR /model 
COPY . /model
RUN python -m pip install --upgrade pip
RUN pip install -r requirements_script_identification.txt
RUN wget https://github.com/kuna71/layout-parser-api/releases/download/weights/Synthetic_wordgenerator_AlexNet_100000_random-font-size_StateDicts.pt -O script-identification-iitb.pth 
CMD [ "python", "infer.py" ]