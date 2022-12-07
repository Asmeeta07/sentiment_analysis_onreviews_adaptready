# sentiment_analysis_onreviews_adaptready

Project's Title. This is the name of the project. ...

THe project contains a 
1.jupyter notebook that shows how the a LSTM has been trained using GLove word Embeddings
2. A trained model 
3. A .py file that runs a flask application and predicts the text
4. A requirements.txt file


TO run the projects
1. python -r "requirements.txt"
2. please use "glove.6B.50d.txt" along with the code. I was not able to place the file in git since was a huge file.
3. python predict.py
4. When the application will be hosted on local host, using postman pass the below payload on a POST request

import requests

url = 'http://127.0.0.1:12345/'
myobj = {"reviews":"This is a good review"}
x = requests.post(url, json = myobj)

print(x.text)
