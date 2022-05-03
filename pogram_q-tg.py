import numpy as np
from flask import Flask, request, render_template
import pickle
import babel.numbers
import decimal

#initialiseren van de app
app = Flask(__name__)

#model opvragen die gedumpt is
model = pickle.load(open('model_Q-TG.pkl', 'rb'))

#default pagina van de applicatie
@app.route('/')
def home():
    return render_template('q-tg.html')


#waarden uit de form ophalen op het moment de button wordt aangeklikt
@app.route('/v_temperatuur',methods=['POST'])
def voorspellen_temperatuur():
    #waarden uit de form ophalen
    int_features = [int(x) for x in request.form.values()]
    print(int_features)
    #waarden worden bewaard
    waarden = [np.array(int_features)]
    print(waarden)
    #waarden wordt naar model gestuurd en bewaard in een variabele:
    v = model.predict(waarden)
    print(v)

    temp = str(round(v[0] * 0.1,1))

    #Waarden wordt terugestuurd naar html pagina in variable antwoord
    return render_template('q-tg.html', antwoord= 'Het voorspeld temperatuur is: ' + temp + " graden C" )

if __name__ == "__main__":
    app.run(debug=True)