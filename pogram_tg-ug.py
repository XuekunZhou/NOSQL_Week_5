import numpy as np
from flask import Flask, request, render_template
import pickle
import babel.numbers
import decimal

#initialiseren van de app
app = Flask(__name__)

#model opvragen die gedumpt is
model = pickle.load(open('model_TG-UG.pkl', 'rb'))

#default pagina van de applicatie
@app.route('/')
def home():
    return render_template('tg-ug.html')


#waarden uit de form ophalen op het moment de button wordt aangeklikt
@app.route('/v_vochtigheid',methods=['POST'])
def voorspellen_vochtigheid():
    #waarden uit de form ophalen
    int_features = [int(x) for x in request.form.values()]
    print(int_features)
    #waarden worden bewaard
    waarden = [np.array(int_features)]
    print(waarden)
    #waarden wordt naar model gestuurd en bewaard in een variabele:
    v = model.predict(waarden)
    print(v)

    hum = str(round(v[0],1))

    #Waarden wordt terugestuurd naar html pagina in variable antwoord
    return render_template('tg-ug.html', antwoord= 'Het voorspeld vochtigheid is: ' + hum + "%" )

if __name__ == "__main__":
    app.run(debug=True)