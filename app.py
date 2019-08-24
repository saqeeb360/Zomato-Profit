from flask import Flask ,render_template,request,url_for
import pickle
import numpy as np

with open(f'model/zomato_profit.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__,template_folder = 'templates')

@app.route('/',methods=['GET','POST'])
def main():
	if request.method == 'GET':
		return render_template('main.html')
	if request.method == 'POST':
		population = float(request.form['popu'])
		population = np.array([population]).reshape(1,-1)
		prediction = model.predict(population)
		return render_template('main.html',
                         original_input={'population':population},
                         result=prediction
                         )


if __name__ == '__main__':
    app.run()

