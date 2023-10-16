import pickle


def load(filename: str):
    with open(filename, 'rb') as f_in:
        return pickle.load(f_in)


dv = load('dv.bin')
model = load('model1.bin')

score_data = {"job": "retired", "duration": 445, "poutcome": "success"}

X = dv.transform([score_data])
y_pred = model.predict_proba(X)[0, 1]

print(y_pred)
