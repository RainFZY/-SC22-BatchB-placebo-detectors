# import requirements needed
from flask import Flask , render_template , request , redirect , url_for
from utils import get_base_url
import pandas as pd
from category_encoders import TargetEncoder
from sklearn.preprocessing import LabelEncoder
import pickle

# setup the webserver
# port may need to be changed if there are multiple flask servers running on same server
port = 12348
base_url = get_base_url(port)

# if the base url is not empty, then the server is running in development, and we need to specify the static folder so that the static files are served
if base_url == '/':
    app = Flask(__name__)
else:
    app = Flask(__name__, static_url_path=base_url+'static')

# set up the routes and logic for the webserver

def encode_data(data):
    x_columns = ['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
       'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
       'stalk-shape', 'stalk-surface-above-ring', 'stalk-surface-below-ring',
       'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-color',
       'ring-number', 'ring-type', 'spore-print-color', 'population',
       'habitat']
    test_df = pd.DataFrame(data, columns=x_columns)
    
    df = pd.read_csv("./mushrooms_Lucas.csv")
    df.drop(['stalk-root', 'veil-type'], axis=1, inplace = True)

    target = df['class']
    input_columns = df.loc[:, df.columns != "class"]
    target[target=="p"] = 1
    target[target=="e"] = 0
       
#     def labelencoder(df):
#         for c in df.columns:
#             if df[c].dtype=='object':
#                 df[c] = df[c].fillna('N')
#                 lbl = LabelEncoder()
#                 lbl.fit(list(df[c].values))
#                 df[c] = lbl.transform(df[c].values)
#         return df
#     df = labelencoder(df.copy())
#     target = labelencoder(target.copy())

    te = TargetEncoder(cols=input_columns.columns).fit(input_columns, target)
    test_df_encoded = te.transform(test_df)
    return test_df_encoded


@app.route(f'{base_url}', methods=["GET", "POST"])
def home():
    if request.method == "POST":
        values = [[i for i in request.form.values()]]
        print(values)
        input_x = encode_data(values)
        print(input_x)
        model = pickle.load(open('knn.sav', 'rb'))
        
        
        html_df = pd.DataFrame(values, columns=input_x.columns)
        df_html = html_df.to_html(classes="table table-dark")
        
        pred = model.predict(input_x)[0]
        print(pred)
        
        return render_template('index.html', values = "0 - Edible" if pred==0 else "1 - Poisonous", df_html = df_html)
#         return render_template('user_input_test.html', values = "0 - Edible" if pred==0 else "1 - Poisonous", df_html = df_html)
    return render_template('index.html')
#     return render_template('user_input_test.html')

# define additional routes here
# for example:
# @app.route(f'{base_url}/team_members')
# def team_members():
#     return render_template('team_members.html') # would need to actually make this page

if __name__ == '__main__':
    # IMPORTANT: change url to the site where you are editing this file.
    website_url = 'https://cocalc13.ai-camp.dev'
    
    print(f'Try to open\n\n    https://{website_url}' + base_url + '\n\n')
    app.run(host = '0.0.0.0', port=port, debug=True)
