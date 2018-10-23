# -*- coding: utf-8 -*-
"""
Router for the endpoint of the AI-sport-recommendations API

exemple of local url: 
    http://127.0.0.1:5000/AI-sport-recommendations?user_ids=50016568329,139023213&results=3&new_sports=True

@author: AI team
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
from main import predict

app = Flask(__name__)
CORS(app)

@app.route('/AI-sport-recommendations')
def get_recommendations():
    #get user ids
    user_ids = [int(i) for i in request.args.get('user_ids').split(',')]    
    #get the number of recommendations desired
    num_recommendations = int(request.args.get('results'))
    #get if we want to filter the sports in the training set or not
    new_sports = int(request.args.get('new_sports'))
    if new_sports == 1:
        new_sports = True
    else:
        new_sports = False
    
    #return recommendations as a json   
    return jsonify(predict(user_ids, k=num_recommendations, filter_train=new_sports, specific_countries=['CA']))
    
if __name__ == '__main__':
    app.run()