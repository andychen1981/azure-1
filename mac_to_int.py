# import the Flask class from the flask module
from flask import Flask, render_template, redirect, \
    url_for, request, session, flash, g

# create the application object
app = Flask(__name__)

@app.route('/mac_to_int')
def mac_to_int(): 
      
    mac_int = int('00:0a:95:9d:68:16'.translate(None, ":.- "), 16)

    print("MAC Int: ",mac_int)
    return mac_int


@app.route('/sum_id')
def sum_id(): 
      
    mac_int = int('00:0a:95:9d:68:16'.translate(None, ":.- "), 16)

    print("MAC Int: ",mac_int)
    return mac_int

    if (pat_code==0) and (pos=='upper'):
    	SELECT (mac_int || pat_code || '0000' || '1' || my_date) AS sum_id