""""
  This code was adapted from:
  1. Description: Running the Model (lines 33 - 74)
  	 Author: Tomaz Bratanic
  	 URL: https://towardsdatascience.com/extract-knowledge-from-text-end-to-end-information-extraction-pipeline-with-spacy-and-neo4j-502b2b1e0754
  	 Accessed: August 2022
  2. Descirption: Basic Flask Code Examples
     Author: Pallets
     URL: https://flask.palletsprojects.com/en/2.2.x/
     Accessed: September 2022
  3. Desciption: Flask Tutorial
     Author: Corey Schafer
     URL: https://www.youtube.com/playlist?list=PL-osiE80TeTs4UjLw5MM6OjgkjFeUxCYH
     Accessed: September 2022
  4. Description: Displaying Plotly Graphs in a Flask App
     Author: Alan Jones
     URL: https://towardsdatascience.com/web-visualization-with-plotly-and-flask-3660abf9c946
     Accessed: November 2022
  5. Description: Displaying Plotly Geograhhic Maps
  	 Author: Plotly Documentation
  	 General layout: https://plotly.com/python/reference/layout/geo/
  	 Bubble maps: https://plotly.com/python/bubble-maps/
  	 Chloropleth maps: https://plotly.com/python/choropleth-maps/
"""

from flask import Flask, render_template, url_for, request
import pandas as pd
import spacy
import crosslingual_coreference
import requests
import re
import hashlib
from spacy import Language
from typing import List
from spacy.tokens import Doc, Span
from transformers import pipeline
from class_funcs import *
from create_dataframes import *
import plotly
import plotly.express as px
import plotly.graph_objects as go
import json
from IPython.display import HTML

app = Flask(__name__)

############ initial set-up for the models: ##############

# set device parameter to CPU (this number indicates the number of the GPU if GPU is used)
DEVICE = -1

# create coref empty model
coref = spacy.load('en_core_web_sm', disable=['ner', 'tagger', 'parser', 'attribute_ruler', 'lemmatizer'])
coref.add_pipe(
    "xx_coref", config={"chunk_size": 2500, "chunk_overlap": 2, "device": DEVICE})

# create rel_ext empty model
rel_ext = spacy.load('en_core_web_sm', disable=['lemmatizer', 'attribute_rules', 'tagger'])
rel_ext.add_pipe("rebel", config={
    'device':DEVICE, 
    'model_name':'Babelscape/rebel-large'})

###########################################

# this is the user's input

def run_coref(input_text):

	coref_text = coref(input_text)._.resolved_text

	return coref_text

def run_rel_ext(coref_text):

	doc = rel_ext(coref_text)

	return doc

def get_raw_results(doc):

	# initial results of the rel_ext model will be stores in this list
	results = []

	for value, rel_dict in doc._.rel.items():
	    one_row = (rel_dict)
	    results.append(one_row)

	return results


# this dict will hold variables that I want to access from multiple functions 
var_dict = {}

@app.route("/about")
def about():

	return render_template('about.html')

@app.route("/")
@app.route("/start", methods=['GET', 'POST'])
def start():
	if request.method == 'GET':

		# Just render the initial form, to get input
		return(render_template('start.html'))

	if request.method == 'POST':

		if request.form.get('user_input_text'):

		# Get the input from the user.
			user_input_text = request.form['user_input_text']
			var_dict['original_text'] = user_input_text

			coref_text = run_coref(user_input_text)
				
			# this takes the longest to run
			doc = run_rel_ext(coref_text)

			# this is the model result
			raw_results = get_raw_results(doc)
			var_dict['raw_results'] = raw_results

			# this is the coreferenced text
			var_dict['coref_text'] = coref_text
						
			return render_template('start.html', input_text = user_input_text)

@app.route("/result", methods=['GET', 'POST'])
def result():

	raw_results = var_dict['raw_results']

	init_df = create_init_df(raw_results)

	results_df = create_results_dict(init_df)
	results = results_df.to_dict('index')

	dropdownMenu_n1 = list(results_df['noun_1'].unique()) 
	dropdownMenu_rl = list(results_df['related_by'].unique()) 
	dropdownMenu_n2 = list(results_df['noun_2'].unique()) 

	dropdownMenu_n1.append('ALL')
	dropdownMenu_rl.append('ALL')
	dropdownMenu_n2.append('ALL')


	if request.method == 'POST':

		if request.form.get('filter_n1'):

				dropdownMenu_n1 = dropdownMenu_n1
				dropdownMenu_rl = dropdownMenu_rl
				dropdownMenu_n2 = dropdownMenu_n2

				selected_n1 = request.form.get('dropdownMenu_n1')

				if selected_n1 == '' or selected_n1 == 'ALL':

					return render_template('result.html', results = results, dropdownMenu_n1 = dropdownMenu_n1,
						dropdownMenu_rl = dropdownMenu_rl, dropdownMenu_n2 = dropdownMenu_n2)

				else: 
					selected_text_n1 = str(selected_n1)

					cond = results_df['noun_1'] == selected_text_n1
					selected_df = results_df[cond]

					filtered_results = selected_df.to_dict('index')

					return render_template('result.html', selected_text_n1 = selected_text_n1, results = filtered_results,
						dropdownMenu_n1 = dropdownMenu_n1, dropdownMenu_rl = dropdownMenu_rl, dropdownMenu_n2 = dropdownMenu_n2)

		elif request.form.get('filter_rl'):

			dropdownMenu_n1 = dropdownMenu_n1
			dropdownMenu_rl = dropdownMenu_rl
			dropdownMenu_n2 = dropdownMenu_n2

			selected_rl = request.form.get('dropdownMenu_rl')

			if selected_rl == '' or selected_rl == 'ALL':

				return render_template('result.html', results = results, dropdownMenu_n1 = dropdownMenu_n1,
					dropdownMenu_rl = dropdownMenu_rl, dropdownMenu_n2 = dropdownMenu_n2)

			else: 

				selected_text_rl = str(selected_rl)

				cond = results_df['related_by'] == selected_text_rl
				selected_df = results_df[cond]

				filtered_results = selected_df.to_dict('index')

				return render_template('result.html', selected_text_rl = selected_text_rl, results = filtered_results,
					dropdownMenu_n1 = dropdownMenu_n1, dropdownMenu_rl = dropdownMenu_rl, dropdownMenu_n2 = dropdownMenu_n2)

		elif request.form.get('filter_n2'):

			dropdownMenu_n1 = dropdownMenu_n1
			dropdownMenu_rl = dropdownMenu_rl
			dropdownMenu_n2 = dropdownMenu_n2

			selected_n2 = request.form.get('dropdownMenu_n2')

			if selected_n2 == '' or selected_n2 == 'ALL':

				return render_template('result.html', results = results, dropdownMenu_n1 = dropdownMenu_n1,
					dropdownMenu_rl = dropdownMenu_rl, dropdownMenu_n2 = dropdownMenu_n2)

			else: 

				selected_text_n2 = str(selected_n2)

				cond = results_df['noun_2'] == selected_text_n2
				selected_df = results_df[cond]

				filtered_results = selected_df.to_dict('index')

				return render_template('result.html', selected_text_n2 = selected_text_n2, results = filtered_results,
					dropdownMenu_n1 = dropdownMenu_n1, dropdownMenu_rl = dropdownMenu_rl, dropdownMenu_n2 = dropdownMenu_n2)

	else:

		return render_template('result.html', results = results, dropdownMenu_n1 = dropdownMenu_n1,
			dropdownMenu_rl = dropdownMenu_rl, dropdownMenu_n2 = dropdownMenu_n2)

@app.route("/stats")
def stats():

	# create another "full" DF (with the NER tags)

	original_text = var_dict['original_text']
	coref_text = var_dict['coref_text']

	raw_results = var_dict['raw_results']

	init_df = create_init_df(raw_results)
	annot_df = create_annot_df(init_df, original_text, coref_text)
	id_df = create_id_df(annot_df)
	df = create_NER_df(id_df)

	# list from which to create all the buttons 

	list_1 = list(df['noun_1'])
	list_2 = list(df['noun_2'])

	all_ents = list_1 + list_2
	all_ents_lc = []

	for ent in all_ents:
	    ent = ent.lower()
	    all_ents_lc.append(ent)

	set_a = set(all_ents_lc)
	all_unique = list(set_a)


    # dict of dicts where each ent is its own dict (row)
	total_dict = create_stats_dict(df)[0]
	print('printing from stats page... total dict', total_dict)
	
	return render_template('stats.html', total_dict = total_dict, all_unique = all_unique)


#enable filtering for specific columns in the CSVs 
@app.route("/csv")
def csv():
	
	original_text = var_dict['original_text']
	coref_text = var_dict['coref_text']

	raw_results = var_dict['raw_results']

	init_df = create_init_df(raw_results)
	annot_df = create_annot_df(init_df, original_text, coref_text)
	id_df = create_id_df(annot_df)
	ner_df = create_NER_df(id_df)

	table = HTML(ner_df.to_html(classes='table table-stripped'))

	csv_data = ner_df.to_csv()
	
	return render_template('csv.html', csv_data = csv_data, table = table)

@app.route("/annotation")
def annotation():

	original_text = var_dict['original_text']
	coref_text = var_dict['coref_text']

	raw_results = var_dict['raw_results']
	init_df = create_init_df(raw_results)
	annot_df = create_annot_df(init_df, original_text, coref_text)

	sents = list(annot_df['coref_sent'].unique())

	to_html = []
	to_js = {}

	for sent in sents:
	    
	    html_dict = {}
	    
	    cond = annot_df['coref_sent'] == sent
	    filtered = annot_df[cond]

	    yellow = list(filtered['noun_1'].unique())
	    
	    for word in yellow:
	        
	        sub_dict = {}
	        cond = filtered['noun_1'] == word
	        subfilt = filtered[cond]
	        
	        this_word = []
	        for i in range(subfilt.shape[0]):
	            dub = subfilt.iloc[i]['double']
	            this_word.append(dub)
	        
	        to_js[word] = this_word
	       
	    html_dict['sentence'] = sent
	    html_dict['yellows'] = yellow

	    to_html.append(html_dict)
	  
	return render_template('annotation.html', original_text = original_text, to_html = to_html, 
		to_js = to_js)

@app.route("/map")
def map():

	raw_results = var_dict['raw_results']
	ners_list = []

	init_df = create_init_df(raw_results)

	for i in range(len(raw_results)):
		ners_list.append(raw_results[i]['NER_tags'])
		
	output = get_map_data(ners_list, init_df)

	countries_set = output[0]
	cities_dict = output[1]

	cities_df = pd.DataFrame.from_dict(cities_dict, orient = 'index', columns=['lat', 'long'])
	
	countries_df = pd.DataFrame(countries_set)
	countries_df = countries_df.rename(columns={0: 'country'})

	# if there are no countries extracted, print an empty map anyway
	if countries_df.shape[0] == 0:
		fig = px.choropleth(countries_df, projection = 'natural earth')
	else:
		fig = px.choropleth(countries_df, locations = countries_set, locationmode = 'country names', 
	    			projection = 'natural earth', color = countries_df['country'])

	for i in range(cities_df.shape[0]):
	    fig.add_trace(go.Scattergeo(
	        locationmode = 'country names',
	        lon = cities_df['long'],
	        lat = cities_df['lat'],
	        text = cities_df.index,
	        marker = dict(
	            color = 'red',
	            line_color = 'rgb(40,40,40)',
	            line_width = 0.75,
	            size = 6.0)))

	fig.update_layout(
        title_text = 'Cities and Countries Mentioned in text',
        showlegend = False,
        autosize = False,
        width = 1000,
        height = 800,
        geo = dict(
            scope = 'world',
            landcolor = 'rgb(217, 217, 217)',
            showcountries = True,
            projection = dict(type = 'natural earth')
        )
    )

	graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

	return render_template('map.html', graphJSON=graphJSON)


@app.route("/index")
def index():
    sentence = "Jinja is a fast, expressive, extensible templating engine. Special placeholders in the template allow writing code similar to Python syntax. Then the template is passed data to render the final document."
    data = {"sentence": Markup(sentence.replace("template", "<mark>template</mark>"))}
    return render_template("index.html", data=data)


if __name__ == "__main__":
	# turn debug mode off --- otherwise it will try to run the model again... 
	app.run(debug=False)

