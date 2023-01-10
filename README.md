# Overview
## What is This?
#### This project is a web app that provides a number of visualizations of user-provided text data. The visualizations are informed by predictions derived from a relation extraction algorithm developed by this project: https://huggingface.co/Babelscape/rebel-large

## What do you mean by "visualizations" of text?
#### This is described in some more detail below, but basically I am referring to lists, charts, maps, annotations --- anything that is a visual representation of text data meant to aid in understanding the content of the original text. 

## What is Relation Extraction?
#### The easiest way to explain what relation extraction is, I think, to provide an example:
#### Let's say you have the following sentence: "Game Freak and Nintendo announced the creation of a new Pokemon game."
#### Using a relation extraction algorithm, you will probably get the following results (or something simialr), called relational triples:
1. Game Freak, partner_of, Nintendo
2. Nintendo, parnter_of, Nintendo
3. Game Freak, producer, game
4. Pokemon, type_of, game
#### In other words, relation extraction algorithms do the following:
- Given a sentence that has two or more proper nouns, output a predicted relation between those two nouns.

## Who is this for?
#### Anyone who is curious about what relation extraction can do.
#### Anyone who wants to know what people/groups/organizations/places are mentioned in a text before they read it. 
#### Anyone else who is willing to go through the steps of downloading this repo.

## What kind of articles should I input?
#### The model was trained mainly on news articles, so it performs best with text that is in that category. You can run the on any text (I tested it on a mystery novel, for example) but again, the results are best with text that is very generally in the category of politics/news. This includes both periodicals and academic journals. 

## What Can I Expect to See on the App?
### Once you input your text in the text box on the first screen of the app and click submit, the model will run. 
### The results come in four different flavors:
1. List of relational triples: lists all of the extracted triples. 
2. Annotation: text in paragraph format but highlighted and annotated with relational triples. 
3. Map: countries and cities that appear in the text, on a Plotly map (so you can zoom in).
4. CSV: displays the data in a spread-sheet-like format, with the option to download as a CSV file. 

# How to Run It Locally - Instructions for Downloading and Installing 
## 1. Required Packages and Versions:

| Package       | Version       |  Specific Functions/Modules 		  | Notes 													|		
| ------------- |:-------------:|:---------------------------------:|:----------------------------------------:	|
| Python        | 3.9 			  | NA										  | https://www.python.org/downloads/				|
| Flask       	 | 2.2    		  | render_template, url_for, request | pip install Flask									|
| Werkzeug 		 | 2.2      	  | NA   								     | comes with Flask										|
| IPython		 | 8.4			  | display, HTML                     | pip install ipython									|
| transformers	 | 4.18          | pipeline							     | pip install transformers							|
| plotly		    | 5.11			  | express, graph_objects			     | pip install plotly									|
| requests		 | 2.28			  | NA								        | pip install requests								|
| spacy			 | 3.3 			  | Language 							     | pip install -U pip setuptools wheel,			|
| 				    |				     | 									        | pip install -U spacy,								|
| 				    |				     |									        | python -m spacy download en_core_web_sm		|
| regex			 | 2022.7.25	  | NA								        | pip install regex									|
| pandas		    | 1.4			  | NA								        | pip install pandas 									|
| fuzzywuzzy	 | 0.18			  | NA								        | pip install python-Levenshtein,					|
| 				    |					  |     									     | pip install fuzzywuzzy								|
| hashlib		 |				     | NA								        | comes with python 									|
| json			 |				     | NA								        | comes with python 									|
| typing		    |				     | NA								        | comes with python 									|

## 2. How to Run the App:

1. Create a new folder in whatever directory you would like to work in (eg: Desktop)
2. Pip install the packages in the table above (just the ones that say "pip install..." in the notes, PLUS python)
	**NOTE** spacy and fuzzywuzzy require multiple pip installs 
3. Download the files from this git repo according to the following hierarchy:
	In your new folder created in step 1 (top level)
	- models folder
	- app.py
	- class_funcs.py
	- create_dataframes.py
	- worldcities.csv
	- static folder
	- templates folder
4. Once all files are downloaded, navigate to your top level folder and from your command line, type "python app.py"
5. After a minute or so of set-up, you should get a message that says:

![Alt text](/ready_to_run.png?raw=true)

6. Copy the url for local host "http://127.0.0.1:5000" into your browser (**Safari recommended**)
7. You should see the starting page of the app:

![Alt text](/start_page.png?raw=true)

8. Copy and paste whatever text you like. 

9. Click the **Submit** button.

10. The model will run. This will take a while. The model runs best on a GPU, but I have it configured to run on a CPU in this iteration.

11. Once the model is finished, you can click on any of the buttons on the right for visualization options. 

### NOTE ###
In the current iteration of the app, it won't be really obvious when the model is complete, so use this as a guideline:

**The text you pasted into the input box will remain on the screen while the model is running. Once the text box becomes *blank again*, that will be the indicator that the feature buttons are ready to be used.**

# Directories and Files:
## The Files in the Project are Organized as Follows


![Alt text](/file_structure.png?raw=true)


# Citation
#### Citations for specific pieces of code are documented in the code files themselves, but I relied heavily on the following two resources to learn how to implement the model:
1. Author: Tomaz Bratanic
   URL: https://towardsdatascience.com/extract-knowledge-from-text-end-to-end-information-extraction-pipeline-with-spacy-and-neo4j-502b2b1e0754
2. Authors: Pere-Lluís Huguet Cabot, Roberto Navigli
   URL: https://github.com/Babelscape/rebel/blob/main/docs/EMNLP_2021_REBEL__Camera_Ready_.pdf

