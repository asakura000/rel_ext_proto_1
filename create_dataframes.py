"""
This code adapted from documentation for pandas:
https://pandas.pydata.org/docs/
Accessed: August, September, October, November, December 2022

"""


import spacy
import pandas as pd
import fuzzywuzzy
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import re
import itertools

# order of build

# 1: init_df
# 2: with IDs
# 3: with NER tags
# 4: use NER df to build dict for stats pop-up windows

# input = the list of dictionaries that is the raw output from the rel_ext model
# output = pandas dataframe
# for some reason when I output this in html, it acts like the first 2 rows don't exist...
def create_init_df(results_list):
    
	initial = pd.DataFrame(results_list, columns = ["sentence", "relation", "head_span", "tail_span", "NER_tags"])
	head_span = initial["head_span"].apply(pd.Series)
	head_span.rename(columns = {"text" : "text_1", "id": 'id_1'}, inplace = True)
	tail_span = initial["tail_span"].apply(pd.Series)
	tail_span.rename(columns = {"text" : "text_2", "id": 'id_2'}, inplace = True)
	df = pd.concat([initial, head_span, tail_span], axis = 1)
	df = df.drop(columns=['head_span', 'tail_span'])
	df.rename(columns = {'sentence':'coref_sent',
	                     'relation':'related_by',
	                     'text_1': 'noun_1',
	                     'id_1': 'noun_1_id',
	                     'text_2': 'noun_2',
	                     'id_2': 'noun_2_id',
	                     'NER_tags': 'ner_tags'}, inplace = True)
	return df

def add_underscore(some_string):
    some_string = some_string.replace(' ', '_')

    return some_string

# code adapted from https://www.googleguide.com/linking.html
def get_google_search(some_string):
    some_string = some_string.replace(' ', '+')

    return some_string

# input = initial df
# output = pandas df with the following columns: relation label, ent1, ent2
def create_results_dict(init_df):

    results_df = init_df[['noun_1', 'noun_1_id', 'related_by', 'noun_2', 'noun_2_id']].copy()

    results_df['noun_1_wiki_page'] = results_df['noun_1'].apply(add_underscore)
    results_df['noun_2_wiki_page'] = results_df['noun_2'].apply(add_underscore)

    results_df['noun_1_google'] = results_df['noun_1'].apply(get_google_search)
    results_df['noun_2_google'] = results_df['noun_2'].apply(get_google_search)

    return results_df


# input = initial df, original text as a string, coref text as a string
# output = pandas dataframe with the following columns: coref_sent, original_sent, relation label, ent1, ent2
def create_annot_df(df, original_text, coref_text):

	nlpO = spacy.load("en_core_web_sm")
	docO = nlpO(original_text)
	originals = []
	for sent in docO.sents:
	    originals.append(sent.text)

	nlpC = spacy.load("en_core_web_sm")
	docC = nlpC(coref_text)
	cr = []
	for sent in docC.sents:
	    cr.append(sent.text)

	lookup = dict(zip(cr, originals))

	df['original_sent'] = df['coref_sent'].map(lookup)

	#annot_df = init_df[['coref_sent', 'original_sent', 'related_by', 'noun_1', 'noun_2']].copy()

	df['double'] = list(zip(df.related_by, df.noun_2))

	#annot_df = annot_df.drop(columns = ['related_by', 'noun_2'])

	return df

################# functions to get unique ID for each entity #####################################

def get_fuzzy_wuzzy_score(str_a, str_b):
    modified_a = re.sub(r'[^\w\s]', '', str_a)
    modified_b = re.sub(r'[^\w\s]', '', str_b)
    ratio = fuzz.ratio(str_a.lower(), str_b.lower())
    return ratio

def get_int_len(int_a):
    count = 0
    while int_a / 10 >= 1:
        int_a /= 10
        count += 1
    return count + 1

def assign_Q(has_id, no_id):
    for u_key in no_id:
        for l_key, l_val in has_id.items():
            str_a = u_key
            str_b = l_key
            ratio = get_fuzzy_wuzzy_score(str_a, str_b)

            if ratio >= 80:
                no_id[u_key] = has_id[l_key]

def assign_local(no_id):
    i = 1
    for k, v in no_id.items():
        i_len = get_int_len(i)
        if i_len < 4:
            append = '0'
            while i_len < 3:
                append += '0'
                i_len += 1
        if no_id[k] == 'id-less':
            no_id[k] =  'Loc' + append + str(i)
            i += 1

def create_id_df(df):
    ed1 = dict(zip(df.noun_1, df.noun_1_id))
    ed2 = dict(zip(df.noun_2, df.noun_2_id))
    ed1.update(ed2) 
    
    has_id = {}
    no_id = {}
    
    for k,v in ed1.items():
        if v != 'id-less':
            has_id[k] = v
        else:
            no_id[k] = v
        
    assign_Q(has_id, no_id)
    assign_local(no_id)
    
    has_id.update(no_id)
    
    df['noun_1_id'] = df['noun_1'].map(has_id)
    df['noun_2_id'] = df['noun_2'].map(has_id)
    
    return df


# function to convert a string that looks like a dictionary to an actual dictionary

def string_to_dict(this_row):
    # get rid of curly braces
    print("this_row is of type", type(this_row))

    this_row = this_row.replace('{', '')
    this_row = this_row.replace('}', '')
    
    # split by commas and turn into array where each elem is a dict pair
    this_row = this_row.split(',')
    
    my_dict = {}
    
    # split each elem in the array further by the colon, first elem is k, second is v
    for elem in this_row:
        elem = elem.split(':')
        my_dict[elem[0]] = elem[1]
    
    return my_dict

# input = df with IDs

def create_NER_df(df):
    
    list_1 = list(df['noun_1'].unique())
    list_2 = list(df['noun_2'].unique())
    all_ents = list_1 + list_2
    
    lower_cased = []
    
    for ent in all_ents:
        ent = ent.lower()
        lower_cased.append(ent)
    
    # add new columns to existing df
    df['noun1_NERtag'] = pd.Series(dtype='str')
    df['noun2_NERtag'] = pd.Series(dtype='str')
    
    for i in range(df.shape[0]):
    
        # get the ner entity tags for the entities in this relational triple and convert into a dict
        
        this_row = df['ner_tags'].iloc[i]
        #this_dict = string_to_dict(this_row)

        current_ent1 = df['noun_1'].iloc[i].lower()
        current_ent2 = df['noun_2'].iloc[i].lower()

        for ent in lower_cased:

            for key, value in this_row.items():

                ratio1 = get_fuzzy_wuzzy_score(key, current_ent1)

                if ratio1 >= 70:
                    df['noun1_NERtag'][i] = value

            for key, value in this_row.items():

                ratio2 = get_fuzzy_wuzzy_score(key, current_ent2)

                if ratio2 >= 70:
                    df['noun2_NERtag'][i] = value
                
    df = df.loc[:,['coref_sent', 'original_sent', 'noun_1', 'noun_1_id', 'noun1_NERtag', 'related_by',
              'noun_2', 'noun_2_id', 'noun2_NERtag', 'ner_tags']]
    
    return df


def create_stats_dict(df):

    # 1. add a new column to NER df of triples (ent 1, relation label, ent 2)
    df['triple'] = list(zip(df.noun_1, df.related_by, df.noun_2))

    # 2. Get list of ALL entities from the df , lower-cased : [all_ents_lc]
    # and Get list of all UNIQUE entities from the df, lower-cased: [all_unique]

    list_1 = list(df['noun_1'])
    list_2 = list(df['noun_2'])
    all_ents = list_1 + list_2
    all_ents_lc = []

    for ent in all_ents:
        ent = ent.lower()
        all_ents_lc.append(ent)

    set_a = set(all_ents_lc)
    all_unique = list(set_a)

    # 3. Do the following, store each in its own dict
    # - For each entity in all_unique, get a count of how many times it was mentioned
    # - For each entity in all_unique, get NER tags associated with it
    # - For each unique entity, get the ID
    # - For each unique entity, get the triples

    count_dict = {}
    ner_dict = {}
    id_dict = {}
    triples_dict = {}

    for ent in all_unique:
    
        my_count = 0
        for word in all_ents_lc:
            if word == ent:
                my_count += 1
        count_dict[ent] = my_count
        
        #print("current ent: ", ent)
        cond1 = df['noun_1'].str.lower() == ent
        cond2 = df['noun_2'].str.lower() == ent
        subdf = df[cond1 | cond2]
        #display(subdf)
        
        ner_candidates = []
        triples_list = []

        my_id = ''
        
        for i in range(subdf.shape[0]):
            if subdf['noun_1'].iloc[i].lower() == ent:
                #print("I am ent1")
                ner_candidates.append(subdf['noun1_NERtag'].iloc[i])
                my_id = (subdf['noun_1_id'].iloc[i])
                triples_list.append(subdf['triple'].iloc[i])
            elif subdf['noun_2'].iloc[i].lower() == ent:
                #print("I am ent2")
                ner_candidates.append(subdf['noun2_NERtag'].iloc[i])
                my_id = (subdf['noun_2_id'].iloc[i])
                triples_list.append(subdf['triple'].iloc[i])
                
        #print(ner_candidates)
        set_ner_candidates = set(ner_candidates)
        ner_dict[ent] = list(set_ner_candidates)
        
        id_dict[ent] = my_id
        
        triples_dict[ent] = triples_list   

    # 1) turn each of the dicts into a pd series
    # 2) combine them into a df
    # 3) convert the df into a dict of dicts where each unique entity is a key 
    #    with count, id, ner tags, relation label as each its own key with values

    counts = pd.Series(count_dict)
    ids = pd.Series(id_dict)
    ners = pd.Series(ner_dict)
    rels = pd.Series(triples_dict)

    data = {'counts': counts,
       'ids': ids,
       'ners': ners,
       'rels': rels}
  
    new = pd.concat(data, axis=1)
    total_dict = new.to_dict('index')

    return [total_dict, id_dict]

def get_map_data(results_list, init_df):

    # import dataset of cities and countries and process it
    # 1) filter only for relevant columns
    # 2) lower case relevant string columns (ascii city names and country names)

    data = pd.read_csv('worldcities.csv')
    data = data[['city_ascii', 'lat', 'lng', 'country', 'id']]
    data['city_ascii'] = data['city_ascii'].str.lower()
    data['country'] = data['country'].str.lower()

    # lists of unique countries and cities from the dataset
    data_countries = list(data['country'].unique())
    data_cities = list(data['city_ascii'].unique())


    # on the other hand, extract all possible geo locations from ALL NERs extracted from the text 
    # this is not the entities that are in relation triples. this is ALL NERs that spacy was able to find in the text
    # with the label GPE, NORP, or LOC

    geo_labels = ['GPE', 'NORP', 'LOC']

    all_geos = []

    for each_dict in results_list:
        for k, v in each_dict.items():
            if v in geo_labels:
                all_geos.append(k)

    uniques = set(all_geos)
    candidates = list(uniques)

    # to the list of candidate entities, add all entities in the entity columns in case spacy missed any
    ent1s = list(init_df['noun_1'].unique())
    ent2s = list(init_df['noun_2'].unique())
    all_ents = ent1s + ent2s

    for ent in all_ents:
        candidates.append(ent)

    all_res = {}

    countries = []
    cities = []

    us_names = ['the united states', 'the U.S.', 'u.s.']

    for c in candidates:
        if c.lower() in us_names:
            countries.append('united states')
        if c.lower() in data_countries:
            countries.append(c.lower())
        elif c.lower() in data_cities:
            cities.append(c.lower())

    all_res['countries'] = set(countries)
    all_res['cities'] = cities

    cities_wLL = {}

    for city in cities:
        cond = data['city_ascii'] == city
        my_slice = data[cond]

        # if there is more than one match for a city name (eg: Moscow, Russia and Moscow, IA, USA, get only the first match)
        if my_slice.shape[0] > 1:
            my_slice = my_slice.iloc[0]

        latitude = my_slice.lat.item()
        longitude = my_slice.lng.item()

    
        cities_wLL[city] = (latitude, longitude)

    all_res = [countries, cities_wLL]


    return all_res




