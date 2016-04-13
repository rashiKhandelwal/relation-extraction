#!/usr/bin/python


import nltk
from nltk.tree import *
from nltk.parse.stanford import StanfordParser
from nltk.parse.stanford import StanfordDependencyParser
from nltk.grammar import DependencyGrammar
from nltk.parse import DependencyGraph, ProjectiveDependencyParser, NonprojectiveDependencyParser
from Queue import *
import rdflib
from rdflib import *
from SPARQLWrapper import SPARQLWrapper, JSON, XML, ASK
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import dependency_treebank
import xml.dom
import json
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.corpus import brown
from rest_srvs.srv import *
import requests

import rospy
import os

import logging

logging.basicConfig()

tsub = [] # temporary list of subjects
sub = [] # final list of subjects
tpred = [] # temporary list of predicates
pred = [] # final list of subjects
tobj = [] # temporary list of objects
obj = [] # final list of objects
cand = [] # list of candidate properties
concept = [] # list of subjects and objects that have been found in the ontology directly or indirectly





#------------------------------------------------- ROS Services -----------------------------------------------

insert_req = InsertIndividualRequest()
insert_req.serverUrl = "http://www.kbai.sys" 
insert_req.repoName = "kb"
insert_req.contextUri = "http://www.kbai.sys/rest/kb/hospital.owl"


try:
	s_insert = rospy.ServiceProxy('/rest_server/insert_individual', InsertIndividual)
except rospy.ServiceException, e:
       		print "Service call failed: %s"%e


select_req = QueryResultsRequest()
select_req.serverUrl = "http://www.kbai.sys" 
select_req.repoName = "kb"

try:
	s_query_result = rospy.ServiceProxy('/rest_server/queryResult', QueryResults)
except rospy.ServiceException, e:
       		print "Service call failed: %s"%e


add_req = AddRDFDataRequest()
add_req.serverUrl = "http://www.kbai.sys" 
add_req.repoName = "kb"
add_req.contextUri = "http://www.kbai.sys/rest/kb/hospital.owl"

try:
	s_add = rospy.ServiceProxy('/rest_server/add_rdf_data', AddRDFData)
except rospy.ServiceException, e:
       		print "Service call failed: %s"%e




#dfs -------------------------------------------------------------------------------

def find_tags(t):
    try:
        t.label()
    except AttributeError:
        return
    if t.label() == "NN" or t.label() == "NNP" or t.label() == "NNPS" or t.label() =="NNS" or t.label() =="PRP" or t.label() =="PRP$":
	if not tsub:
		for child in t:
			tsub.append(child)
			concept.append(child)
		#tobj.append(child)
	else:
		for child in t:
			tobj.append(child)
			concept.append(child)	
    elif t.label() == "VBG" or t.label()== "VBD" or t.label()== "VBN" or t.label()== "VBP" or t.label()== "VB":
	for child in t:
		a = WordNetLemmatizer().lemmatize(child,'v')   # v is for verbs   	
		tpred.append(a)
    elif t.label() == "JJ":
	for child in t:
		tpred.append(child)
    for child in t:
         find_tags(child)


def find_VBZ(t):
	
	try:
		t.label()
	except AttributeError:
		return
	
	global first,flag,tag
	
	# if VBZ in between 2 nouns that are concepts in ontology then its the predicate
	if t.label() == "NN" or t.label() == "NNP":
		if flag == 0:
			for child in t:
				first = child  # to store the noun occuring before VBZ
				#print first
		if flag == 1:
			if (first in sub) or (first in concept):
				for child in t:
					if (child in sub) or (child in concept) :		
					# to check if noun occuring just before and after VBZ is in ontology 
						tpred.append(tag_VBZ)
					
	if t.label() == "VBZ":
		flag = 1
		for child in t:
			tag = child
		
	
	for child in t:
        	find_VBZ(child)
		
def find_IN(t):
	
	try:
		t.label()
	except AttributeError:
		return

	# if IN is right before a noun that is a concept in ontology then its the predicate	
	
	global flag,tag
	
	if t.label() == "NN" or t.label() == "NNP":
		if flag == 1:
			for child in t:
				if (child in obj) or (child in concept): 
# to check if noun occuring just before and after VBZ is in ontology 
					tpred.append(tag)
			
	if t.label() == "IN":
		flag = 1
		for child in t:
			tag = child
		
			
	for child in t:
        	find_IN(child)

				
# ------------------------- parse tree --------------------------

def parse(inp):
	english_parser = StanfordParser('/home/rashi/stanford-parser-full-2014-08-27/stanford-parser.jar', '/home/rashi/stanford-parser-full-2014-08-27/stanford-parser-3.4.1-models.jar')
	a=english_parser.raw_parse(inp)
	'''dep_parser = StanfordDependencyParser('/home/rashi/stanford-parser-full-2014-08-27/stanford-parser.jar', '/home/rashi/stanford-parser-full-2014-08-27/stanford-parser-3.4.1-models.jar')
	dp = dep_parser(inp)
	print dp'''
	return a




def insert_individual(individual,parent):
	insert_req.classUri = "http://www.kbai.sys/rest/kb/kb.owl#"+parent
        insert_req.individualUri = "http://www.kbai.sys/rest/kb/hospital.owl#"+individual
	s_insert(insert_req) 
    	

def assert_property(inst,prop,val):
	add_req.instanceUri = inst #"http://www.kbai.sys/rest/kb/hospital.owl#"+i
        add_req.propertyUri = prop
	add_req.value = val	
	s_add(add_req) 
	
# --------------------------  named entity recognition ----------------------------------

def named_entity_recog(inp):
	chunked =nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(inp)))
	chunked.draw()
        continuous_chunk = []
        for i in chunked:
        	if type(i) == Tree:
			if ( i.label() == "PERSON"):
				insert_individual(i.leaves()[0][0],"Human")		#hardcoding the parent type for PERSON to be Human						
				continuous_chunk.append(i.leaves()[0][0])
			elif( i.label() == "LOCATION"):
				insert_individual(i.leaves()[0][0],"Place")		#hardcoding the parent type for LOCATION to be Place
				continuous_chunk.append(i.leaves()[0][0])
			elif( i.label() == "ORGANIZATION"):	
				insert_individual(i.leaves()[0][0],"Place")		#hardcoding the parent type for ORGANIZATION to be Place
				continuous_chunk.append(i.leaves()[0][0])
			elif( i.label() == "GPE"):					
				insert_individual(i.leaves()[0][0],"Place")		#hardcoding the parent type for GPE to be Place
				continuous_chunk.append(i.leaves()[0][0])
        return continuous_chunk
	'''tokenized = nltk.word_tokenize(inp)
	tagged = nltk.pos_tag(tokenized)
	print tagged
	namedEnt = nltk.ne_chunk(tagged)
	for i in namedEnt:
		for x,y in i:
			print y
	return namedEnt
	#namedEnt.draw()''' 



# -----------------check in wordnet ------------------------------------
def check_wordnet(word,tag):
	temp = []
	for s in wn.synsets(word):		
		if s.hypernyms():					
			for p in s.hypernym_paths():
				count = 0
				for rev in reversed(p):	
					count = count + 1	
					#print k.lemma_names()[0]
					if rev.lemma_names()[0] in temp:
						break
					temp.append(rev.lemma_names()[0])
							
					
					if check_onto_concept(rev.lemma_names()[0],tag):	
						return 
						
					if (count > 4):  # threshold to not travel uptil root of graph for each word
						break
							


# --------------------- using SPARQLWrapper ------------------------------

def check_onto_concept(word,tag):
	select_req.query = "SELECT DISTINCT ?x WHERE { ?x a owl:Class. }"
	
	row = s_query_result(select_req)
	res = row.resultJSON
	if res == None:
		return 
	res = (res[8:]).split('"')
	flag = 0
	for r in res:
		if word.lower() in r.lower():
			if tag == "subject":
				if r not in sub:
					sub.append(r)
					concept.append(r)
					#print r
					return
    			if tag == "object":
				if r not in sub:
					obj.append(r)
					concept.append(r)
					return
			if tag == "predicate":
				if r not in sub:
					pred.append(r)
					return

def check_onto_prop(word):
	select_req.query = "SELECT DISTINCT ?x WHERE { ?x a owl:ObjectProperty. }"
	row = s_query_result(select_req)
	res = row.resultJSON
	if res == None:
		return False
	#print res
	if word.lower() in res.lower():
		if word not in pred:
			return True	
    	

def check_onto_individual(word):
	select_req.query = "SELECT DISTINCT ?x WHERE { ?x a owl:NamedIndividual. }"

	row = s_query_result(select_req)
	res = row.resultJSON
	#print res
	if res == None:
		return False
	if word.lower() in res.lower():
		if word not in pred:
			return True	
   


def candidate_list(i,j):
	select_req.query = """select distinct ?x where {
  ?x rdfs:domain ?domain ;
            rdfs:range ?range .
{
?y (^rdfs:domain/rdfs:range)* ?domain . <http://www.kbai.sys/rest/kb/hospital.owl#"""+i+"""> a ?y.

  ?range (^rdfs:domain/rdfs:range)* ?z. <http://www.kbai.sys/rest/kb/hospital.owl#"""+j+"""> a ?z. } 

UNION  

{
?y (^rdfs:domain/rdfs:range)* ?domain . <http://www.kbai.sys/rest/kb/hospital.owl#"""+i+"""> a ?y.

  ?range (^rdfs:domain/rdfs:range)* <http://www.kbai.sys/rest/kb/kb.owl#"""+j+"""> . } 

UNION

{
<http://www.kbai.sys/rest/kb/kb.owl#"""+i+"""> (^rdfs:domain/rdfs:range)* ?domain .

  ?range (^rdfs:domain/rdfs:range)* ?z. <http://www.kbai.sys/rest/kb/hospital.owl#"""+j+"""> a ?z. }

UNION

{
<http://www.kbai.sys/rest/kb/kb.owl#"""+i+"""> (^rdfs:domain/rdfs:range)* ?domain .

  ?range (^rdfs:domain/rdfs:range)*  <http://www.kbai.sys/rest/kb/kb.owl#"""+j+"""> . } }
"""
    
	row = s_query_result(select_req)
	
	res = row.resultJSON
	
	if res == None:
		return None
	
	return res
	




#------------------- extract relation -----------------------------------------
def get_rel():
	for s in sub:
		for o in obj:
			qres = candidate_list(s,o)
			if not qres:
				return
			qres = (qres[8:]).split('"')
			flag = 0
			for q in qres:
				if flag == 0:
					cand.append(q)
					flag = 1
				else:
					flag = 0
			#if not qres:
			#	return
			#for row in qres['x']:
			#	print row				
			#	cand.append(row[35:]) #obtaining just the name of the concept from the uri		
			for word in tpred:
				for c in cand:
					if word.lower() in c.lower():
							pred.append(c)
							return 
								
					
					
#------------------- extract similar relations -----------------------------------------	
def get_similar_rel():
	for c in cand:
		for word in tpred:
			for s1 in wn.synsets(word):
				for s2 in wn.synsets(c[35:]):
					#print s1,s2,wn.wup_similarity(s1,s2)
			#-------- threshold set for similarity matching ----------
					if(wn.wup_similarity(s1,s2) >= 0.85): 
						pred.append(c)
						return
				
			
# ---------------------------  MAIN   -------------------------------------

#print brown.tagged_sents()

inp = "John is seeing Anna"  # input 

tree = parse(inp)

for i in tree:
	for j in i:
		find_tags(j)
		print(j)
		
#print tsub, tpred, tobj

n = named_entity_recog(inp)

#print n

#get entities in subject list that are classes in ontology
for i in tsub:    
	check_onto_concept(i,"subject")
		#sub.append(i)
	if i in n:
		sub.append(i)
		concept.append(i)


#get entities in object list that are classes in ontology
for i in tobj:
	check_onto_concept(i,"object")
		#obj.append(i)
	if i in n:
		obj.append(i)
		concept.append(i)


#get entities similar to subject list from wordnet and match with ontology
if not sub:  
	for i in tsub:
		#print i
		check_wordnet(i,"subject")
		if sub:
			break


#get entities similar to object list from wordnet and match with ontology
if not obj:  
	for i in tobj:
		#print i
		check_wordnet(i,"object")
		if obj:
			break

#find list of possible predicates

tree = parse(inp)

if not tpred:
	for i in tree:
		for j in i:
			flag = 0
			find_IN(j)
			if not tpred:
				flag = 0
				find_VBZ(j)
			

#print sub,tpred, obj

#get relation after matching with ontology
if sub:
	if obj:
		get_rel()
		if not pred:
			get_similar_rel()
		if not pred:
			pred.append("Unknown Relation")
		else:
			for s in sub:
				for o in obj:
					for p in pred:     
						# asserting property in ontology , assuming subject and object are individuals in ontology 
						assert_property("http://www.kbai.sys/rest/kb/hospital.owl#"+s,p,"http://www.kbai.sys/rest/kb/hospital.owl#"+o)
		

print sub,pred,obj






