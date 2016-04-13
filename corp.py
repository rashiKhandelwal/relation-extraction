#!/usr/bin/python


import nltk
from nltk.tree import *
from nltk.parse.stanford import StanfordParser
from nltk.grammar import DependencyGrammar
from nltk.parse import DependencyGraph, ProjectiveDependencyParser, NonprojectiveDependencyParser
from Queue import *
import rdflib
from rdflib import *
from SPARQLWrapper import SPARQLWrapper, JSON, XML, ASK
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import dependency_treebank
from pattern.en import *
import xml.dom
import json
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.corpus import names
from rest_srvs.srv import *
import requests
import operator
import rospy
import os
import random
import time


import logging
from sklearn import svm
from svmutil import *
import warnings

logging.basicConfig()


warnings.simplefilter("ignore",RuntimeWarning)
warnings.simplefilter("ignore",SyntaxWarning)

t_concept = {}	# lookup table for ontology concepts
t_property = {}	# lookup table for ontology properties
wn_noun = {}
wn_verb = {}
wn_prep = {}
pr = {}


temp = []

person = []	# to store all names of persons in sentence
loc = []	# to store all names of locations in sentence
org = []	# to store all names of organisations in sentence
gpe = []	# to store all names of gpe in sentence


train_x = []
train_y = []
train_set = []
train_ref = []
train_x_ref = []
train_y_ref = []


dist = 0	# to find distance between two entities in the tree
ftr = []	# to store feature matrix
ref = []
pos = 0		# to get position of noun in the feature matrix
flag = 0	# flag to check if reached another noun
dist_btwn = 0
sub_found = 0
obj_found = 0

entity1 = None
entity2 = None


total = 0
FP = 0		# false positive from classifier
TP = 0		# true positive from classifier
FN = 0		# false negative from classifier
TN = 0		# true negative from classifier

totalR = 0
FPR = 0		# false positive from classifier
TPR = 0		# true positive from classifier
FNR = 0		# false negative from classifier
TNR = 0		# true negative from classifier

#------------------------------------------------- ROS Services -----------------------------------------------

insert_req = InsertIndividualRequest()				# Insert query
insert_req.serverUrl = "http://www.kbai.sys" 
insert_req.repoName = "kb"
insert_req.contextUri = "http://www.kbai.sys/rest/kb/hospital.owl"


try:
	s_insert = rospy.ServiceProxy('/rest_server/insert_individual', InsertIndividual)
except rospy.ServiceException, e:
       		print "Service call failed: %s"%e


select_req = QueryResultsRequest()				# Select query		
select_req.serverUrl = "http://www.kbai.sys" 
select_req.repoName = "kb"

try:
	s_query_result = rospy.ServiceProxy('/rest_server/queryResult', QueryResults)
except rospy.ServiceException, e:
       		print "Service call failed: %s"%e


add_req = AddRDFDataRequest()					# Add request
add_req.serverUrl = "http://www.kbai.sys" 
add_req.repoName = "kb"
add_req.contextUri = "http://www.kbai.sys/rest/kb/hospital.owl"

try:
	s_add = rospy.ServiceProxy('/rest_server/add_rdf_data', AddRDFData)
except rospy.ServiceException, e:
       		print "Service call failed: %s"%e




#--------------------------- dfs traversal  --------------------------------------------

def main_traverse(t):						# dfs to get the nouns

	try:
        	t.label()
    	except AttributeError:
        	return
	
	#print "in main traverse",t
	global dist,pos,totalR
	dist = dist + 1
	for child in t:
        	if t.label() == "NN" or t.label() == "NNP" or t.label() == "NNPS" or t.label() =="NNS":  	# to get noun
			#if t.leaves()[-1].lower() is not in entities:
			#	entities.append(t.leaves()[-1].lower())
			features(t,'N',t.leaves()[-1])
			pos = pos+1
		elif t.label() =="PRP" or t.label() =="PRP$":
			new = resolve(t)
			totalR = totalR + 1
			if new!= None:
				features(t,'P',new)
			else:
				features(t,'P',t.leaves()[-1])
			pos = pos+1
				
	for child in t:
     		main_traverse(child)

def classify_pronoun(a):
	if a == 'her' or a == 'she' or a == 'hers':
		return "female"
	else:
		return "male"	


def gender_features(name):
    feat = {}
    feat["first_letter"] = name[0].lower()
    feat["last_letter"] = name[-1].lower()
    for letter in 'abcdefghijklmnopqrstuvwxyz':
        feat["count({})".format(letter)] = name.lower().count(letter)
        feat["has({})".format(letter)] = (letter in name.lower())
    return feat


def gender_match(a,b):
	if classify_pronoun(a) == classifier.classify(gender_features(b)):
		return 1
	else:
		return 0


def number_match(a,b):
	if singularize(a) == a and singularize(b) == b:
		return 1
	elif pluralize(a) == a and pluralize(b) == b:
		return 1
	else:
		return 0


def semantic_type_match(a,b):
	if semantic_type(a) == semantic_type(b):
		return 1
	else:
		return 0


def resolve(t):						## --------------------------- currently checking with synsets, check with hypernyms next for similarity measures
	word = t.leaves()[-1] 
	global ref,TPR,TNR,FPR,FNR,totalR
	print ref, len(ref)
	#ref_classifier = nltk.NaiveBayesClassifier.train(train_ref)
	m = train_classifier_ref()
	for i in reversed(ref):
		#print i
		lst = []
		lst.append(int(get_pr(word.lower())))
		lst.append(int(semantic_type_match(word,i)))
		lst.append(int(gender_match(word,i)))
		lst.append(int(number_match(word,i)))
		print "lst: ",lst
		pred = test_classifier_ref(m,lst)
		print "Result: ",pred
		#print [word.lower(),semantic_type_match(word,i[0]),gender_match(word,i[0]),number_match(word,i[0])]
		if pred[0] == 1.0:
			#totalR = totalR + 1
			print word," resolved to ",i," if correct press y/Y: "
			response = raw_input()
			if response == 'y' or response == 'Y':	
				TPR = TPR + 1		
				return i
			else:
				train_x_ref.append(lst)	
				train_y_ref.append(0)	
				FPR = FPR + 1			
				'''response = raw_input("Give correct resolution")
				lst = []
				lst.append(int(get_pr(word.lower())))
				lst.append(int(semantic_type_match(word,response)))
				lst.append(int(gender_match(word,response)))
				lst.append(int(number_match(word,response)))
				train_x_ref.append(lst)
				print "training data : ",[get_pr(word.lower()),semantic_type_match(word,response),gender_match(word,response),number_match(word,response)]
				train_y_ref.append(1)
				#print>>train_out, [word,semantic_type_match(word,response),gender_match(word,response),number_match(word,response)],1
				return response'''

	print "Give correct resolution for ",word,": "
	TNR = TNR + 1	
	response = raw_input()
	lst = []
	lst.append(int(get_pr(word.lower())))
	lst.append(int(semantic_type_match(word,response)))
	lst.append(int(gender_match(word,response)))
	lst.append(int(number_match(word,response)))
	train_x_ref.append(lst)
	print "training data : ",[get_pr(word.lower()),semantic_type_match(word,response),gender_match(word,response),number_match(word,response)]
	train_y_ref.append(1)
	#print>>train_out, [word,semantic_type_match(word,response),gender_match(word,response),number_match(word,response)],1
	return response

			
		

'''def resolve(t):						## --------------------------- currently checking with synsets, check with hypernyms next for similarity measures
	word = t.leaves()[-1] 
	global ref
	print ref, len(ref)
	for i in reversed(ref):
		print i
		for s1 in wn.synsets(i[0]):
				for s2 in wn.synsets(word):
					#print s1,s2,wn.wup_similarity(s1,s2)
			#-------- threshold set for similarity matching ----------
					sim = wn.wup_similarity(s1,s2)
					print s1,i[0],s2,word,sim
					if(sim >= 0.3):	
						#t.leaves()[-1] = i[0]
						return i[0]

					
					if s1.hypernyms():					
						for p in s1.hypernym_paths():
							count1 = 0
							for rev in reversed(p):	
								count1 = count1 + 1
								sim = wn.wup_similarity(rev,s2)	
								print rev,i[0],s2,word,sim
								if(sim >= 0.3):
									#t.leaves()[-1] = 
									return i[0]
								if s2.hypernyms():					
									for p2 in s2.hypernym_paths():
										count2 = 0
										for rev2 in reversed(p2):
											count2 = count2 + 1
											sim = wn.wup_similarity(rev,rev2)
											print rev,i[0],rev2,word,sim
											if(sim >= 0.3):	
												#t.leaves()[-1] = i[0]
												return i[0]
											if count2 > 3:
												break	
										if count2 > 3:
											break
								if count1 > 3:
									break
							if count1 > 3:
								break'''
			



def get_vector(sub,obj,t,rel):
	
	try:
        	t.label()
    	except AttributeError:
        	return
	
	global dist_btwn,sub_found,obj_found,entity1,entity2
	dist_btwn = dist_btwn + 1
	for child in t:
		if child == sub:
			entity1 = t.leaves()[-1]
			dist_btwn = 0 
			sub_found = 1
        	if child == obj:
			entity2 = t.leaves()[-1]
			obj_found = 1
		if sub_found == 1 and obj_found == 1:
			train_x.append([match_concept_wn(entity1),semantic_type(entity1),0,dist_btwn,match_concept_wn(entity2),semantic_type(entity2),rel,0])
			train_y.append(rel)
			sub_found = 0
			obj_found = 0
			return		
	for child in t:
     		get_vector(sub,obj,child,rel)

# -------------------------- creating feature matrix , each row is a feature set for classifier ------------------------

def features(t,ch,new):
	global ftr,ref
	#lst = []
	f1 = f2 = f3 = f4 = 0	#features
	e1 = None	# entity 1
	#e2 = None	# entity 2	

	if ch == 'P':
		
		print "Replacing ", t.leaves()[-1], new
		e1 = new
	else:
		e1 = t.leaves()[-1] 				# entity (noun)
	#print "e1",e1
	#lst.append(e1)

	f2 = semantic_type(e1)

	if f2 == 1:
		f1 = match_concept_wn("person")
	elif f2 == 2:
		f1 = match_concept_wn("location")
	elif f2 == 3:
		f1 = match_concept_wn("organization")
	elif f2 == 4:
		f1 = match_concept_wn("place")
	else:
		f1 = match_concept_wn(e1)	
#	f1 = match_concept(e1)				# match concepts with ontology
#	if f1 == 0:
#		f1 = check_wordnet(e1)
	#lst.append(f1)
	#print "e1",f1
	
	#lst.append(f2)
	
	#-------------------- to check words before noun ----------------------------------------

	cnt = 0						# counter to get only 3 words before e1
	
    	current = t
	before = 0 					# to see if verb/pp is found before e1
	
	while (current is not None) and (current.parent() is not None) :
			if cnt > 8:			# threshold while traversing edges in tree to get words before noun
				break
			#print "curr1->",current
			cnt = cnt + 1
    			while current is not None and current.left_sibling() is not None:
				current = current.left_sibling()
				#print "curr2->",current	
				if cnt > 8:		
			
					break

				else:
					cnt = cnt+1
					if before == 0:  # flag to see if verb is already found in the sentence 
				
						if current.label() == "VBG" or current.label() == "VBD" or current.label() == "VBN" or current.label() == "VBP" or current.label() == "VB":
							for child in current:
								f3 = match_property_wn(verb_stem(child))
								#f3 = match_property(verb_stem(child))
								#if f3 == 0:
								#	f3 = wordnet_property(verb_stem(child))
								#print "f3",f3,child
								before = 1
						elif current.label() == "IN" or current.label() == "JJ" or current.label() == "JJR" or current.label() == "JJS" or current.label() == "RB" or current.label() == "RBR" or current.label() == "RBS": 
							#elif temp[1] == "IN": 
							for child in current:
								f3 = match_property_wn(child)
								#if f3 == 0:
								#	f3 = wordnet_property(child)
								before = 1
						elif current.label() == "NN" or current.label() == "NNP" or current.label() == "NNPS" or current.label() =="NNS" or current.label() =="PRP" or current.label() =="PRP$":	
							#print current
							cnt = 9		# counter to stop traversal if previous noun is encountered 
							break	
					else:
						break
			#print "curr->",current
			
			if current.parent() is not None:
				current = current.parent()
			else:
				break 
	#lst.append(f3)
	#print "e1",f3
	f4 = dist
	#lst.append(f4)
	#print "e1",f4
	current = t
	#lst.append(0)

	f5 = []
	while current is not None:
		if current.right_sibling() is not None:
			current = current.right_sibling()
			#print "hi1",current
			after_traverse(current,f5)	# to get verbs/prepositions after noun
			#print "f5",f5
			break
		else:
			if current.parent() is not None:
				current = current.parent()
			else:
				break

	#print t,f5
	#print "e1",f5
	

	if len(f5) == 0:				# if no verb/preposition after noun
		lst = []
		lst.append(e1)
		lst.append(f1)
		lst.append(f2)
		lst.append(f3)
		lst.append(f4)
		lst.append(0)
		#lst.append(f5[i])
		#print lst
		#print "here",lst
		ftr.append(lst)
		if e1.lower() not in ref:
			ref.append(e1.lower())
	else:
		for i in range(0,len(f5)):		# if more than one verb/preposition after noun
			lst = []
			lst.append(e1)
			lst.append(f1)
			lst.append(f2)
			lst.append(f3)
			lst.append(f4)
			lst.append(f5[i])
			#lst.append(f5[i])
			#print i,lst
			#print lst
			ftr.append(lst)
			if e1.lower() not in ref:
				ref.append(e1.lower())
			#lst.pop()
	#after_traverse(current,lst)
	#if lst not in ftr:
	#	ftr.append(lst)

	#print ftr

def after_traverse(t,f5):				# to get verbs/prepositions after noun

	try:
        	t.label()
    	except AttributeError:
        	return
	
	#print "in main traverse",t
	global flag,pos
	
	for child in t:
		if flag == 1:
			
			return
		elif t.label() == "VBG" or t.label() == "VBD" or t.label() == "VBN" or t.label() == "VBP" or t.label() == "VB":
				for child in t:
					f5.append(match_property_wn(verb_stem(child)))
					#if lst not in ftr:
					#ftr.append(lst)
					#print "f5",ftr
					#print "here2",child
					#print "appending",t	
		elif t.label() == "JJ" or t.label() == "JJR" or t.label() == "JJS" or t.label() == "RB" or t.label() == "RBR" or t.label() == "RBS" or t.label() == "IN" :	
				for child in t:
					#lst[len(lst)-1] = match_property(child)
					f5.append(match_property_wn(child))
					#if lst not in ftr:
					#ftr.append(lst)	
					#print "f5 in ",match_property(child),child,lst				
					#lst.append(match_property(child))
					#print "here3",child
					#print "appending",t				
				#flag = 1
				#return	
        	elif t.label() == "NN" or t.label() == "NNP" or t.label() == "NNPS" or t.label() =="NNS" or t.label() =="PRP" or t.label() =="PRP$":
			flag = 1 			# flag to check if another noun is found then return
			return
			#print "in ", t
			#done = 0
			#found_e1 = 0
			#dist = 0
			#traverse(t) # to extract features of first noun
		
					
			
				
	for child in t:
		if flag == 1:
			#if len(lst) < 6:	
			#	lst.append(0)
			return
     		after_traverse(child,f5)


def match_concept_wn(word):

	for k,v in wn_noun.iteritems():
		for s in wn.synsets(word.lower()):
			if s.lemmas()[0].key() == k:
				return v
	return 0

def match_property_wn(word):

	for k,v in wn_verb.iteritems():
		for s in wn.synsets(word.lower()):
			if s.lemmas()[0].key() == k:
				return v
	return 0


def get_property(inp):
	for k,v in wn_verb.iteritems():
		if inp == v:
			return wn.lemma_from_key(k)
	return "Unknown"


def get_pr(inp):
	for k,v in pr.iteritems():
		if inp == k:
			return v
	return 0


# ------------ return value corresponding to matched concept in ontology----------------

def match_concept(word):
	
	for k,v in t_concept.iteritems():
		if word.lower() in k.lower():
			return v
	return 0


# ----------- return value corresponding to matched property in ontology -----------------

def match_property(word):

	for k,v in t_property.iteritems():
		if word.lower() in k.lower():
			return v
	return 0

# --------------- create lists of named entities present in the sentence ----------------------

def semantic_lists(inp):	 	
	
	chunked = nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(inp)))
        for word in chunked:
        	if type(word) == Tree:
			if word.label() == "PERSON":
				person.append(word.leaves()[0][0])
				insert_individual(word.leaves()[0][0],"Human")
				
			elif word.label() == "LOCATION":
				loc.append(word.leaves()[0][0])
				insert_individual(word.leaves()[0][0],"Place")	
				
			elif word.label() == "ORGANIZATION":
				org.append(word.leaves()[0][0])	
				insert_individual(word.leaves()[0][0],"Place")			
				
			elif word.label() == "GPE":
				gpe.append(word.leaves()[0][0])
				insert_individual(word.leaves()[0][0],"Place")	
				
			


# --------------- return value corresponding to named type of concept----------------------

def semantic_type(word):		
	
	if word in person:
		return 1
	elif word in loc:
		return 2
	elif word in org:
		return 3
	elif word in gpe:
		return 4
	else:
		return 0


# ------------- returns the stem verb for a verb  -------------------------------------

def verb_stem(word):
	return WordNetLemmatizer().lemmatize(str(word),'v')

					
# ------------------------- parse tree --------------------------

def parse(inp):
	english_parser = StanfordParser('/home/rashi/stanford-parser-full-2014-08-27/stanford-parser.jar', '/home/rashi/stanford-parser-full-2014-08-27/stanford-parser-3.4.1-models.jar')
	a=english_parser.raw_parse(inp)
	return a

# ------------------------ insert individual instance in ontology --------------------

def insert_individual(individual,parent):
	insert_req.classUri = "http://www.kbai.sys/rest/kb/kb.owl#"+parent
        insert_req.individualUri = "http://www.kbai.sys/rest/kb/hospital.owl#"+individual
	s_insert(insert_req) 
    	

# ------------------------ assert property in ontology -----------------------------

def assert_property(inst,prop,val):
	add_req.instanceUri = inst #"http://www.kbai.sys/rest/kb/hospital.owl#"+i
        add_req.propertyUri = prop
	add_req.value = val	
	s_add(add_req) 

# -----------------check in wordnet ------------------------------------ 

def check_wordnet(word):
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
					res = match_concept(rev.lemma_names()[0])
					#print "--------------------------",rev.lemma_names()[0], rev.lemma().key()  ## rev.lemma().key()  or try rev.key() ----> should give unique key number
					if res == 0:
						continue	
					else:
						return res 
					if (count > 4):  		# threshold to not travel uptil root of graph for each word
						break
	return 0						

def wordnet_property(word):
	temp = []
	for s in wn.synsets(word):		
		if s.hypernyms():					
			for p in s.hypernym_paths():
				count = 0
				for rev in reversed(p):	
					count = count + 1
					print 
					#print k.lemma_names()[0]
					if rev.lemma_names()[0] in temp:
						break
					temp.append(rev.lemma_names()[0])
					res = match_property(rev.lemma_names()[0]) 
					#print "--------------------------",rev.lemma_names()[0], rev.lemmas()[0].key()
					if res == 0:
						continue	
					else:
						return res 
					if (count > 4):  		# threshold to not travel uptil root of graph for each word
						break
	return 0					


def get_similar_rel(word):
	for s1 in wn.synsets(word):
		for s2 in wn.synsets(c[35:]):
					#print s1,s2,wn.wup_similarity(s1,s2)
			#-------- threshold set for similarity matching ----------
					if(wn.wup_similarity(s1,s2) >= 0.85): 
						pred.append(c)
						return

#------------------- lookup tables ---------------------------------------------------

def concept_table():
	t_concept["Unknown"] = 0
	select_req.query = "SELECT DISTINCT ?x WHERE { ?x a owl:Class. }"
	row = s_query_result(select_req)
	res = row.resultJSON
	if res == None:
		return 
	res = (res[8:]).split('"')
	ctr = 1
	for r in res:
		split = r.split('#')
		if len(split) > 1 :
			t_concept[split[1]] = ctr
			ctr = ctr + 1

	select_req.query = "SELECT DISTINCT ?x WHERE { ?x a owl:NamedIndividual. ?x a ?y. ?y a owl:Class. }"
	row = s_query_result(select_req)
	res = row.resultJSON
	if res == None:
		return 
	res = (res[8:]).split('"')
	for r in res:
		split = r.split('#')
		if len(split) > 1 :
			t_concept[split[1]] = ctr
			ctr = ctr + 1

def property_table():
	t_property["Unknown"] = 0
	select_req.query = "SELECT DISTINCT ?x WHERE { ?x a owl:ObjectProperty. }"
	row = s_query_result(select_req)
	res = row.resultJSON
	if res == None:
		return 
	res = (res[8:]).split('"')
	ctr = 1
	for r in res:
		split = r.split('#')
		if len(split) > 1 :
			t_property[split[1]] = ctr
			ctr = ctr + 1



def wn_noun_table():
	wn_noun["Unknown"] = 0
	ctr = 1
	for synset in list(wn.all_synsets('n')):
		for s in synset.lemmas():
			#wn_noun[synset.lemmas()[0].key()] = ctr
			wn_noun[s.key()] = ctr
			ctr = ctr+1
		ctr = ctr+1
	#print wn_noun

def wn_verb_table():
	wn_verb["Unknown"] = 0
	ctr = 1
	for synset in list(wn.all_synsets('v')):
		for s in synset.lemmas():
			#wn_noun[synset.lemmas()[0].key()] = ctr
			wn_verb[s.key()] = ctr
			ctr = ctr+1
		#wn_verb[synset.lemmas()[0].key()] = ctr
		ctr = ctr+1
	for synset in list(wn.all_synsets('a')):			# adjective
		for s in synset.lemmas():
			#wn_noun[synset.lemmas()[0].key()] = ctr
			wn_verb[s.key()] = ctr
			ctr = ctr+1
		#wn_verb[synset.lemmas()[0].key()] = ctr
		ctr = ctr+1
	for synset in list(wn.all_synsets('r')):			# adverb
		for s in synset.lemmas():
			#wn_noun[synset.lemmas()[0].key()] = ctr
			wn_verb[s.key()] = ctr
			ctr = ctr+1
		#wn_verb[synset.lemmas()[0].key()] = ctr
		ctr = ctr+1
	#print wn_verb


def pron_table():
	
	pr["Unknown"] = 0
	pr["his"] = 1
	pr["he"] = 2
	pr["him"] = 3
	pr["she"] = 4
	pr["her"] = 5
	pr["hers"] = 6
	pr["it"] = 7
	pr["they"] = 8
	pr["their"] = 9
	pr["them"] = 10
	pr["theirs"] = 11
	pr["I"] = 12
	pr["me"] = 13
	pr["mine"] = 14
	pr["you"] = 15
	pr["your"] = 16
	pr["yours"] = 17
	pr["my"] = 18
	pr["myself"] = 19
	pr["this"] = 20
	pr["that"] = 21
	pr["these"] = 22
	pr["those"] = 23

#----------------- classifier  ------------------------------------------


def train_classifier():
	

	prob = svm_problem(train_y, train_x)  		# svm_problem(the classes for the corresponding data, data with the features)

	param = svm_parameter('-q')
	param.kernel_type = RBF
	param.C = 10
	
	m = svm_train(prob, param)
	return m


def test_classifier(m,inp):
	test_x = [inp]
	#print>>f_out, a = svm_predict([0]*len(test_x), test_x, m, -q)
	#a = svm_predict([0]*len(test_x), test_x, m)
	a,b,c = svm_predict([0]*len(test_x), test_x, m)
	'''a -> predicted class 
	b -> a tuple including  accuracy (for classification), mean-squared 
        error, and squared correlation coefficient (for regression)
	c -> a list of decision values or probability estimates (if '-b 1' 
        is specified). If k is the number of classes, for decision values,
        each element includes results of predicting k(k-1)/2 binary-class
        SVMs. For probabilities, each element contains k values indicating
        the probability that the testing instance is in each class.
        Note that the order of classes here is the same as 'model.label'
        field in the model structure.'''
	#return a
	
	
	for k,v in wn_verb.iteritems():
		if v == a[0]:
			return k
	#print "Relation: ",t_property[a]
	#return t_property[a]
	return "Unknown"	
	#m.predict([1,1,1]) #model predicts the class


def train_classifier_ref():
	
	print train_y_ref , train_x_ref
	prob = svm_problem(train_y_ref, train_x_ref)  		# svm_problem(the classes for the corresponding data, data with the features)

	param = svm_parameter('-q')
	param.kernel_type = RBF
	param.C = 10
	
	m = svm_train(prob, param)
	return m


def test_classifier_ref(m,inp):
	test_x = [inp]
	#print>>f_out, a = svm_predict([0]*len(test_x), test_x, m, -q)
	#a = svm_predict([0]*len(test_x), test_x, m)
	a,b,c = svm_predict([0]*len(test_x), test_x, m)
	'''a -> predicted class 
	b -> a tuple including  accuracy (for classification), mean-squared 
        error, and squared correlation coefficient (for regression)
	c -> a list of decision values or probability estimates (if '-b 1' 
        is specified). If k is the number of classes, for decision values,
        each element includes results of predicting k(k-1)/2 binary-class
        SVMs. For probabilities, each element contains k values indicating
        the probability that the testing instance is in each class.
        Note that the order of classes here is the same as 'model.label'
        field in the model structure.'''
	return a
	






# ---------------------------  MAIN   -------------------------------------

wn_noun_table()
	
wn_verb_table()

pron_table()

x_data = []
print "\nHi, I am Joey ! Nice to meet you :) "	
fo = open("Training_tuples.txt","r+")
text = fo.read()
text = text[1:].split("]]")
data = (text[0]).split("], ")
for i in data:
	j = i[1:].split(", ")
	x_data = []
	for x in j:
		x_data.append(int(x))
	train_x.append(x_data)
	del x_data
data = (text[1])[1:-2]
data = data.split(",")		
for i in data:
	#print i
	train_y.append(int(i[1:]))
#print train_x
#print train_y
fo.close()

fo = open("Training_ref.txt","r+")
text = fo.read()
text = text[1:].split("]]")
data = (text[0]).split("], ")
for i in data:
	#print i
	j = i[1:].split(",")
	#print j
	#x_data = []
	#x_data.append(j[0])
	#for x in j[1:]:
	#x_data.append(,int(data[1]))
	train_x_ref.append([int(j[0]),int(j[1]),int(j[2]),int(j[3])])
	#del x_data
data = (text[1])[1:-2]
data = data.split(",")	
#print data	
for i in data:
	
	if i!=" ":
		#print i
		train_y_ref.append(int(i[1:]))
#print train_x
#print train_y
fo.close()

'''fo = open("Training_ref.txt","r+")
text = fo.read()
f_inp = text.split("\n")
for i in f_inp:
	if i == "":
		break
	data = i.split("],")
	print data
	j = ((data[0])[1:]).split(",")
	print j
	print "Training reference resolution ",[j[0],int(j[1]),int(j[2]),int(j[3])],int(data[1]) 
	train_ref.append(([j[0],int(j[1]),int(j[2]),int(j[3])],int(data[1])))
	
fo.close()'''



f_out = open("log_c.txt", "a")
sent_out = open("inputs_c.txt", "a")
localtime = time.asctime( time.localtime(time.time()) )
print>>f_out, "\n",localtime


labeled_names = ([(name, 'male') for name in names.words('male.txt')] + [(name, 'female') for name in names.words('female.txt')])
random.shuffle(labeled_names)
featuresets = [(gender_features(n), gender) for (n, gender) in labeled_names]
trainset, testset = featuresets[500:], featuresets[:500]
classifier = nltk.NaiveBayesClassifier.train(trainset)

#print "Gender match:",gender_match("her","John")
#print "Number match:",number_match("it","John")

while True:

	
	choice = raw_input("\nEnter one of the following : \n 1 Train me with a file \n 2 Test me with a file \n 3 Train me with annotated file \n 4 Extract Relations Yourself \n 5 Quit\n")

	if choice == '1':
		f_name = raw_input("Enter file name to annotate :")
		fo = open(f_name,"r+")
		text = fo.read()
		#f_inp = file_inp.split(".")
		#num = 0
		f_inp = nltk.sent_tokenize(text)
		#global ref
		train_set = []
		print>>f_out, "Training\n"
		for inp in f_inp:
		
		#inp = raw_input("Enter a sentence: ")

		#train_set = []
	
		
			print inp
			print>>f_out, inp
			print>>sent_out, inp
			num = 1
			person = []	# to store all names of persons in sentence
			loc = []	# to store all names of locations in sentence
			org = []	# to store all names of organisations in sentence
			gpe = []	# to store all names of gpe in sentence	
			
			dist = 0	# to find distance between two entities in the tree
			ftr = []	# to store feature matrix
			pos = 0		# to get position of noun in the feature matrix
			flag = 0	# flag to check if reached another noun
	
			semantic_lists(inp)				# create lists for different named entities
		
			#concept_table()				# create look up table for ontology classes
			'''for k,v in t_concept.iteritems():
			print k,v
			'''
			#property_table()			# create look up table for ontology properties
			'''for k,v in t_property.iteritems():
			print k,v'''
		
			#print tree
			#print ner_res
			#extract_noun(tree)
	
			out = parse(inp)				# parse the input to get the parse tree
			#print out
			temp_str = str("") 				# string that stores the entire parse tree 
	
			for i in out:
				temp_str = temp_str + str(i)
			#ner_res.draw()
			#t = ParentedTree.fromstring(str(ner_res))
			t = ParentedTree.fromstring(temp_str)		# get the parse tree structure
			#t = Tree.fromstring(temp)
			#t.draw()
		
		
			main_traverse(t)				# do dfs and get the feature matrix
	
			#print ftr
	
			#m = train_classifier()				# train the classifier
		
			for i in range(0,len(ftr)-1):			# forming feature vectors from the matrix and giving them to classifier
				j = i + 1
				#train_set[:] = []
				if ftr[i][0] == ftr[j][0]:		# if same entity has 2 relations and 2 rows in matrix it should not have any relation with itself
					continue
				while j < len(ftr):
				
					if (ftr[i][5]!=0):
						temp = []		# creating the feature vector
						temp.append(ftr[i][1])
						temp.append(ftr[i][2])
						temp.append(ftr[i][3])
						temp.append(ftr[j][4] - ftr[i][4])
						temp.append(ftr[j][1])
						temp.append(ftr[j][2])
						temp.append(ftr[i][5])
						temp.append(ftr[j][5])
						train_set.append(temp)
						#pred = test_classifier(m,temp)
						#print ftr[i][0],ftr[j][0],temp
						print num,ftr[i][0],ftr[j][0],(((str(get_property(ftr[i][5])).split('\''))[1]).split('.'))[0]
						print>>f_out, num,ftr[i][0],ftr[j][0],(((str(get_property(ftr[i][5])).split('\''))[1]).split('.'))[0]
							#for k,v in t_property.iteritems():
						#	if v == temp[len(temp)-1]:
						#		print k
						#		break  
						num = num + 1
						del temp 

	 				if (ftr[i][5] != ftr[j][3] and ftr[j][3]!=0):		# if more than one verb in between the two nouns
						temp = []			# creating the feature vector
						temp.append(ftr[i][1])
						temp.append(ftr[i][2])
						temp.append(ftr[i][3])
						temp.append(ftr[j][4] - ftr[i][4])
						temp.append(ftr[j][1])
						temp.append(ftr[j][2])
						temp.append(ftr[j][3])
						temp.append(ftr[j][5])
						train_set.append(temp)
						#classifier(temp)
						#print ftr[i][0],ftr[j][0],temp
						#pred = test_classifier(m,temp)
						if ftr[j][3]:
							print num,ftr[i][0],ftr[j][0],(((str(get_property(ftr[j][3])).split('\''))[1]).split('.'))[0]#get_property(ftr[j][3])
							print>>f_out, num,ftr[i][0],ftr[j][0],(((str(get_property(ftr[j][3])).split('\''))[1]).split('.'))[0]
						else:
							print num,ftr[i][0],ftr[j][0], "Unknown"
							print>>f_out, num,ftr[i][0],ftr[j][0], "Unknown"
						#for k,v in t_property.iteritems():
						#	if v == temp[len(temp)-1]:
							#		print k
						#		break  
						num = num + 1
						del temp 

					if (ftr[i][5] == ftr[j][3] and ftr[j][3]==0):
						temp = []		# creating the feature vector
						temp.append(ftr[i][1])
						temp.append(ftr[i][2])
						temp.append(ftr[i][3])
						temp.append(ftr[j][4] - ftr[i][4])
						temp.append(ftr[j][1])
						temp.append(ftr[j][2])
						temp.append(ftr[j][3])
						temp.append(ftr[j][5])
						train_set.append(temp)
						#pred = test_classifier(m,temp)
						#print ftr[i][0],ftr[j][0],temp
						print num,ftr[i][0],ftr[j][0], "Unknown" #get_property(ftr[j][3])
						print>>f_out, num,ftr[i][0],ftr[j][0], "Unknown"
						num = num + 1
						del temp 

					j = j + 1
			print "\nEnter relation number [space] relation: (Press 0 to stop entering) "
			while True:
				response = raw_input()
				response = response.split(' ',1)	# to split into max 2 elements by space 
				if int(response[0]) == 0:		# response[0] = the relation number, response[1] = the relation to be annotated
					break

				if ( int(response[0]) > len(train_set) ):
					print "Incorrect Format. Try again "
					continue
				train_x.append(train_set[int(response[0])-1])
				train_y.append(match_property_wn(response[1]))
				print>>f_out, "\n",response[0]," ",response[1]

				
			#m = train_classifier()
	



	

	elif choice == '2':

		f_name = raw_input("Enter file name to test :")
		fp = open(f_name,"r+")
		text = fp.read()
		#f_inp = file_inp.split(".")
		#num = 0
		#global ref
		file_inp = nltk.sent_tokenize(text)
		#inp = raw_input("Enter test sentence : ")		# input 
		print>>f_out, "Testing\n"
		for inp in file_inp:
		
		#inp = raw_input("Enter test sentence : ")	# input 
			print inp
			print>>f_out, inp
			print>>sent_out, inp
			train_set = []
			num = 1
			person = []					# to store all names of persons in sentence
			loc = []					# to store all names of locations in sentence
			org = []					# to store all names of organisations in sentence
			gpe = []					# to store all names of gpe in sentence			
			dist = 0					# to find distance between two entities in the tree
			ftr = []					# to store feature matrix
			pos = 0						# to get position of noun in the feature matrix
			flag = 0					# flag to check if reached another noun

			semantic_lists(inp)				# create lists for different named entities
		
			#concept_table()				# create look up table for ontology classes
			'''for k,v in t_concept.iteritems():
					print k,v
			'''
			#property_table()				# create look up table for ontology properties
			'''for k,v in t_property.iteritems():
				print k,v'''
		
	
			out = parse(inp)				# parse the input to get the parse tree
	
			#print out
			temp_str = str("") 				# string that stores the entire parse tree 
	
			for i in out:
				temp_str = temp_str + str(i)
			#ner_res.draw()
			#t = ParentedTree.fromstring(str(ner_res))
			t = ParentedTree.fromstring(temp_str)		# get the parse tree structure
			#t = Tree.fromstring(temp)
			#t.draw()
		
			t_copy = t
			main_traverse(t)				# do dfs and get the feature matrix
			t = t_copy	
		
			print>>f_out, ftr
			print "\nExtracted relations : "
		
			m = train_classifier()				# train the classifier
			test_res = []
			train_sug = []
			for i in range(0,len(ftr)-1):			# forming feature vectors from the matrix and giving them to classifier
				j = i + 1
				if ftr[i][0] == ftr[j][0]:		# if same entity has 2 relations and 2 rows in matrix it should not have any relation with itself
					continue
				while j < len(ftr):
					if (ftr[i][5]!=0):
						temp = []		# creating the feature vector
						temp.append(ftr[i][1])
						temp.append(ftr[i][2])
						temp.append(ftr[i][3])
						temp.append(ftr[j][4] - ftr[i][4])
						temp.append(ftr[j][1])
						temp.append(ftr[j][2])
						temp.append(ftr[i][5])
						temp.append(ftr[j][5])
					
						#print ftr[i][0],ftr[j][0],temp
						st = (((str(get_property(ftr[i][5])).split('\''))[1]).split('.'))[0]
						if st != 'Unknown' and st != None and ftr[i][0] != None and ftr[j][0] != None:
							train_sug.append([ftr[i][0]+" "+ftr[j][0]+" "+st])
							'''print "\nTraining suggestions :"
							print>>f_out, "\nTraining suggestions :"
							print ftr[i][0],ftr[j][0],st
							print>>f_out, ftr[i][0],ftr[j][0],st'''
							#num = num + 1

						pred = test_classifier(m,temp)
						train_set.append(temp)
					
						#print "\nTest Results :"
						if pred != 'Unknown':
							#print "here",pred
							test_res.append([str(num)+" "+ftr[i][0]+" "+ftr[j][0]+" "+(((str(wn.lemma_from_key(pred)).split('\''))[1]).split('.'))[0]])
							#print "\n",num,ftr[i][0],ftr[j][0],(((str(wn.lemma_from_key(pred)).split('\''))[1]).split('.'))[0]
							#print>>f_out, num,ftr[i][0],ftr[j][0],(((str(wn.lemma_from_key(pred)).split('\''))[1]).split('.'))[0]
						else:
							test_res.append([str(num)+" "+ftr[i][0]+" "+ftr[j][0]+" Unknown"])
							#print "\n",num,ftr[i][0],ftr[j][0],"Unknown"
							#print>>f_out, ftr[i][0],ftr[j][0],"Unknown"
				
						#for k,v in t_property.iteritems():
						#	if v == temp[len(temp)-1]:
						#		print k
							#		break  
						num = num + 1
						del temp 
	 	
					if (ftr[i][5] != ftr[j][3] and ftr[j][3]!=0):		# if more than one verb in between the two nouns
						temp = []			# creating the feature vector
						temp.append(ftr[i][1])
						temp.append(ftr[i][2])
						temp.append(ftr[i][3])
						temp.append(ftr[j][4] - ftr[i][4])
						temp.append(ftr[j][1])
						temp.append(ftr[j][2])
						temp.append(ftr[j][3])
						temp.append(ftr[j][5])
						train_set.append(temp)
						st = (((str(get_property(ftr[j][3])).split('\''))[1]).split('.'))[0]
						#classifier(temp)
						#print ftr[i][0],ftr[j][0],temp
						#print ftr[i][0],ftr[j][0]
						if st != "Unknown" and st != None and ftr[i][0] != None and ftr[j][0] != None:
							train_sug.append([ftr[i][0]+" "+ftr[j][0]+" "+st])
							'''print "\nTraining suggestions:"
							print>>f_out, "\nTraining suggestions:"
							print ftr[i][0],ftr[j][0],st
							print>>f_out, ftr[i][0],ftr[j][0],st'''
							#num = num + 1
					
						pred = test_classifier(m,temp)

						#print "\nTest Results :"
						if pred != "Unknown":
							test_res.append([str(num)+" "+ftr[i][0]+" "+ftr[j][0]+" "+(((str(wn.lemma_from_key(pred)).split('\''))[1]).split('.'))[0]])
							#print "\n",num,ftr[i][0],ftr[j][0],(((str(wn.lemma_from_key(pred)).split('\''))[1]).split('.'))[0]
							#print>>f_out, num,ftr[i][0],ftr[j][0],(((str(wn.lemma_from_key(pred)).split('\''))[1]).split('.'))[0]
						else:
							test_res.append([str(num)+" "+ftr[i][0]+" "+ftr[j][0]+" Unknown"])
							#print "\n",num,ftr[i][0],ftr[j][0],"Unknown"
							#print>>f_out, num,ftr[i][0],ftr[j][0],"Unknown"
						#for k,v in t_property.iteritems():
							#	if v == temp[len(temp)-1]:
						#		print k
						#		break  
						num = num + 1
						del temp 

					if (ftr[i][5] == ftr[j][3] and ftr[j][3]==0):
						temp = []		# creating the feature vector
						temp.append(ftr[i][1])
						temp.append(ftr[i][2])
						temp.append(ftr[i][3])
						temp.append(ftr[j][4] - ftr[i][4])
						temp.append(ftr[j][1])
						temp.append(ftr[j][2])
						temp.append(ftr[j][3])
						temp.append(ftr[j][5])
				
						train_set.append(temp)
					
						pred = test_classifier(m,temp)

						#print "\nTest Results :"
						#print ftr[i][0],ftr[j][0],temp
						test_res.append([str(num)+" "+ftr[i][0]+" "+ftr[j][0]+" Unknown"])
						#print "\n",num,ftr[i][0],ftr[j][0],"Unknown"#,wn.lemma_from_key(pred)
						#print>>f_out, num,ftr[i][0],ftr[j][0],"Unknown"#,wn.lemma_from_key(pred)
						num = num + 1
						del temp 

					j = j + 1

		
			total = total + num

			if len(train_sug)>0:
				print "\nTraining suggestions :"
				print>>f_out, "\nTraining suggestions :"
				for i in train_sug:
					#print "\n"
					for j in i:
						print j," "
					for j in i:
						print>>f_out,j," "

			if len(test_res)>0:
				print "\nTest Results :"
				print>>f_out, "\nTest Results :"
				for i in test_res:
					#print "\n"
					for j in i:
						print j," "
					for j in i:
						print>>f_out,j," "


			print "\nEnter correctly classified relations: (Enter number and relation , Enter 0 to stop entering) "
			print>>f_out, "\nEnter correctly classified relations: "
			#TP = 0

			while True:
				response = raw_input()
				response = response.split(' ',1)	# to split into max 2 elements by space 
				if int(response[0]) == 0:		# response[0] = the relation number, response[1] = the relation to be annotated
					break
				TP = TP + 1
				if ( int(response[0]) > len(train_set) ):
					print "Incorrect Format. Try again "
					continue
				train_x.append(train_set[int(response[0])-1])
				train_y.append(match_property_wn(response[1]))
				print>>f_out, "\n",response[0]," ",response[1]

			print "\nEnter wrongly classified relations and rectify them: (Enter number and relation , Enter 0 to stop entering) "
			print>>f_out, "\nEnter wrongly classified relations and rectify them: "
			#FP = 0

			while True:
				response = raw_input()
				response = response.split(' ',1)	# to split into max 2 elements by space 
				if int(response[0]) == 0:		# response[0] = the relation number, response[1] = the relation to be annotated
					break
				FP = FP + 1
				if ( int(response[0]) > len(train_set) ):
					print "Incorrect Format. Try again "
					continue
				train_x.append(train_set[int(response[0])-1])
				train_y.append(match_property_wn(response[1]))
				print>>f_out, "\n",response[0]," ",response[1]

			print "\nMissing relations : (Y/N)"		# entities of missing relations are from among the combinations given already
			print>>f_out, "\nMissing relations : (Y/N) "		
			#FN = 0
			response = raw_input()
			if response == 'Y' or response == 'y':
				print "\nExisting subjects : (Y/N)"
				print>>f_out, "\nExisting subjects : (Y/N)"
				response = raw_input()
				if response == 'Y' or response == 'y':
					print "\nEnter number and relation , Enter 0 to stop entering :"
					print>>f_out, "\nEnter number and relation , Enter 0 to stop entering :"
					while True:
						response = raw_input()
						response = response.split(' ')		# to split into max 2 elements by space 
						if int(response[0]) == 0:		# response[0] = the relation number, response[1] = the relation to be annotated
							break
						total = total + 1
						'''check_misclassified = 1
						if check_misclassified == 1:
							print>>sent_out, "\n",inp
							check_misclassified = 0'''
						FN = FN + 1
						if ( int(response[0]) > len(train_set) ):
							print "Incorrect Format. Try again "
							continue
						train_x.append(train_set[int(response[0])-1])
						train_y.append(match_property_wn(response[1]))
						print>>f_out, "\n",response[0]," ",response[1]

				else:
					while True:
						print "\nEnter subject : (Enter 0 to stop)"
						print>>f_out, "\nEnter subject : (Enter 0 to stop)"
						sub = raw_input()
						if sub == '0':
							break
						total = total + 1
						print "\nEnter object :"
						print>>f_out, "\nEnter object :"
						obj = raw_input()

						print "\nEnter relation :"
						print>>f_out, "\nEnter relation :"
						rel = raw_input()

						get_vector(sub,obj,t,match_property_wn(rel))
					
						t = t_copy
		

	elif choice == '3':
		x_data = []
		f_name = raw_input("Enter annotated file name :")
		fo = open(f_name,"r+")
		text = fo.read()
		text = text[1:].split("]]")
		#print text
		data = (text[0]).split("], ")
		#print data
		for i in data:
			j = i[1:].split(", ")
			x_data = []
			for x in j:
				#print x
				x_data.append(int(x))
			train_x.append(x_data)
		data = (text[1])[1:-2]
		data = data.split(",")		
		for i in data:
			#print i
			train_y.append(int(i[1:]))
		#print train_x
		#print train_y
		fo.close()

	elif choice == '4':
		print "Extract relations from the given sentences :\n"
		f_name = "Sentences.txt"
		fo = open(f_name,"r+")
		text = fo.read()
		f_inp = nltk.sent_tokenize(text)
		
		rel_out = open("Relations.txt", "a")

		for inp in f_inp:
			print inp
			print>>rel_out, "\n",localtime
			print>>rel_out, "\n",inp
			print "\nEnter relations that you see (Enter number and relation , Enter 0 to stop entering) :\n"
			while True:
				response = raw_input()
				if int(response[0]) == 0:		
					break
				print>>rel_out, response
			
		print>>rel_out, "==================================================================================================================================\n"
		rel_out.close()
		
	
	else:
		#print "Outputing training data to file Training_tuples.txt"
		out = open("Training_tuples.txt", "w")
		print>>out, train_x
		print>>out, train_y
		#print>>out, wn_noun
		out.close()
	
		out = open("Training_ref.txt", "w")
		print>>out, train_x_ref
		print>>out, train_y_ref
		out.close()
		break

TN = total - TP - FP - FN

if TP+FP == 0:
	P = "Undefined"
else:
	P = float("{0:.2f}".format((TP/float(TP+FP))))


if TP+FN == 0:
	R = "Undefined"

else:
	R = float("{0:.2f}".format((TP/float(TP+FN))))



if P == "Undefined" or R == "Undefined":
	F = 0

elif P+R == 0:
	F = "Undefined"

else:
	F = float("{0:.2f}".format((2*(P*R)/float(P+R))))

if total != 0:
	A = float("{0:.2f}".format(((TP+TN)/float (total))))
else:
	A = "Undefined"

print>>f_out, "\nFalse Positive : ", FP,"\nTrue Positive : ",TP,"\nFalse Negative : ",FN,"\nTrue Negative : ",TN, "\nPrecision : ",P, "\nRecall : ",R, "\nF-Measure : ",F, "\nAccuracy : ",A  
#print>>f_out, "==================================================================================================================================\n"

#FNR = totalR - TPR - FPR - TNR

totalR = TPR + FPR + TNR + FNR

if TPR+FPR == 0:
	PR = "Undefined"
else:
	PR = float("{0:.2f}".format((TPR/float(TPR+FPR))))


if TPR+FNR == 0:
	RR = "Undefined"

else:
	RR = float("{0:.2f}".format((TPR/float(TPR+FNR))))



if PR == "Undefined" or RR == "Undefined":
	FR = 0

elif PR+RR == 0:
	FR = "Undefined"

else:
	FR = float("{0:.2f}".format((2*(PR*RR)/float(PR+RR))))

if totalR != 0:
	AR = float("{0:.2f}".format(((TPR+TNR)/float (totalR))))
else:
	AR = "Undefined"

print>>f_out, "\nFor Resolution: ","\nFalse Positive : ", FPR,"\nTrue Positive : ",TPR,"\nFalse Negative : ",FNR,"\nTrue Negative : ",TNR, "\nPrecision : ",PR, "\nRecall : ",RR, "\nF-Measure : ",FR, "\nAccuracy : ",AR  
print>>f_out, "==================================================================================================================================\n"

f_out.close()

print "Goodbye! And have a good day :)"
sent_out.close()



#-----------------------------  printing the look up tables  ----------------------------------------------------

#for k,v in t_concept.items():					# print contents of concept table
#	print k,':',v

#print "--------------------------------------------------------"

#for k,v in t_property.items():					# print contents of property table
#	print k,':',v





