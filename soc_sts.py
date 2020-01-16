


#

import math
P = "Many consider Maradona as the best player in soccer history"
R = "Maradona is one of the best soccer player"

def wordSim(P = P, R = R):
	#P = P.strip().split()
	#R = R.strip().split()
	if len(R) > len(P):
		P, R = R, P
	common = {}
	count = 1
	for i in P:
		if i in R:
			common.setdefault(i, [])
			common[i].append(count)
			count += 1

	#print common, P
	count = 1
	for i in R:
		if i in common:
			common[i].append(count)
			count += 1
	#print common, P
	sumi = 0.0
	for i in common:
		sumi += abs(common[i][0] - common[i][1])
	#print sumi
	#Calculating Similiarity
	if len(common) == 0:
		return 0, []
	try:
		if len(common)%2 ==0:
			return 1 - (2 * sumi / float(len(common)**2)), common
		elif len(common)%2!=0 and len(common)>1:
			return 1 - (2 * sumi / (float(len(common)**2) - 1)), common
		elif len(common)%2!=0 and len(common)==1:
			return 1, common
	except:
		return 0, []	
#print wordSim()


def display(array, len1, len2, string1, string2):
	print(string2)
	for i in range(1, len1+1):
		print (string1[i-1], array[i])

def dynammicProg(string1="Pritish", string2="Yuvraj"):
	len1, len2 = len(string1), len(string2)
	string1, string2 = string1.lower(), string2.lower()
	array = [[0 for i in range(len2+1)] for j in range(len1+1)]
	for i in range(len1+1):
		for j in range(len2+1):
			if i==0 or j==0:
				array[i][j] = 0
				continue
			if string1[i-1] == string2[j-1]:
				array[i][j] = array[i-1][j-1] + 1
			else:
				array[i][j] = max(array[i-1][j], array[i][j-1])
	#print array
	#print "Printing"
	#display(array, len1, len2, string1, string2)
	return array[i][j]

def starting(string1="Pritish", string2="Pritieish"):
	if len(string2) > len(string1):
		string1, string2 = string2, string1
	count = 0
	string1, string2 = string1.lower(), string2.lower()
	for i in range(len(string2)):
		if string1[i] == string2[i]:
			count += 1
	return count

def maxCommon(string1="Pritish", string2="Yuvraj"):
	if len(string2) > len(string1):
		string1, string2 = string2, string1
	string1, string2 = string1.lower(), string2.lower()
	count, i, j = 0, 0, 0
	maxi = 0
	while(count<len(string2)):
		j = count
		#print "Loop1", i, j
		while i<len(string1) and j<len(string2):
			#print "Loop2", i, j, string1[i], string2[j], string1, string2
			if string1[i]!=string2[j]:
				i += 1
				continue
			else:
				temp = 0
				while(string1[i]==string2[j]):
					#print "Loop3", i, j, string1[i]
					i += 1
					j += 1
					temp += 1		
					if i>=len(string1) or j>=len(string2):
						break
				if temp > maxi: 
					maxi = temp
				if i>=len(string1) or j>=len(string2):
					break
		count += 1
		#print "Count", count
	#print "Common", maxi
	return maxi

def calling(string1 = "place", string2="land"):
	LCS = dynammicProg(string1, string2)
	#print LCS, float(LCS**2), float(len(string1)),float(len(string2))
	V1 = float(LCS**2) / (float(len(string1)) * float(len(string2)) )

	start = starting(string1, string2)
	#print start
	V2 = float(start)**2 / (float(len(string1))* float(len(string2)))

	nmax = maxCommon(string1, string2)
	#print nmax
	V3 = float(nmax)**2 / (float(len(string1))*float(len(string2)))
	#print V1, V2, V3
	alpha = 0.33 * (V1 + V2 + V3)
	#print alpha
	return alpha











#try1.py file

import os
try:
	os.remove("try1.pyc")
except:
	#print ("Duplicate Module doesn't exist")
	pass
	
def display(array, len1, len2, string1, string2):
	print (list(string2))
	for i in range(1, len1+1):
		print (string1[i-1], array[i])

def dynammicProg(string1="Pritish", string2="Yuvraj"):
	len1, len2 = len(string1), len(string2)
	string1, string2 = string1.lower(), string2.lower()
	array = [[0 for i in range(len2+1)] for j in range(len1+1)]
	for i in range(len1+1):
		for j in range(len2+1):
			if i==0 or j==0:
				array[i][j] = 0
				continue
			if string1[i-1] == string2[j-1]:
				array[i][j] = array[i-1][j-1] + 1
			else:
				array[i][j] = max(array[i-1][j], array[i][j-1])
	#print array
	#print "Printing"
	#display(array, len1, len2, string1, string2)
	return array[i][j]

def starting(string1="Pritish", string2="Pritieish"):
	if len(string2) > len(string1):
		string1, string2 = string2, string1
	count = 0
	string1, string2 = string1.lower(), string2.lower()
	for i in range(len(string2)):
		if string1[i] == string2[i]:
			count += 1
	return count

def maxCommon(string1="Pritish", string2="Yuvraj"):
	if len(string2) > len(string1):
		string1, string2 = string2, string1
	string1, string2 = string1.lower(), string2.lower()
	count, i, j = 0, 0, 0
	maxi = 0
	while(count<len(string2)):
		j = count
		#print "Loop1", i, j
		while i<len(string1) and j<len(string2):
			#print "Loop2", i, j, string1[i], string2[j], string1, string2
			if string1[i]!=string2[j]:
				i += 1
				continue
			else:
				temp = 0
				while(string1[i]==string2[j]):
					#print "Loop3", i, j, string1[i]
					i += 1
					j += 1
					temp += 1		
					if i>=len(string1) or j>=len(string2):
						break
				if temp > maxi: 
					maxi = temp
				if i>=len(string1) or j>=len(string2):
					break
		count += 1
		#print "Count", count
	#print "Common", maxi
	return maxi

def calling(string1 = "place", string2="land"):
	LCS = dynammicProg(string1, string2)
	#print LCS, float(LCS**2), float(len(string1)),float(len(string2))
	V1 = float(LCS**2) / (float(len(string1)) * float(len(string2)) )

	start = starting(string1, string2)
	#print start
	V2 = float(start)**2 / (float(len(string1))* float(len(string2)))

	nmax = maxCommon(string1, string2)
	#print nmax
	V3 = float(nmax)**2 / (float(len(string1))*float(len(string2)))
	#print V1, V2, V3
	alpha = 0.33 * (V1 + V2 + V3)
	#print alpha
	return alpha









#wn3.py file
from nltk.corpus import wordnet

def returnWordSim(word1, word2):
	word1 = wordnet.synsets(word1)
	word2 = wordnet.synsets(word2)
	if word1 and word2:
		ans = word1[0].wup_similarity(word2[0])
		if ans == None:
			ans = 0.0
		return ans
	else: 
		return 0.0












#main.py 


import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

from utils import readCSV


def soc_sts(S1, S2):

	#import wordSim
	#import try1
	#import word as wd 
	#import wn3
	#S1 = raw_input("")
	#S2 = raw_input("")
	#S1 = "A cemetry is a place where dead people's bodies or their ashes are buried."
	#S2 = "A graveyard is an area of land, sometimes near a church, where dead people are buried."
	#S1 = "The president was assinated in his car"
	#S2 = "The president was assinated and driver could do nothing"
	#S1 = "The minor was raped inside a car, people stood and did nothing"
	#S2 = "The girl was raped by the driver and people did nothing"
	#S2 = "Cricket match was being played between two decisive teams"
	#S1, S2 = S1.strip().split(), S2.strip().split()
	tokenizer = RegexpTokenizer(r'\w+')
	S1 = tokenizer.tokenize(S1)
	S2 = tokenizer.tokenize(S2)

	#ps = PorterStemmer()
	#S1 = [ps.stem(word.lower()) for word in S1]
	#S2 = [ps.stem(word.lower()) for word in S2]
	ltz = WordNetLemmatizer()
	S1 = [ltz.lemmatize(word.lower()) for word in S1]
	S2 = [ltz.lemmatize(word.lower()) for word in S2]

	#print S1, S2
	'''
	tokenizer = RegexpTokenizer(r'\w+')
	S1 = tokenizer.tokenize(S1)
	S2 = tokenizer.tokenize(S2)
	'''
	S1_filtered = [word for word in S1 if word not in stopwords.words('english')]
	S2_filtered = [word for word in S2 if word not in stopwords.words('english')]
	#print (S1, S2)
	#print S1_filtered, S2_filtered
	#End of Step 1
	#print stopwords.words('english')

	#Start of Step 2
	score, common = wordSim(S1_filtered, S2_filtered)
	S1_next = [word for word in S1_filtered if word not in common]
	S2_next = [word for word in S2_filtered if word not in common]
	print ("Common", common)
	print ("Paragraph", S1_next, S2_next)
	h, w = len(S1_next), len(S2_next)
	Matrix1 = [[0.0 for x in range(w)] for x in range(h)]
	print (S2_next)
	#for i in range(len(Matrix1)):
	#	print (S1_next[i], Matrix1[i])
	for i in range(len(S1_next)):
		for j in range(len(S2_next)):
			Matrix1[i][j] = calling(S1_next[i], S2_next[j])
	print (S2_next)
	#for i in range(len(Matrix1)):
		#print (S1_next[i], Matrix1[i])
	#End of Step 3

	#Begining of Step 4
	Matrix2 = [[0.0 for x in range(w)] for x in range(h)]
	#print ("SOCPMI")
	for i in range(len(S1_next)):
		for j in range(len(S2_next)):
			Matrix2[i][j] = returnWordSim(S1_next[i], S2_next[j])
	print (S2_next)
	#for i in range(len(Matrix2)):
		#print (S1_next[i], Matrix2[i])
	#End of Step 4

	#Begining of Step 5
	Matrix = [[0.0 for x in range(w)] for x in range(h)]
	print ("Final Matrix")
	for i in range(len(S1_next)):
		for j in range(len(S2_next)):
			Matrix[i][j] = (0.5*Matrix1[i][j]) + (0.5*Matrix2[i][j])
	#print (S2_next)
	#for i in range(len(Matrix)):
		#print (S1_next[i], Matrix[i])
	#Looping to find Pi
	def delete(matrix, i, j):
		for row in matrix:
			del row[j]
		matrix = [matrix[i1] for i1 in range(len(matrix)) if i1 != i]
		return matrix
	Pi = []
	while(len(Matrix)>0 and len(Matrix[i])>0):
		#Search for maximum Element
		maxelement = 0
		maxi, maxj = 0, 0
		for i in range(len(Matrix)):
			for j in range(len(Matrix[i])):
				if Matrix[i][j] > maxelement:
					maxelement = Matrix[i][j]
					maxi = i
					maxj = j
		Pi.append(Matrix[maxi][maxj])
		Matrix = delete(Matrix, maxi, maxj)
		print ("Matrix")
		for i in range(len(Matrix)):
			print (Matrix[i])
		print ("Pi", Pi)

	#End of Step 5

	#Begining of Step 6
	Delta = 2.0
	try:
		similarity = ((Delta + sum(Pi)) * (len(S1_next) + len(S2_next)))/(2*len(S1_next)*len(S2_next))
	except:
		return 0
	print ("Similarity Score", similarity)
	return similarity



#S1 = "The president was assinated in his car"
#S2 = "The president was assinated and driver could do nothing"
#soc_sts(S1, S2)

#S1 = "Would you like to go out to drink with me tonight?"
#S2 = "I really don't know what to eat tonight so I might go out somewhere."

#soc_sts(S1, S2)



from scipy import stats
STSS_131_DATA = "data/STSS-131.csv"
sentences = readCSV(STSS_131_DATA)

sim_values, STSS_values = [], []
for s in sentences:
	#soc_sts(s.first_sentence, s.second_sentence)
	sim_values.append(soc_sts(s.first_sentence, s.second_sentence)*4)
	STSS_values.append(s.human_SS)

print("******************************************************")
p = stats.pearsonr(sim_values,STSS_values)
print("soc_sts correlation with STSS: ", p)


'''
S1 = "Would you like to go out to drink with me tonight?"
S2 = "I really don't know what to eat tonight so I might go out somewhere."
soc_sts(S1, S2)
'''