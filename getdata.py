import json
import csv
import numpy as np 
import pandas as pd 
from pprint import pprint
x, y = [], []
with open('movies.csv','r') as csvfile:
	plot = csv.reader(csvfile,delimiter=',')
	for row in plot:
		x.append(row[9])
		y.append(row[4])
# 2851 movies, 2851 users, 2851 ratings; movies.csv
print"\n****************Stats****************\n"
data={'Movies':x,'Ratings':y}
df=pd.DataFrame(data)
# 919 users; 10,000 ratings; users.csv
language,gender,state,job=[],[],[],[];age=[]
with open('users.csv','r') as cf:
	p=csv.reader(cf,delimiter=',')
	for row in p:
		language.append(row[1])
		gender.append(row[5])
		job.append(row[2])
		age.append(row[4][6:])
male=0;student,service,others=0,0,0;se=0
for rows in gender:
	if rows.lower()=="male":
		male+=1
for work in job:
	if work.lower() == "student":
		student+=1;
	elif work.lower() == "service":
		service+=1
	elif work.lower() == "others":
		others+=1
	else:
		se+=1
print "Movie [ Watchers ] Distribution : \n\n","Students : ",student,"\nService : ",service,"\nSelf-Employed : ",se,"\nOthers : ",others
print "\nMovie Watchers : Gender Distribution : \n\n","Male : ",male,"\nFemale : ",919-male
print "\nMovie Watchers : Age Distribution : \n\n";i=0
age=[x.strip(' ') for x in age]
for years in age:
	if i>20:
		print" ";i=0;
	else:
		print years,
		i+=1;
