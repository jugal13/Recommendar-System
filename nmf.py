import numpy as np 
import math
from scipy import spatial
R = np.array([
	
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [1, 0, 0, 4],
    [0, 1, 5, 4],
	
	])
	
user,item=R.shape
alpha,beta,iterations=0.1,0.01,20 #alpha is a learning parameter, beta is for regularization. as iterations increase, a more fine tuned recomm. is obtained
K=2 #no. of dimensions

def full_matrix(b,b_u,b_i,P,Q): #returns PxQ
	return b+b_u[:,np.newaxis] + b_i[np.newaxis:,] + P.dot(Q.T)

def get_rating(i,j,b,b_u,b_i,P,Q): #returns rating
	return b + b_u[i] + b_i[j] + P[i, :].dot(Q[j, :].T)

def sgd(P,Q,samples,b,b_u,b_i):
	global alpha
	global beta
	for i,j,r in samples:
		prediction = get_rating(i,j,b,b_u,b_i,P,Q)
		e=(r-prediction)

		b_u[i]+= alpha*(e- beta*b_u[i])
		b_i[j]+= alpha*(e- beta*b_i[j])

		P[i, :] += alpha * (e*Q[j, :] - beta*P[i, :])
		Q[j, :] += alpha * (e*P[i, :] - beta*Q[j, :])

def mse(R,b,b_u,b_i,P,Q):
	xs,ys=R.nonzero()
	predicted = full_matrix(b,b_u,b_i,P,Q); 
	error = 0;
	for x,y in zip(xs,ys):
		error+= (R[x,y]-predicted[x,y]) ** 2
		#print "Square error : ",error
	return np.sqrt(error)

def train(R,user,item,K,iterations):
	P=np.random.normal(scale=1./K, size=(user,K))
	Q=np.random.normal(scale=1./K, size=(item,K))
	#b_u = user bias, b_i = item bias, b= bias. Bias is set on mean, where R!=0
	b_u=np.zeros(user)
	b_i=np.zeros(item)
	b=np.mean(R[np.where(R != 0)])
	samples=[
				(i,j,R[i,j])
				for i in range(user)
				for j in range(item)
				if R[i,j]>0
			]
	training_process = []
	for i in range(iterations):
		np.random.shuffle(samples)
		sgd(P,Q,samples,b,b_u,b_i) #stochastic gradient descent
		mean_sq_error=mse(R,b,b_u,b_i,P,Q)   #mean square error
		training_process.append((i,mean_sq_error))
		if (i+1)%10 == 0:
			print "Iteration : ",i+1," error : ",mean_sq_error
	#print "\nFactorized Matrix : \n"
	return training_process, full_matrix(b,b_u,b_i,P,Q)
def normalize(fact_mat,scale):
	biggest , smallest = -float('inf') , float('inf');
	print "\nFactorized Matrix : \n"
	for row in fact_mat:
		for item in row:
			print "%1.5f "%(item),
		print " "
	new_list = [] #1D array. 0(n) to print , make , evaluate.
	#Find biggest element in the matrix. Since review scale is based on 'scale' stars, if matrix has any element > scale ; normalize matrix
	#Find smallest element in the matrix. A movie review cant be -ve, so if smallest < 0; normalize matrix
	for row in fact_mat:
		for item in row:
			if item> biggest:
				biggest=item
			if item < smallest:
				smallest=item
	print "\nBiggest element in the matrix : ",biggest,"\nSmallest element in the matrix : ",smallest
	if biggest > scale or smallest < 0:
		print"\nNormalized matrix : \n"
		for row in fact_mat:
			for item in row:
				item=item/biggest;item=scale*item; item=abs(item)
				new_list.append(item)
	else:
		print "No need to normalize matrix"
	return new_list
#get the factorized matrix
print "User Intended Matrix : "
for rows in R:
	for item in rows:
		print item,
	print " "
print "Training Model : \n"
trained_model, fact_mat = train(R,user,item,K,40)
#normalize function takes in the factorized matrix and max approved rating as input, gives normalized matrix as output
ans_mat = normalize(fact_mat,5); i = 0;
for row in ans_mat:
	i+=1;
	print "%1.5f "%(row),
	if i>3:
		print " "; i = 0;
print "\nCosine Similarity : User to User\n"
l1,l2=[],[]
Ai= []
for i in range(len(R)-1):
	for j in range(i, len(R)-1):
		l1.append(R[i])
		l2.append(R[j])
		desc= 1-spatial.distance.cosine(R[i],R[j])
		if desc != 1:
			print "Cosine similarity for user ",i+1," and user ",j+1," is : %1.5f"%(desc)
			Ai.append([i+1,j+1,desc])
print "\nCosine Similarity calculated\n"
for item in Ai:
	if item[2] > 0.68:
		print "CS relevance of user-user : U1 - ",item[0]," U2 - ",item[1]," Similarity - %1.5f"%(item[2])
