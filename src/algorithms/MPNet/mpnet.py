from typing import List, Tuple

import numpy as np

from algorithms.algorithm import Algorithm
from algorithms.basic_testing import BasicTesting
from algorithms.configuration.entities.agent import Agent
from algorithms.configuration.entities.goal import Goal
from algorithms.configuration.maps.map import Map
from algorithms.configuration.entities.obstacle import Obstacle
from simulator.services.services import Services
from simulator.views.map_displays.gradient_map_display import GradientMapDisplay
from simulator.views.map_displays.solid_color_map_display import SolidColorMapDisplay
from simulator.views.map_displays.map_display import MapDisplay
from simulator.views.map_displays.numbers_map_display import NumbersMapDisplay
from structures import Point
import copy


import torch
import torch.nn as nn
from torch.autograd import Variable 
import torch.utils.data as data
#import nltk
#from PIL import Image
import os
import os.path
import random
import math
import time
import pickle
from modelmpnet import MLP 

class Encoder(nn.Module):
	def __init__(self):
		super(Encoder, self).__init__()
		self.encoder = nn.Sequential(nn.Linear(2800, 512),nn.PReLU(),nn.Linear(512, 256),nn.PReLU(),nn.Linear(256, 128),nn.PReLU(),nn.Linear(128, 28))
			
	def forward(self, x):
		x = self.encoder(x)
		return x

def IsInCollision(x,idx, counter,linepath):#oblen):
	
	s=np.zeros(2,dtype=np.int32) #float32)
	
	#print('in isincollision')
	#print (linepath)
	#print('s in isincollsion', s)
	s = [x[0],x[1]]
	
	#print('obc counter len = ',obc[idx][:counter], type(obc))
	#print ('s', s)

	obslist = obc.tolist()[0][:counter]
	#print ('obslist ==', obslist)
	
	#for i in range(0, counter):#oblen):
		
	if s not in obslist:
		return False
	#there is collision
	return True
			
	# #print('point',str(i))
	# for j in range(0,2):
	#     #print('abs(obc[idx][i][j]) ', abs(obc[idx][i][j]), 's[j] ',s[j])
	#     print('obc[]['+str(i)+']['+str(j)+'] = ',obc[idx][i][j])
	#     if obc[idx][i][j] != s[j]:
			
	#         cf=False
	#         break
	#     # there's a collision
	#     if cf==True:						
	#         return True
	# return False


# check if a straight path can be connected between 2 states
def steerTo (start, end, idx, counter, grid1):#oblen):

	#print ('inside steerTo')
	
	DISCRETIZATION_STEP=1
	dists=np.zeros(2,dtype=np.float32)

	#print('start =', start, 'end =',end)
	
	## subtract waypoint from next point
	for i in range(0,2): 
		dists[i] = round(float(end[i])) - round(float(start[i]))
	
	#print('dist =', dists)

	## get straight line distance front waypoint to next point by sqrting (x^2 + y^2)
	distTotal = 0
	for i in range(0,2): 
		distTotal =distTotal+ dists[i]*dists[i]

	#print('distTotal = ', distTotal)
	distTotal = math.sqrt(distTotal)
	## disTotal becomes the straight line distance from waypoint to nextpoint
	#print('math.sqrt(distTotal) = ', distTotal)

	if distTotal>0:
		incrementTotal = distTotal/DISCRETIZATION_STEP
		#print ('incrementTotal = ', incrementTotal)
		for i in range(0,2): 
			dists[i] = dists[i]/incrementTotal
			#print('dists[i] =', dists[i])

		#numSegments = int(math.floor(incrementTotal))
		#print ('numSegments= ', numSegments)

		stateCurr = np.zeros(2,dtype=np.float32) #float32)
		for i in range(0,2): 
			
			if (start[i] - int(start[i])) >=0.5: 
				stateCurr[i] = math.ceil(start[i])
				#print ('stateCurr['+str(i)+']= ',stateCurr[i])
			else:
				stateCurr[i] = math.floor(start[i])
				#print ('stateCurr['+str(i)+']= ',stateCurr[i])

		## get start and endpoint for the grid.get_line_sequence func to see what cells are touched with straight line
		startpoint = Point(int(stateCurr[0]),int(stateCurr[1]))
		#print('startpoint =', startpoint)
		endpoint = Point(round(float(end[0])),round(float(end[1])))
		#print('endpoint =', endpoint)
		
		global linepath
		linepath = grid1.get_line_sequence(startpoint,endpoint)
		linepath = [[point.x,point.y] for point in linepath]
		#print('linepath ----', linepath)

		# for i in range(0,numSegments):
			
		#     ## stateCurr = waypoint ; idx = 0 ; counter = num of obstacles
		#     if IsInCollision(stateCurr,idx, counter):# oblen):
		#         print("there's collision")
		#         return 0

		#     for j in range(0,2):
		#         #print('in num segments ----- stateCurr['+str(j)+'] = ', stateCurr[j])
		#         stateCurr[j] = stateCurr[j]+(dists[j])
		
		for i in range(0,len(linepath)):
			
			#print ('linepath [i]', i, linepath[i])
			## stateCurr = waypoint ; idx = 0 ; counter = num of obstacles
			if IsInCollision(linepath[i],idx, counter, linepath):# oblen):
				print("there's collision")
				return 0

			# for j in range(0,2):
			#     #print('in num segments ----- stateCurr['+str(j)+'] = ', stateCurr[j])
			#     stateCurr[j] = stateCurr[j]+round(float(dists[j]))

		## end = end point ; idx = 0 ; counter = num of obstacles
		if IsInCollision(end,idx, counter,linepath): #oblen):
			print("there's collision")
			return 0
		print("there's no collision")
		return 1

# checks the feasibility of entire path including the path edges
def feasibility_check(path,idx, counter, grid1): #oblen):

	for i in range(0,len(path)-1):
		ind=steerTo(path[i],path[i+1],idx, counter, grid1) #oblen)
		if ind==0:
			return 0
	return 1


# checks the feasibility of path nodes only
def collision_check(path,idx, counter,linepath): #oblen):

	for i in range(0,len(path)):
		if IsInCollision(path[i],idx, counter,linepath): #oblen):
			return 0
	return 1

def to_var(x, volatile=False):
	if torch.cuda.is_available():
		x = x.cuda()
	return Variable(x, volatile=volatile)

def get_input(i,dataset,targets,seq,bs):
	bi=np.zeros((bs,18),dtype=np.float32)
	bt=np.zeros((bs,2),dtype=np.float32)
	k=0	
	for b in range(i,i+bs):
		bi[k]=dataset[seq[i]].flatten()
		bt[k]=targets[seq[i]].flatten()
		k=k+1
	return torch.from_numpy(bi),torch.from_numpy(bt)



def is_reaching_target(start1,start2):
	s1=np.zeros(2,dtype=np.float32)
	s1[0]=start1[0]
	s1[1]=start1[1]

	s2=np.zeros(2,dtype=np.float32)
	s2[0]=start2[0]
	s2[1]=start2[1]


	for i in range(0,2):
		if abs(s1[i]-s2[i]) > 1.0: 
			return False
	return True

#lazy vertex contraction 
def lvc(path,idx, counter, grid1): #oblen):

	for i in range(0,len(path)-1):
		for j in range(len(path)-1,i+1,-1):
			ind=0
			ind=steerTo(path[i],path[j],idx, counter, grid1) #oblen)
			if ind==1:
				pc=[]
				for k in range(0,i+1):
					pc.append(path[k])
				for k in range(j,len(path)):
					pc.append(path[k])

				return lvc(pc,idx, counter, grid1) #oblen)
				
	return path

def re_iterate_path2(p,g,idx,obs, counter, grid1, linepath): #oblen):
	step=0
	path=[]
	path.append(p[0])
	for i in range(1,len(p)-1):
		if not IsInCollision(p[i],idx, counter, linepath): #oblen):
			path.append(p[i])
	path.append(g)			
	new_path=[]
	for i in range(0,len(path)-1):
		target_reached=False

	 
		st=path[i]
		gl=path[i+1]
		steer=steerTo(st, gl, idx, counter, grid1) #oblen)
		if steer==1:
			new_path.append(st)
			new_path.append(gl)
		else:
			itr=0
			target_reached=False
			while (not target_reached) and itr<50 :
				new_path.append(st)
				itr=itr+1
				ip=torch.cat((obs,st,gl))
				ip=to_var(ip)
				st=mlp(ip)
				st=st.data.cpu()		
				target_reached=is_reaching_target(st,gl)
			if target_reached==False:
				return 0

	#new_path.append(g)
	return new_path

def replan_path(p,g,idx,obs, counter, grid1, linepath):
	print('______________inside replan path______________________')
	print(' -- p = ',p)
	print(' -- g = ',g)
	print(' -- idx = ',idx)
	print(' -- obs = ',obs)
	print(' -- counter = ',counter)
	print(' -- linepath = ',linepath)

	
	step=0
	path=[]
	path.append(p[0])

	for i in range(1,len(p)-1):
		if not IsInCollision(p[i],idx, counter,linepath): #oblen):
			path.append(p[i])
	path.append(g)
	print(' ------paht =', path)

	new_path=[]
	for i in range(0,len(path)-1):
		target_reached=False

		
		st=path[i]
		print(' -- st = ',st)
		gl=path[i+1]
		print(' -- gl = ',gl)
		steer=steerTo(st, gl, idx, counter, grid1) #oblen)
		print('steer = ', steer)
		if steer==1:
			new_path.append(st)
			new_path.append(gl)
		else:
			itr=0
			pA=[]
			pA.append(st)
			pB=[]
			pB.append(gl)
			target_reached=0
			tree=0
			while target_reached==0 and itr<50 :
				itr=itr+1
				if tree==0:
					oldst = st
					ip1=torch.cat((obs,st,gl))
					ip1=to_var(ip1)
					st=mlp(ip1)
					print ('----- new st = ', st)
					st=st.data.cpu()
					if (st[0]>=0 and st[0]<xw-0.5) and (st[1]>=0 and st[1]<yw-0.5):
						pA.append(st)
						tree=1  
					else:
						st = oldst
						tree=0
				else:
					oldgl = gl
					ip2=torch.cat((obs,gl,st))
					ip2=to_var(ip2)
					gl=mlp(ip2)
					print ('----- new gl = ', gl)
					gl=gl.data.cpu()
					if (gl[0]>=0 and gl[0]<xw-0.5) and (gl[1]>=0 and gl[1]<yw-0.5):
						pB.append(gl)
						tree=0
					else:
						gl = oldgl
						tree=1	
				target_reached=steerTo(st, gl, idx, counter, grid1) #oblen)
				print('target_reached ====', target_reached)
			if target_reached==0:
				return 0
			else:
				for p1 in range(0,len(pA)):
					new_path.append(pA[p1])
				for p2 in range(len(pB)-1,-1,-1):
					new_path.append(pB[p2])

	return new_path	
	

def main(grid):  
	# size of obstacle block
	global size
	size=1

	# Load trained model for path generation
	global mlp
	mlp = MLP(32, 2) # simple @D
	mlp.load_state_dict(torch.load('models/mlp_100_4000_PReLU_ae_dd_final_pb0.pkl'))#newccfinaldata.pkl'))#_pba.pkl'))
	#mlp.load_state_dict(torch.load('models/mlp_100_4000_PReLU_ae_dd_finalnew100.pkl'))  
	if torch.cuda.is_available():
		mlp.cuda()

	agent: Agent = grid.agent
	goal: Goal = grid.goal

	obstaclespb: Obstacle =grid.obstacles

	global xw
	global yw
	xw = grid.size.width
	yw = grid.size.height

	# make empty map
	# mp = [[1 for i in range(yw)] for i in range(xw)]
	
	# zero array of 1 x n x 2 (each row is an obstacle's coordinates )
	global oblen
	oblen = 1400 
	
	#oblen = len(obstaclespb)

	print('len(obs)', len(obstaclespb))
	#print('obstacles', obstaclespb[5].position.x, obstaclespb[5].position.y)
		


	global obc
	obc=np.zeros((1,oblen,2),dtype=np.int32)

	count = 0

	for i in range(0,1):
		for ob in obstaclespb:
			# this if statement is for houseexpo maps 
			#if ob.position.x >=20 and ob.position.x <= 82 and ob.position.y >=5 and ob.position.y <= 95: 
			obc[0][count][0]=ob.position.x
			obc[0][count][1]=ob.position.y
			count+=1
				#print(ob.position.x, ob.position.y)
			# if count ==1400:
			# 	break
			#print('idx ----',idx)

	# array of obstacle coordinates
	print('obc --------',obc[0][:20])
	print("obc.size -----------", obc.size)
	
	number1 = obc.size
	global counter99
	counter99 = int(len(obstaclespb))
	counter = count
	#print('counter ------',counter)
	
	if counter != 1400:
		for i in range(0,1):
			while counter != 1400:
				#for idx,ob in enumerate(obstaclespb):
				obc[0][counter][0]=obc[0][0][0]
				obc[0][counter][1]=obc[0][0][1]
				counter+=1


	
	# obs_rep set up 			
	Q = Encoder()
	Q.load_state_dict(torch.load('./models/cae_encoderpb1.pkl'))#.pkl'))#pb1.pkl'))
	#Q.load_state_dict(torch.load('./models/cae_encoder_withitem.pkl'))
	if torch.cuda.is_available():
		Q.cuda()

	obs_rep=np.zeros((1,28),dtype=np.float32)	
	k=0

	for i in range(0,1):
		#temp=np.fromfile('./models/obc0.dat')	

		temp = obc

		# for i in range(0,1):
		#     for idx,ob in enumerate(obstaclespb):
		#         obc[0][idx][0]=ob.position.x
		#         obc[0][idx][1]=ob.position.y


		# nx2 matrix of all obstacles' coordinates (2800 elements)
		newtemp=np.zeros((1400,2),dtype=np.int32) #float32)
		temp=temp.reshape(temp.size//2,2)

		#print('temp88888',temp.shape)
		newtemp[0:oblen,0:] = temp
		#print("newtempppppppp",newtemp)



		obstacles=np.zeros((1,2800),dtype=np.int32)#float32)

		
		#print("obstacles",obstacles,"\nlen",obstacles.size)

		obstacles[0]=newtemp.flatten()
		inp=torch.from_numpy(obstacles).float()   ## added .float() at end  
		#print("obstacles",obstacles[0],"\nlen",obstacles.size)		

		#inp=Variable(inp).cuda()
		inp=Variable(inp).cpu()
		
		output=Q(inp)
		output=output.data.cpu()
		#print('output--------------',output.numpy())
		obs_rep[k]=output.numpy()
		#print("obs_rep",obs_rep[0],"\nlen",obs_rep.size)	
		k=k+1

	## calculating length of the longest trajectory
	# max_length=0
	# path_lengths=np.zeros((N,NP),dtype=np.int8)
	# for i in range(0,N):
	#     for j in range(0,NP):
	#         fname='./dataset2/e'+str(i+s)+'/path'+str(j+sp)+'.dat'
	#         if os.path.isfile(fname):
	#             path=np.fromfile(fname)
	#             path=path.reshape(len(path)//2,2)
	#             path_lengths[i][j]=len(path)	
	#             if len(path)> max_length:
	#                 max_length=len(path)
			

	# paths=np.zeros((N,NP,max_length,2), dtype=np.float32)   ## padded paths

	# for i in range(0,N):
	#     for j in range(0,NP):
	#         fname='./dataset2/e'+str(i+s)+'/path'+str(j+sp)+'.dat'
	#         if os.path.isfile(fname):
	#             path=np.fromfile(fname)
	#             path=path.reshape(len(path)//2,2)
	#             for k in range(0,len(path)):
	#                 paths[i][j][k]=path[k]

					
	#print("______________________________________________________________________")
	#obc is nx2 matrix of all obstacle coordinates
	#obs_rep
	#paths is coordinates provided in path file
	#path_length is number of coordinates (waypoint) of the path

	#print("obc ----", obc)
	#print('length of obc', len(obc))
	#print("obs_rep (latent space model) ----", obs_rep)
	#print("paths ----", paths)
	#print("path_lengths ----", path_lengths)


	i=0
	j=0
	tp=0
	fp=0
	tot=[]
	et=[]
	p1_ind=0
	p2_ind=0
	p_ind=0	
	#if path_lengths[i][j]>0:	
	# 							
	start=np.zeros((2),dtype=np.int32) #float32)
	goals=np.zeros((2),dtype=np.int32) #float32)
	
	# load start and goal position
	for l in range(0,1):
		start[l]= agent.position.x
		start[l+1]= agent.position.y			
	for l in range(0,1):
		goals[l]= goal.position.x
		goals[l+1]= goal.position.y	
	
	#print('start and goal', start,goals)

	#start and goal for bidirectional generation
	## starting point
	start1=torch.from_numpy(start)
	goal2=torch.from_numpy(start)
	##goal point
	goal1=torch.from_numpy(goals)
	start2=torch.from_numpy(goals)
	oldstart2=start2
	##obstacles
	obs = obs_rep[0]

	#print('start1 and start2',start1, start2)

	#print("______________________________________________________________________")

	#print("\n")
	#print("---- planning starts ")
	#print("\n")
		
	#print("______________________________________________________________________")


	#print ('obs torch', obs)
	obs=torch.from_numpy(obs)
	#print ('obs torch2', len(obs))
	##generated paths
	path1=[] 
	path1.append(start1)
	path2=[]
	path2.append(start2)
	path=[]
	target_reached=0
	step=0	
	path=[] # stores end2end path by concatenating path1 and path2
	tree=0	
	tic = time.clock()                #perf_counter 
	
	newstart1=None
	newstart2=None
				
	# start planning with network
	while target_reached==0 and step<300 :
		step=step+1
		if tree==0:
			#while start
			oldstart=start1
			#print('oldstart =',oldstart)
			inp1=torch.cat((obs,start1,start2)).float()
			#print('inpl---1',inp1)
			inp1=to_var(inp1)#.float()
			#print('inpl---2',inp1)
			start1=mlp(inp1)#.float()
			#print('start1---1',start1)
			start1=start1.data.cpu()
			print('start1---2',start1)
			#if start1 
			#print ('#############################################start1 and inp1',start1, inp1)
			#print ('plan from start: ',start1[0], start1[1]) 
			if (start1[0]>=0 and start1[0]<xw-0.5) and (start1[1]>=0 and start1[1]<yw-0.5):
			#if (start1[0]>=20 and start1[0]<82-0.5) and (start1[1]>=5 and start1[1]<95-0.5):
				#if (start1[0]>=22 and start1[0]<77-0.5) and (start1[1]>=49 and start1[1]<71-0.5):  	
				path1.append(start1)
				tree=1
				# else:
				# 	start1 = oldstart
				# 	tree=0
			else:
				start1 = oldstart
				tree=0
		else:
			#while True:
			oldstart2=start2
			inp2=torch.cat((obs,start2,start1)).float()
			#print('inp2---1',inp2)
			inp2=to_var(inp2)#.float()
			#print('inp2---1',inp2)
			start2=mlp(inp2)#.float()
			#print('start2---1',start2)
			start2=start2.data.cpu()
			print('start2---2',start2)
			#print ('plan from end: ',start2[0], start2[1]) 
			if (start2[0]>=0 and start2[0]<xw-0.5) and (start2[1]>=0 and start2[1]<yw-0.5):
			#if (start2[0]>=20 and start2[0]<82-0.5) and (start2[1]>=5 and start2[1]<95-0.5):
				#if (start2[0]>=22 and start2[0]<77-0.5) and (start2[1]>=49 and start2[1]<71-0.5):  	
				path2.append(start2)
				tree=0
				# else:
				# 	start2 = oldstart2
				# 	tree=1
			else:
				start2 = oldstart2
				tree=1
			
		#check if a straight line can be connected between the 2 points
		# steerTo is taking a long time 
		if (list(oldstart) != list(start1) and tree !=0) or (list(oldstart2) != list(start2) and tree !=1): 
			target_reached=steerTo(start1,start2,i, counter99, grid)
			#print('-----------------------target_reached-------------------------',target_reached)

	tp=tp+1

	if target_reached==1:
		for p1 in range(0,len(path1)):
			path.append(path1[p1])
		for p2 in range(len(path2)-1,-1,-1):
			path.append(path2[p2])	
		
		#print ('path before lvc = ', path)
		path=lvc(path,i,counter99, grid) #oblen)
		#print ('path after lvc = ', path)
		
		#print('*************feasibility_check start*************')
		indicator=feasibility_check(path,i, counter99, grid) #oblen)

		#print('---------------------------indictor = ', indicator)
		
		
		
		if indicator ==0:
			for count1 in range(len(path)):
				path[count1]= path[count1]
			#print('--------------------------path = ', path)

		if indicator ==1:
			for count1 in range(len(path)):
				path[count1] = [round(path[count1].tolist()[0]),round(path[count1].tolist()[1])]
				#print('path[count1]===', path[count1])
			#print('--------------------------path = ', path)        



		if indicator==1:
			#print ("_______________________________")
			print ("planning is done")
			# toc = time.clock() #time.perf_counter
			# t=toc-tic
			# et.append(t)
			# print ("path[0]:")
			# for p in range(0,len(path)):
			#     print (path[p][0])
			# print ("path[1]:")
			# for p in range(0,len(path)):
			#     print (path[p][1])
		else:
			sp=0
			indicator=0
			#print("______________________________________________________________________")

			#print("\n")
			#print("----- path is invalid. need to replan now ")
			#print("\n")
				
			#print("______________________________________________________________________")
			while indicator==0 and sp<10 and path !=0:
				sp=sp+1
				g=np.zeros(2,dtype=np.int32)
				#print('path[-1] ============', [path[-1][0],path[-1][1]])
				
				# g gives the last point in path
				g=torch.from_numpy(np.array([path[-1][0],path[-1][1]])) #paths[i][j][path_lengths[i][j]-1])
				
				#print('g --',g)
				
				#replan at coarse level -> while for map constraint 
				path=replan_path(path,g,i,obs, counter99,grid,linepath) #oblen) #replanning at coarse level
				#print('new path --', path)
				#print('indicator --', indicator)
				if path !=0:
					# condense new path and check if new path is feasible
					path=lvc(path,i, counter99, grid) #oblen)         
					indicator=feasibility_check(path,i, counter99,grid)
					#print('inside indicator --', indicator)
		
				if indicator==1:
					#print ("_______________________________")
					#print ("planning is done")
					for count1 in range(len(path)):
						path[count1] = [round(path[count1].tolist()[0]),round(path[count1].tolist()[1])]
					#print('path[count1]===', path[count1])
					#print('--------------------------path = ', path)  
					# toc = time.clock()
					# t=toc-tic
					# if len(path)<20:
					#     print ("new_path[0]:")
					#     for p in range(0,len(path)):
					#         print (path[p][0])
					#     print ("new_path[1]:")
					#     for p in range(0,len(path)):
					#         print (path[p][1])
					#     print ("path found, dont worry")					
		if indicator == 0:
			return path == None

	else:
		return path == None

	#print('final path', path)
	return path




class MPNetTest(Algorithm):
	step_grid: List[List[int]]
	color1: np.ndarray = (102, 0, 102)

	def __init__(self, services: Services, testing: BasicTesting = None):
		super().__init__(services, testing)
		self.step_grid = []
		self.waypoint = set()
		

	def set_display_info(self) -> List[MapDisplay]:
		"""
		Read super description
		"""
		display_info: List[MapDisplay] = [SolidColorMapDisplay(self._services, self.waypoint, self.color1)]+ super().set_display_info() #+ [
			#SolidColorMapDisplay(self._services, self.mem.visited, self.VISITED_COLOR, z_index=50)
			#SolidColorMapDisplay(self._services, self.waypoint, self.color1)
			#NumbersMapDisplay(self._services, copy.deepcopy(self.step_grid))
		#]
		return display_info   


	def _find_path_internal(self) -> None:
		"""
		Read super description
		The internal implementation of :ref:`find_path`
		"""
		#._get_grid() is in Algorithm class and gets the map
		grid: Map = self._get_grid()

		trace = main(grid)

		if trace == None or trace ==0:
			print ('No path found')
			pass
		else:
			# for point in trace:
			#     point.numpy()
			print("final path --------", trace)
			
			for point in trace[1:-1]:
				self.waypoint.add(Point(point[0]+0.25,point[1]-0.25))
				self.waypoint.add(Point(point[0]+0.25,point[1]+0.25))
				self.waypoint.add(Point(point[0]-0.25,point[1]+0.25))
				self.waypoint.add(Point(point[0]-0.25,point[1]-0.25))
			print ('waypoint =====', self.waypoint)
			
			if type(trace) != bool:
				print('______________________________________________________________')
				print('trace ======', trace)
				for point in trace:
					self.move_agent(Point(int(point[0]),int(point[1])))
					self.key_frame(ignore_key_frame_skip=True)            

		# print("trace", trace)

		# for ele in trace:
		#     print("******************************************8")
		#     print(list(ele))

		#trace = np.ndarray.tolist(trace)








