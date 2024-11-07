#'Bi-Tree
import numpy as np
import copy
import time
const_G=1#4.3*10**(-3)

class Node:
    def __init__(self,mass,mposition,lowerbound,upperbound):
        self.mass=mass
        self.mposition=copy.deepcopy(mposition)
        self.lowerbound=copy.deepcopy(lowerbound)
        self.upperbound=copy.deepcopy(upperbound)
        self.depth=0
        self.index=-1
        self.left_node=None
        self.right_node=None
        self.last_node=None

    
class Particle:
    def __init__(self,mass,position,vel,acc,seq):
        self.mass=mass
        self.position=copy.deepcopy(position)
        self.vel=vel
        self.acc=acc
        self.seq=seq
        self.parent=None

    def update_time(self,new_time):
        self.time=new_time
    
    def cal_gforce(self,p_array):
        self.acc=np.zeros(3)
        for i in range (len(p_array)):
            if i!=self.seq:
                del_pos=np.array(p_array[i].position-self.position)
                rr=(del_pos[0]**2+del_pos[1]**2+del_pos[2]**2)**0.5
                self.acc+=p_array[i].mass*del_pos/rr**3
        self.acc*=const_G
        return
    
def insert_Node_tree(par,list_par,Node_tree):
    #print(par.seq,Node_tree.depth,par.position,Node_tree.lowerbound,Node_tree.upperbound)
    remainder_dep=Node_tree.depth%3
    Node_tree.mass+=par.mass
    Node_tree.mposition+=par.mass*par.position
    if Node_tree.left_node is None and Node_tree.right_node is None:
        if Node_tree.index==-1:
            #Node_tree.mass+=par.mass
            #Node_tree.mposition+=par.mass*par.position
            Node_tree.index=par.seq
            par.parent=Node_tree
        else:
            node1=Node(0,np.zeros(3),Node_tree.lowerbound,Node_tree.upperbound)
            node2=Node(0,np.zeros(3),Node_tree.lowerbound,Node_tree.upperbound)
            cut=(int(Node_tree.upperbound[remainder_dep])+int(Node_tree.lowerbound[remainder_dep]))/2#;print(Node_tree.upperbound[remainder_dep],Node_tree.lowerbound[remainder_dep],remainder_dep,'cut=',cut)
            node1.upperbound[remainder_dep]=cut
            node2.lowerbound[remainder_dep]=cut
            node1.last_node=Node_tree
            node2.last_node=Node_tree
            node1.depth=Node_tree.depth+1
            node2.depth=Node_tree.depth+1
            Node_tree.left_node=node1
            Node_tree.right_node=node2
            if list_par[Node_tree.index].position[remainder_dep]<=cut:
                insert_Node_tree(list_par[Node_tree.index],list_par,Node_tree.left_node)
            else:
                insert_Node_tree(list_par[Node_tree.index],list_par,Node_tree.right_node)
            if par.position[remainder_dep]<=cut:
                insert_Node_tree(par,list_par,Node_tree.left_node)
            else:
                insert_Node_tree(par,list_par,Node_tree.right_node)
            Node_tree.index=-1
    else:
        cut=(int(Node_tree.upperbound[remainder_dep])+int(Node_tree.lowerbound[remainder_dep]))/2#;print(Node_tree.upperbound[remainder_dep],Node_tree.lowerbound[remainder_dep],remainder_dep,'cut=',cut)
        if par.position[remainder_dep]<=cut:
            insert_Node_tree(par,list_par,Node_tree.left_node)
        else:
            insert_Node_tree(par,list_par,Node_tree.right_node) 
    return

def print_Node_tree(Node_tree,list_par):
    if Node_tree.left_node is None and Node_tree.right_node is None:
        if Node_tree.index!=-1:
            print("particle",list_par[Node_tree.index].position,"lowerB",Node_tree.lowerbound,"upperB",Node_tree.upperbound,"depth",Node_tree.depth)
    else:
        print_Node_tree(Node_tree.left_node,list_par)
        print_Node_tree(Node_tree.right_node,list_par)
    return


def move_particle(par):
    par.parent.mass-=par.mass
    par.parent.mposition-=par.mass*par.position
    par.parent.index=-1
    if par.parent.lowerbound[0] < par.position[0] <=par.parent.upperbound[0] \
        and par.parent.lowerbound[1] < par.position[1] <=par.parent.upperbound[1] \
        and par.parent.lowerbound[2] < par.position[2] <=par.parent.upperbound[2]:
        insert_Node_tree(par,list_par,par.parent)
    else:
        par.parent=par.parent.last_node
        move_particle(par)
    return

def clip_Node_tree(Node_tree):
    if Node_tree.left_node is None and Node_tree.right_node is None:
        if Node_tree.index==-1:
            Node_tree.last_node=None
    else:
        clip_Node_tree(Node_tree.left_node)
        clip_Node_tree(Node_tree.right_node)
    return

def cal_force(par,Node_tree):
    if Node_tree is None or Node_tree.mass==0 or Node_tree.index==par.seq:
        return 0
    del_pos=Node_tree.mposition/Node_tree.mass-par.position
    rr=np.sum(del_pos**2)**0.5
    if np.sum((Node_tree.upperbound-Node_tree.lowerbound)**2)**0.5/rr<1 or Node_tree.index!=-1:
        #print(Node_tree.depth,',',const_G*Node_tree.mass*del_pos/rr**3)
        return const_G*Node_tree.mass*del_pos/rr**3
    else:
        return cal_force(par,Node_tree.left_node)+cal_force(par,Node_tree.right_node)
    

box_range=[np.full((3),0),np.full((3),2**5)]
list_par=[]
root=Node(0,np.zeros(3),box_range[0],box_range[1])
num=1000
for i in range(num):
    pp=Particle(1,np.array([np.random.randint(0,2**5),np.random.randint(0,2**5),np.random.randint(0,2**5)]),np.zeros(3),np.zeros(3),i)
    list_par.append(pp)
    #print(pp.position)
for i in range(num):
    insert_Node_tree(list_par[i],list_par,root)
    #print_Node_tree(root,list_par)
    #print('---')


stime=time.time()
for i in range(num):
    #print(cal_force(list_par[i],root))
    cal_force(list_par[i],root)
etime=time.time()
print("time="+"%.5f"%(etime-stime)+" s")
print('\n')

stime=time.time()
for i in range(num):
    list_par[i].cal_gforce(list_par)
    #print(list_par[i].acc)
etime=time.time()
print("time="+"%.5f"%(etime-stime)+" s")
print('\n')