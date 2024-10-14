#calculate gravitational force between particles
import numpy as np

const_G=4.3*10**(-3)
class Particle:

    def __init__(self,mass,position,vel,acc,seq):
        self.mass=mass
        self.position=position
        self.vel=vel
        self.acc=acc
        self.time=0
        self.seq=seq


    def update_time(self,new_time):
        self.time=new_time
    
    def cal_gforce(self,p_array):#array of all particles
        self.acc=np.zeros(3)
        for i in range (len(p_array)):
            if i!=self.seq:
                del_pos=np.array(p_array[i].position-self.position)
                rr=(del_pos[0]**2+del_pos[1]**2+del_pos[2]**2)**0.5
                self.acc+=p_array[i].mass*del_pos/rr**3
                print(rr)
        self.acc*=const_G
        return
