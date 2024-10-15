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

if __name__=="__main__":
    p_array=[]
    pp=np.array([0,0,0])
    vv=np.array([0,0,0])
    aa=np.array([0,0,0])
    p_array.append(Particle(1,pp,vv,aa,0))
    pp=np.array([1,0,0])
    p_array.append(Particle(1,pp,vv,aa,1))
    pp=np.array([0,1,0])
    p_array.append(Particle(1,pp,vv,aa,2))
    p_array=np.array(p_array)
    p_array[1].cal_gforce(p_array)
    print(p_array[1].acc)