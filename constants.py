#PLEASE USE THIS CODE FOR ALL YOUR PHYSICAL CONSTANTS, DO NOT TRY YO CALCULATE THEM BY HAND! YOU WILL DEFINATELY MESS THAT UP!
#THANKS FOR YOUR COOPERATION :)

"""
Code to store all the physical constant
The original Constants are in the SI units and then converted to desired units using pint library
The desired units are Kpc, Solar Mass, and yr
"""

import pint
ureg = pint.UnitRegistry()

#we have to define the solar mass for the sake of our sanity
ureg.define('solar_mass = 1.98847e30 * kilogram')

#calculating the gravitational constant in our desired units
G = 6.67*10**-11 * ureg.m**3/(ureg.kg*ureg.s**2)
G = G.to(ureg.kpc**3/(ureg.solar_masss*ureg.yr**2))
G = G.magnitude

# The Speed of Light
c = 3e8 * ureg.m/ureg.s
c = c.to(ureg.kpc/ureg.yr)
c = c.magnitude

#Boltzman constant (to be used for velocity distribution)
k_B = 1.380649e-23 *(ureg.J/ureg.K) # Boltzmann constant in J/K
k_B = k_B.to(ureg. solar_mass*ureg. kpc**2/( ureg.year**2*ureg.K)) #Boltzman constant in our desirable units
k_B = k_B.magnitude