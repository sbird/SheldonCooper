#PLEASE USE THIS CODE FOR ALL YOUR PHYSICAL CONSTANT, DO NOT TRY YO CALCULATE THEM BY HAND! YOU WILL DEFINATELY MESS THAT UP!
#THANKS FOR YOUR COOPERATION :)
"""
Code to store all the physical constant
The original Constants are in SI units and then converted to desired units using pint library
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