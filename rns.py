#!/usr/bin/env python2
  
import matplotlib.pyplot as plt
import scipy.integrate   as integrate
import numpy             as np

class rns():

    def __init__(self):
        self.year2sec = 365.25 * 86400.0
        return

    def parameters(self,model,laws='Aging'):
        """
        ----------------------------------------------------------------------------------------------
        @Summary: Set the parameters for solving the rate-and-state equations 
        @Author : Cedric Twardzik
        @Date   : August 9, 2019
        @Note   : [0]:Dc(m), [1]:asigm(MPa), [2]:bsigm(MPa), [3]:k(MPa/m), [4]:V0(m/yr), [5]:dtau(MPa)
        ----------------------------------------------------------------------------------------------
        """
        self.dc    = model[0]
        self.asigm = model[1] * 1.0E+6
        self.bsigm = model[2] * 1.0E+6
        self.k     = model[3] * 1.0E+6
        self.V0    = model[4] / self.year2sec
        self.dtau  = model[5] * 1.0E+6
        self.laws  = laws
        return

    def initial_conditions(self):
        """
        ----------------------------------------------------------------------------------------------------------
        @Summary: Set the initial conditions assuming steady-state before the perturbation (i.e., dtheta/dt = 0.0)
        @Author : Cedric Twardzik
        @Date   : August 9, 2019
        ----------------------------------------------------------------------------------------------------------
        """
        self.initTheta = self.dc / self.V0
        self.initV     = self.V0 * np.exp(self.dtau/self.asigm)
        return

    def rate_and_state(self,y,t):
        """
        -----------------------------------------------------------------------
        @Summary: Set of differential equations of the rate and state framework
        @Author : Cedric Twardzik
        @Date   : August 9, 2019
        -----------------------------------------------------------------------
        """
        dydt    = [0.0,0.0]
        if self.laws == 'Aging': dydt[0] = 1.0 - y[1]*y[0]/self.dc
        if self.laws == 'Slip' : dydt[0] = -y[1]*y[0]/self.dc * np.log(y[1]*y[0]/self.dc)
        dydt[1] = y[1]/self.asigm * (self.k*(self.V0-y[1]) - self.bsigm/y[0]*dydt[0])
        return dydt

    def solver(self,time):
        """
        ------------------------------------------------
        @Summary: Solve for the rate and state equations 
        @Author : Cedric Twardzik
        @Date   : August 9, 2019
        ------------------------------------------------
        """
        self.sol = integrate.odeint(self.rate_and_state,[self.initTheta,self.initV],time)
        return
        
if __name__ == "__main__":

    # ---------------------------------------------------------------------------- #
    # Example using the values from Fukuda et al. (2009), doi:10.1029/2008JB006166 #
    # ---------------------------------------------------------------------------- #
    model = [0.001,0.400,0.170,1.000,0.080,2.000]
    time  = np.linspace(0.0,18000.0,601)

    plt.figure(figsize=(15,4))

    # ------------------------- #
    # Solve using the Aging law #
    # ------------------------- #
    fric  = rns()
    fric.parameters(model,laws='Aging')
    fric.initial_conditions()
    fric.solver(time)
    plt.subplot(131)
    plt.plot(time/3600.0,fric.sol[:,1]*3600.0*100.0,label='Aging law')
    plt.xlim(0.0,5.0)
    plt.ylim(0.0,6.0)
    plt.xlabel('Time (hours)')
    plt.ylabel('Slip velocity (cm/hour)')
    plt.title ('Slip velocity')
    plt.subplot(132)
    plt.plot(time/3600.0,np.cumsum(fric.sol[:,1])*(time[1]-time[0])*100.0)
    plt.xlim(0.0, 5.0)
    plt.ylim(0.0,15.0)
    plt.xlabel('Time (hours)')
    plt.ylabel('Slip (cm)')
    plt.title ('Cumulative slip')
    plt.subplot(133)
    plt.plot(time/3600.0,fric.sol[:,0]/3600.0)
    plt.xlim(0.0, 5.0)
    plt.xlabel('Time (hours)')
    plt.ylabel('$\Theta$ (hours)')
    plt.title ('State variable')

    # ------------------------ #
    # Solve using the Slip law #
    # ------------------------ #
    fric  = rns()
    fric.parameters(model,laws='Slip')
    fric.initial_conditions()
    fric.solver(time)
    plt.subplot(131)
    plt.plot(time/3600.0,fric.sol[:,1]*3600.0*100.0,label='Slip law')
    plt.xlim(0.0,5.0)
    plt.ylim(0.0,6.0)
    plt.xlabel('Time (hours)')
    plt.ylabel('Slip velocity (cm/hour)')
    plt.title ('Slip velocity')
    plt.legend()
    plt.subplot(132)
    plt.plot(time/3600.0,np.cumsum(fric.sol[:,1])*(time[1]-time[0])*100.0)
    plt.xlim(0.0, 5.0)
    plt.ylim(0.0,15.0)
    plt.xlabel('Time (hours)')
    plt.ylabel('Slip (cm)')
    plt.title ('Cumulative slip')
    plt.subplot(133)
    plt.plot(time/3600.0,fric.sol[:,0]/3600.0)
    plt.xlim(0.0, 5.0)
    plt.xlabel('Time (hours)')
    plt.ylabel('$\Theta$ (hours)')
    plt.title ('State variable')

    plt.show()
    
