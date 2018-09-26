# -*- coding: utf-8 -*-
"""
Tools for diffraction and FEMTO analysis


Created on Tue Apr 12 14:28:22 2016

@author: esposito_v
"""


import numpy as np
import diffractionAngles_modes as diff_mode




def hklFromAngles(E, delta, gamma, omega, alpha, U, B):
    """
    Calculate the hkl vector for a given set of angles. For horizontal geometry
    """
    
    wavelength = 12.3984/E;
    K = 2*np.pi/wavelength;    
    
    delta = np.deg2rad(delta)
    gamma = -np.deg2rad(gamma) #sign convention
    omega = np.deg2rad(omega)
    alpha = np.deg2rad(alpha)
    
    """rotation matrices"""
    Delta = np.array([[1, 0, 0],
                      [0, np.cos(delta), -np.sin(delta)],
                      [0, np.sin(delta), np.cos(delta)]])

    Gamma = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                      [np.sin(gamma), np.cos(gamma), 0],
                      [0, 0, 1]])
    
    Omega = np.array([[np.cos(omega), -np.sin(omega), 0],
                      [np.sin(omega), np.cos(omega), 0],
                      [0, 0, 1]])
    
    Alpha = np.array([[1, 0, 0],
                      [0, np.cos(alpha), -np.sin(alpha)],
                      [0, np.sin(alpha), np.cos(alpha)]])
    
    """ calculate H """
    UBH = np.dot(Gamma, Delta) - np.identity(3)
    UBH = np.dot(UBH, np.array([0,K,0]))
    UBH = np.dot(np.linalg.inv(Alpha), UBH)
    UBH = np.dot(np.linalg.inv(Omega), UBH)

    Uinv = np.linalg.inv(U)
    Binv = np.linalg.inv(B)
    
    H = np.dot(Uinv,UBH)
    H = np.dot(Binv, H)
    
    Q = np.linalg.norm(UBH)
    
    return H, Q




def hklToAngles(hkl, E, alpha, U, B, mode):
    """
    Calculates the diffractometer angles for a given crystal (UB mat) and hkl
    mode 1: horizontal, fixed incident angle
    """
    
    wavelength = 12.3984/E 
    K = 2*np.pi/wavelength
    
    Hw = U.dot(np.dot(B,hkl))
    
    """Test for the direct beam """
    Uinv = np.linalg.inv(U)
    Binv = np.linalg.inv(B)
    Y = np.dot(Uinv, np.array([0,1,0]))
    Y = np.dot(Binv, Y)
    
    print("The reciprocal lattice vector parallel to the x-ray beam at omega = 0 is: [%.4f %.4f %.4f]" % (Y[0], Y[1], Y[2]))
    
    if mode == 1:
        delta, gamma, omega, angout = diff_mode.diffAngles_mode1(Hw,K,alpha)
        
    tt = np.arccos( np.cos(gamma) * np.cos(delta) )

    delta=np.rad2deg(delta)
    gamma=np.rad2deg(gamma)
    omega=np.rad2deg(omega)
    tt = np.rad2deg(tt)
    angout=np.rad2deg(angout)
        
        
    return delta, gamma, omega, tt, angout




def UBmat(a, aa, N):
    """Find the length and angles of the real and reciprocal vectors"""
    a0,a1,a2 = vectorFromLengthAndAngles(a,aa)
    a,aa,b,ba = lengthAndAngles(a0,a1,a2)
    
    """Orthonormalize the reciprocal lattice vectors (B matrix)"""
    B,N = ortho_recip(N,a,aa,b,ba)
    
    """Rotation angle and normal with respect to z axis (U matrix)"""
    z = np.array([0,0,1])
    angle = np.arccos( np.dot(z,N) /np.linalg.norm(z)/np.linalg.norm(N))  #radian
    rotN = np.cross(N,z)/np.linalg.norm(np.cross(N,z)) #axis of rotation
    U = rotation_matrix(rotN, angle)
    
    return U, B
    
    
    
    
def vectorFromLengthAndAngles(a,aa):
    """
    Takes a set of lattice vectors lengths and angles and calculates the cartesian 
    vectors.
    a   : Length of lattice vectors.
    aa  : Angles between lattice vectors [Between a3 and a2, a3 and a1, a1
        and a2].
    b   : Length of reciprocal vectors.
    ba  : Angles between reciprocal vectors.
    a1,a2,a3    : Lattice vectors given in cubic coordinates as [x y z].
    """
    aa = np.deg2rad(aa)

    a0 = a[0]*  np.array([1,0,0])
    a1 = a[1]* np.array([np.cos(aa[2]), np.sin(aa[2]), 0 ])
    v = np.sqrt(1-np.cos(aa[0])**2 - np.cos(aa[1])**2 - np.cos(aa[2])**2
        + 2*np.cos(aa[0])*np.cos(aa[1])*np.cos(aa[2]) )
    a2 = a[2]* np.array([np.cos(aa[1]), (np.cos(aa[0])-np.cos(aa[1])*np.cos(aa[2]))/np.sin(aa[2]),
        v/np.sin(aa[2])] )

    return a0, a1, a2

    


def lengthAndAngles(a0,a1,a2):
    """
    Takes a set of lattice vectors, and calculates the
    length of and angles between both the vectors and their corresponding
    reciprocal vectors.
    
    a   : Length of lattice vectors.
    aa  : Angles between lattice vectors [Between a3 and a3, a3 and a1, a1
        and a2].
    b   : Length of reciprocal vectors.
    ba  : Angles between reciprocal vectors.
    a1,a2,a3    : Lattice vectors given in cubic coordinates as [x y z].
    """

    a = np.array([np.linalg.norm(a0), np.linalg.norm(a1), np.linalg.norm(a2) ])
    aa = np.array([np.arccos(np.dot(a1,a2)/a[1]/a[2]), 
                   np.arccos(np.dot(a2,a0)/a[2]/a[0]), 
                   np.arccos(np.dot(a0,a1)/a[0]/a[1])])
    
    nor = np.dot(a0,np.cross(a1,a2))
    b0 = 2*np.pi*np.cross(a1,a2)/nor
    b1 = 2*np.pi*np.cross(a2,a0)/nor
    b2 = 2*np.pi*np.cross(a0,a1)/nor
    
    b = np.array([np.linalg.norm(b0), np.linalg.norm(b1), np.linalg.norm(b2) ])
    ba = np.array([np.arccos(np.dot(b1,b2)/b[1]/b[2]),
                   np.arccos(np.dot(b2,b0)/b[2]/b[0]),
                   np.arccos(np.dot(b0,b1)/b[0]/b[1])])
    
    aa = np.rad2deg(aa)
    ba = np.rad2deg(ba)
                   
    return a, aa, b, ba
    
    


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    theta = np.asarray(theta)
    axis = axis/np.sqrt(np.dot(axis, axis))
    a = np.cos(theta/2.0)
    b, c, d = -axis*np.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])


                     

def ortho_recip(H,a,aa,b,ba):
    """
    Rewrite a reciprocal lattice vector in terms of a cartesian
    basic.
 
    [Hw,B] = cartesian(H,a,aa,b,ba);
    
    Hw       : [h;k;l], reciprocal letter in the orthogonal coordinate
            system.
    B       : The orthonomalization matrix.
    H      : [hw;kw;lw] reciprocal lattice vector in original coordiante
            system.
    a,aa,b,ba: Output of lengthAndAngles.m.
    """

    ba = np.deg2rad(ba)
    aa= np.deg2rad(aa)
    
    B = np.array([ [b[0], b[1]*np.cos(ba[2]), b[2]*np.cos(ba[1])],
                   [0, b[1]*np.sin(ba[2]), -b[2]*np.sin(ba[1])*np.cos(aa[0])],
                   [0, 0, 2*np.pi/a[2]] ])
    Hw = np.dot(B,H)
    
    return B, Hw
    
    
    
    
def anglesFromPos(x,y,xr,yr,d):
    """
    Calculate the angles delta and gamma (elevation and azimuth) for a position
    (x,y), with respect to a reference (xr,yr) and a source distance d.
    
    Typically used to calculate diffraction angles based on the position on
    a detector
    """
    
    dx = x-xr
    dy = y-yr
        
    gamma = np.rad2deg( np.arctan(dx/d) )
    delta = np.rad2deg( np.arctan(dy/(d**2 + dx**2)**0.5) )
    
    return delta, gamma
    
    
    
        


























