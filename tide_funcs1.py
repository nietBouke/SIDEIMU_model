# =============================================================================
# functions for tidal model
# analytical solutions to the tidal salt balance with w included. 
# =============================================================================
import numpy as np

def sti(z,c,pars,w_on = 1): 
    # =============================================================================
    # calculate tidal salinity (complex notation, so \hat s_ti)
    # newer version which has a better performance but should have the same result as the old formulation
    # =============================================================================
    c1,c2,c3,c4 = c
    c3,c4 = c3*w_on , c4*w_on
    dA,H,B,n0 = pars
    
    #calculate some returning parts for better perfomance
    zn = z/H
    c1_wo = np.sqrt(c1) 
    
    sbar = (-c2*(-B*H**2*np.cosh(dA*zn) - B*H*c1_wo*dA*np.cos(z/c1_wo)*np.sinh(dA)*1/np.sin(H/c1_wo) + H**2 + c1*dA**2)/(H**2 + c1*dA**2))[0]

    def spri(n):
        sin_pinz = np.sin(np.pi*n*zn)
        cos_pin = np.cos(np.pi*n)
        cos_zcw = np.cos(z/c1_wo)
        sin_Hcw = np.sin(H/c1_wo)
        
        return c3 * (H*(-H*sin_pinz  - np.pi*c1_wo*n*(-cos_pin + np.cos(H/c1_wo))*cos_zcw*1/sin_Hcw + np.pi*c1_wo*n*np.sin(z/c1_wo))/(H**2 - np.pi**2*c1*n**2)) \
                + c4 * (H**2*(-B*H**7*sin_pinz *np.sinh(dA*zn) - np.pi*B*H**6*c1_wo*n*cos_pin*cos_zcw*np.sinh(dA)*1/sin_Hcw - B*H**5*c1*dA**2*sin_pinz *np.sinh(dA*zn) + 3*np.pi**2*B*H**5*c1*n**2*sin_pinz *np.sinh(dA*zn) \
                     + np.pi*B*H**4*c1**(3/2)*dA**2*n*cos_pin*cos_zcw*np.sinh(dA)*1/sin_Hcw + 3*np.pi**3*B*H**4*c1**(3/2)*n**3*cos_pin*cos_zcw*np.sinh(dA)*1/sin_Hcw + 2*np.pi**2*B*H**3*c1**2*dA**2*n**2*sin_pinz *np.sinh(dA*zn) \
                     - 3*np.pi**4*B*H**3*c1**2*n**4*sin_pinz *np.sinh(dA*zn) - 2*np.pi**3*B*H**2*c1**(5/2)*dA**2*n**3*cos_pin*cos_zcw*np.sinh(dA)*1/sin_Hcw - 3*np.pi**5*B*H**2*c1**(5/2)*n**5*cos_pin*cos_zcw*np.sinh(dA)*1/sin_Hcw \
                     - np.pi**4*B*H*c1**3*dA**2*n**4*sin_pinz *np.sinh(dA*zn) + np.pi**6*B*H*c1**3*n**6*sin_pinz *np.sinh(dA*zn) + 2*np.pi*B*H*c1*dA*n*(H**2 - np.pi**2*c1*n**2)**2*np.cos(np.pi*n*zn)*np.cosh(dA*zn) \
                     + np.pi**5*B*c1**(7/2)*dA**2*n**5*cos_pin*cos_zcw*np.sinh(dA)*1/sin_Hcw + np.pi**7*B*c1**(7/2)*n**7*cos_pin*cos_zcw*np.sinh(dA)*1/sin_Hcw \
                     - B*c1_wo*dA*(H**2 - np.pi**2*c1*n**2)**2*(H**2 + c1*dA**2 + np.pi**2*c1*n**2)*np.sin(np.pi*n)*cos_zcw*np.cosh(dA)*1/sin_Hcw + np.pi*H**6*c1_wo*dA*n*cos_pin*cos_zcw*1/sin_Hcw \
                     + H**6*c1_wo*dA*np.sin(np.pi*n)*cos_zcw*1/sin_Hcw + H**6*dA*z*sin_pinz  - 2*np.pi*H**5*c1*dA*n*np.cos(np.pi*n*zn) + 2*np.pi*H**4*c1**(3/2)*dA**3*n*cos_pin*cos_zcw*1/sin_Hcw \
                     + 2*H**4*c1**(3/2)*dA**3*np.sin(np.pi*n)*cos_zcw*1/sin_Hcw - 3*np.pi**3*H**4*c1**(3/2)*dA*n**3*cos_pin*cos_zcw*1/sin_Hcw - np.pi**2*H**4*c1**(3/2)*dA*n**2*np.sin(np.pi*n)*cos_zcw*1/sin_Hcw \
                     + 2*H**4*c1*dA**3*z*sin_pinz  - 3*np.pi**2*H**4*c1*dA*n**2*z*sin_pinz  - 4*np.pi*H**3*c1**2*dA**3*n*np.cos(np.pi*n*zn) + 4*np.pi**3*H**3*c1**2*dA*n**3*np.cos(np.pi*n*zn) + np.pi*H**2*c1**(5/2)*dA**5*n*cos_pin*cos_zcw*1/sin_Hcw \
                     + H**2*c1**(5/2)*dA**5*np.sin(np.pi*n)*cos_zcw*1/sin_Hcw + 4*np.pi**2*H**2*c1**(5/2)*dA**3*n**2*np.sin(np.pi*n)*cos_zcw*1/sin_Hcw + 3*np.pi**5*H**2*c1**(5/2)*dA*n**5*cos_pin*cos_zcw*1/sin_Hcw\
                     - np.pi**4*H**2*c1**(5/2)*dA*n**4*np.sin(np.pi*n)*cos_zcw*1/sin_Hcw + H**2*c1**2*dA**5*z*sin_pinz  + 3*np.pi**4*H**2*c1**2*dA*n**4*z*sin_pinz  - 2*np.pi*H*c1**3*dA**5*n*np.cos(np.pi*n*zn) - 4*np.pi**3*H*c1**3*dA**3*n**3*np.cos(np.pi*n*zn) \
                     - 2*np.pi**5*H*c1**3*dA*n**5*np.cos(np.pi*n*zn) - np.pi**3*c1**(7/2)*dA**5*n**3*cos_pin*cos_zcw*1/sin_Hcw + np.pi**2*c1**(7/2)*dA**5*n**2*np.sin(np.pi*n)*cos_zcw*1/sin_Hcw \
                     - 2*np.pi**5*c1**(7/2)*dA**3*n**5*cos_pin*cos_zcw*1/sin_Hcw + 2*np.pi**4*c1**(7/2)*dA**3*n**4*np.sin(np.pi*n)*cos_zcw*1/sin_Hcw - np.pi**7*c1**(7/2)*dA*n**7*cos_pin*cos_zcw*1/sin_Hcw \
                     + np.pi**6*c1**(7/2)*dA*n**6*np.sin(np.pi*n)*cos_zcw*1/sin_Hcw - np.pi**2*c1**3*dA**5*n**2*z*sin_pinz  - 2*np.pi**4*c1**3*dA**3*n**4*z*sin_pinz  \
                     - np.pi**6*c1**3*dA*n**6*z*sin_pinz )/(dA*(H**2 - np.pi**2*c1*n**2)**2*(H**4 + 2*H**2*c1*(dA**2 - np.pi**2*n**2) + c1**2*(dA**2 + np.pi**2*n**2)**2)))

    out = sbar+spri(n0).sum(0)

    return out


#derivatives for chain rule

def dsti_dc2(z,c1,pars,w_on = 1): 
    # =============================================================================
    #     derivative of s to c2
    # =============================================================================
    dA,H,B,n0 = pars
    
    dsb_dc2 = (-(-B*H**2*np.cosh(dA*z/H) - B*H*np.sqrt(c1)*dA*np.cos(z/np.sqrt(c1))*np.sinh(dA)*1/np.sin(H/np.sqrt(c1)) + H**2 + c1*dA**2)/(H**2 + c1*dA**2))
    dsp_dc2 = 0    
    ds_dc2 = dsb_dc2+dsp_dc2

    return ds_dc2


def dsti_dc3(z,c1,pars,w_on = 1): 
    # =============================================================================
    #     derivative of s to c3
    # =============================================================================
    dA,H,B,n0 = pars
    
    dsb_dc3 = 0
    dsp_dc3 = lambda n: H*(-H*np.sin(np.pi*n*z/H) - np.pi*np.sqrt(c1)*n*(-np.cos(np.pi*n) + np.cos(H/np.sqrt(c1)))*np.cos(z/np.sqrt(c1))*1/np.sin(H/np.sqrt(c1)) + np.pi*np.sqrt(c1)*n*np.sin(z/np.sqrt(c1)))/(H**2 - np.pi**2*c1*n**2)
    
    ds_dc3 = dsb_dc3 + dsp_dc3(n0)

    return ds_dc3



def dsti_dc4(z,c1,pars,w_on = 1): 
    # =============================================================================
    #     derivative of s to c4
    # =============================================================================
    dA,H,B,n0 = pars
    
    #calculate some returning parts for better perfomance
    zn = z/H
    c1_wo = np.sqrt(c1)
    
    dsb_dc4 = 0
    
    def dsp_dc4(n):
        sin_pinz = np.sin(np.pi*n*zn)
        cos_pin = np.cos(np.pi*n)
        cos_zcw = np.cos(z/c1_wo)
        sin_Hcw = np.sin(H/c1_wo)
        
        return (H**2*(-B*H**7*sin_pinz *np.sinh(dA*zn) - np.pi*B*H**6*c1_wo*n*cos_pin*cos_zcw*np.sinh(dA)*1/sin_Hcw - B*H**5*c1*dA**2*sin_pinz *np.sinh(dA*zn) + 3*np.pi**2*B*H**5*c1*n**2*sin_pinz *np.sinh(dA*zn) \
                + np.pi*B*H**4*c1**(3/2)*dA**2*n*cos_pin*cos_zcw*np.sinh(dA)*1/sin_Hcw + 3*np.pi**3*B*H**4*c1**(3/2)*n**3*cos_pin*cos_zcw*np.sinh(dA)*1/sin_Hcw + 2*np.pi**2*B*H**3*c1**2*dA**2*n**2*sin_pinz *np.sinh(dA*zn) \
                - 3*np.pi**4*B*H**3*c1**2*n**4*sin_pinz *np.sinh(dA*zn) - 2*np.pi**3*B*H**2*c1**(5/2)*dA**2*n**3*cos_pin*cos_zcw*np.sinh(dA)*1/sin_Hcw - 3*np.pi**5*B*H**2*c1**(5/2)*n**5*cos_pin*cos_zcw*np.sinh(dA)*1/sin_Hcw \
                - np.pi**4*B*H*c1**3*dA**2*n**4*sin_pinz *np.sinh(dA*zn) + np.pi**6*B*H*c1**3*n**6*sin_pinz *np.sinh(dA*zn) + 2*np.pi*B*H*c1*dA*n*(H**2 - np.pi**2*c1*n**2)**2*np.cos(np.pi*n*zn)*np.cosh(dA*zn) \
                + np.pi**5*B*c1**(7/2)*dA**2*n**5*cos_pin*cos_zcw*np.sinh(dA)*1/sin_Hcw + np.pi**7*B*c1**(7/2)*n**7*cos_pin*cos_zcw*np.sinh(dA)*1/sin_Hcw \
                - B*c1_wo*dA*(H**2 - np.pi**2*c1*n**2)**2*(H**2 + c1*dA**2 + np.pi**2*c1*n**2)*np.sin(np.pi*n)*cos_zcw*np.cosh(dA)*1/sin_Hcw + np.pi*H**6*c1_wo*dA*n*cos_pin*cos_zcw*1/sin_Hcw \
                + H**6*c1_wo*dA*np.sin(np.pi*n)*cos_zcw*1/sin_Hcw + H**6*dA*z*sin_pinz  - 2*np.pi*H**5*c1*dA*n*np.cos(np.pi*n*zn) + 2*np.pi*H**4*c1**(3/2)*dA**3*n*cos_pin*cos_zcw*1/sin_Hcw \
                + 2*H**4*c1**(3/2)*dA**3*np.sin(np.pi*n)*cos_zcw*1/sin_Hcw - 3*np.pi**3*H**4*c1**(3/2)*dA*n**3*cos_pin*cos_zcw*1/sin_Hcw - np.pi**2*H**4*c1**(3/2)*dA*n**2*np.sin(np.pi*n)*cos_zcw*1/sin_Hcw \
                + 2*H**4*c1*dA**3*z*sin_pinz  - 3*np.pi**2*H**4*c1*dA*n**2*z*sin_pinz  - 4*np.pi*H**3*c1**2*dA**3*n*np.cos(np.pi*n*zn) + 4*np.pi**3*H**3*c1**2*dA*n**3*np.cos(np.pi*n*zn) + np.pi*H**2*c1**(5/2)*dA**5*n*cos_pin*cos_zcw*1/sin_Hcw \
                + H**2*c1**(5/2)*dA**5*np.sin(np.pi*n)*cos_zcw*1/sin_Hcw + 4*np.pi**2*H**2*c1**(5/2)*dA**3*n**2*np.sin(np.pi*n)*cos_zcw*1/sin_Hcw + 3*np.pi**5*H**2*c1**(5/2)*dA*n**5*cos_pin*cos_zcw*1/sin_Hcw\
                - np.pi**4*H**2*c1**(5/2)*dA*n**4*np.sin(np.pi*n)*cos_zcw*1/sin_Hcw + H**2*c1**2*dA**5*z*sin_pinz  + 3*np.pi**4*H**2*c1**2*dA*n**4*z*sin_pinz  - 2*np.pi*H*c1**3*dA**5*n*np.cos(np.pi*n*zn) - 4*np.pi**3*H*c1**3*dA**3*n**3*np.cos(np.pi*n*zn) \
                - 2*np.pi**5*H*c1**3*dA*n**5*np.cos(np.pi*n*zn) - np.pi**3*c1**(7/2)*dA**5*n**3*cos_pin*cos_zcw*1/sin_Hcw + np.pi**2*c1**(7/2)*dA**5*n**2*np.sin(np.pi*n)*cos_zcw*1/sin_Hcw \
                - 2*np.pi**5*c1**(7/2)*dA**3*n**5*cos_pin*cos_zcw*1/sin_Hcw + 2*np.pi**4*c1**(7/2)*dA**3*n**4*np.sin(np.pi*n)*cos_zcw*1/sin_Hcw - np.pi**7*c1**(7/2)*dA*n**7*cos_pin*cos_zcw*1/sin_Hcw \
                + np.pi**6*c1**(7/2)*dA*n**6*np.sin(np.pi*n)*cos_zcw*1/sin_Hcw - np.pi**2*c1**3*dA**5*n**2*z*sin_pinz  - 2*np.pi**4*c1**3*dA**3*n**4*z*sin_pinz  \
                - np.pi**6*c1**3*dA*n**6*z*sin_pinz )/(dA*(H**2 - np.pi**2*c1*n**2)**2*(H**4 + 2*H**2*c1*(dA**2 - np.pi**2*n**2) + c1**2*(dA**2 + np.pi**2*n**2)**2)))

    ds_dc4 = dsb_dc4 + dsp_dc4(n0)
    return ds_dc4



def dsti_dz(z,c,pars,w_on = 1): 
    # =============================================================================
    # calculate vertical derivative of tidal salinity (complex notation, so \hat s_ti)
    # newer version which has a better performance but should have the same result as the old formulation
    # =============================================================================
    c1,c2,c3,c4 = c
    c3,c4 = c3*w_on , c4*w_on
    dA,H,B,n0 = pars
    
    #calculate some returning parts for better perfomance
    zn = z/H
    c1_wo = np.sqrt(c1) 
    sin_Hcw = np.sin(H/c1_wo)

    sbar = (c2*B*H*dA*(-np.sin(z/c1_wo)*np.sinh(dA)*1/sin_Hcw + np.sinh(dA*zn))/(H**2 + c1*dA**2))[0]             
     
    def spri(n):
        sin_pinz = np.sin(np.pi*n*zn)
        cos_pin = np.cos(np.pi*n)
        
        return c3 * (np.pi*H*n*((cos_pin*1/sin_Hcw - 1/np.tan(H/c1_wo))*np.sin(z/c1_wo) - np.cos(z/c1_wo) + np.cos(np.pi*n*zn))/(-H**2 + np.pi**2*c1*n**2)) \
             + c4 * (-H*(H*c1**(3/2)*(np.pi*B*n*(3*np.pi**2*H**4*n**2 - 3*np.pi**4*H**2*c1*n**4 + np.pi**6*c1**2*n**6 + dA**2*(H**2 - np.pi**2*c1*n**2)**2)*cos_pin*np.sinh(dA) - np.pi*dA*n*(H**4*(-2*dA**2 + 3*np.pi**2*n**2) - H**2*c1*(dA**4 + 3*np.pi**4*n**4) \
             + np.pi**2*c1**2*n**2*(dA**2 + np.pi**2*n**2)**2)*cos_pin + dA*(H**4*(2*dA**2 - np.pi**2*n**2) + H**2*c1*(dA**4 + 4*np.pi**2*dA**2*n**2 - np.pi**4*n**4) + np.pi**2*c1**2*n**2*(dA**2 + np.pi**2*n**2)**2)*np.sin(np.pi*n))*np.sin(z/c1_wo)*1/sin_Hcw \
             + c1_wo*(-np.pi*B*H**7*n*np.sin(z/c1_wo)*cos_pin*np.sinh(dA)*1/sin_Hcw + np.pi*B*H**7*n*np.cos(np.pi*n*zn)*np.sinh(dA*zn) - np.pi*B*H**5*c1*dA**2*n*np.cos(np.pi*n*zn)*np.sinh(dA*zn) - 3*np.pi**3*B*H**5*c1*n**3*np.cos(np.pi*n*zn)*np.sinh(dA*zn) \
             + 2*np.pi**3*B*H**3*c1**2*dA**2*n**3*np.cos(np.pi*n*zn)*np.sinh(dA*zn) + 3*np.pi**5*B*H**3*c1**2*n**5*np.cos(np.pi*n*zn)*np.sinh(dA*zn) - np.pi**5*B*H*c1**3*dA**2*n**5*np.cos(np.pi*n*zn)*np.sinh(dA*zn) \
             - np.pi**7*B*H*c1**3*n**7*np.cos(np.pi*n*zn)*np.sinh(dA*zn) + B*H*dA*(H**2 + c1*(dA**2 + np.pi**2*n**2))*(H**2 - np.pi**2*c1*n**2)**2*sin_pinz*np.cosh(dA*zn) - B*H*dA*(H**2 - np.pi**2*c1*n**2)**2*(H**2 + c1*dA**2 \
             + np.pi**2*c1*n**2)*np.sin(np.pi*n)*np.sin(z/c1_wo)*np.cosh(dA)*1/sin_Hcw + np.pi*H**7*dA*n*np.sin(z/c1_wo)*cos_pin*1/sin_Hcw + H**7*dA*np.sin(np.pi*n)*np.sin(z/c1_wo)*1/sin_Hcw - H**7*dA*sin_pinz \
             - np.pi*H**6*dA*n*z*np.cos(np.pi*n*zn) - 2*H**5*c1*dA**3*sin_pinz + np.pi**2*H**5*c1*dA*n**2*sin_pinz - 2*np.pi*H**4*c1*dA**3*n*z*np.cos(np.pi*n*zn) + 3*np.pi**3*H**4*c1*dA*n**3*z*np.cos(np.pi*n*zn) - H**3*c1**2*dA**5*sin_pinz \
             - 4*np.pi**2*H**3*c1**2*dA**3*n**2*sin_pinz + np.pi**4*H**3*c1**2*dA*n**4*sin_pinz - np.pi*H**2*c1**2*dA**5*n*z*np.cos(np.pi*n*zn) - 3*np.pi**5*H**2*c1**2*dA*n**5*z*np.cos(np.pi*n*zn) - np.pi**2*H*c1**3*dA**5*n**2*sin_pinz \
             - 2*np.pi**4*H*c1**3*dA**3*n**4*sin_pinz - np.pi**6*H*c1**3*dA*n**6*sin_pinz + np.pi**3*c1**3*dA**5*n**3*z*np.cos(np.pi*n*zn) + 2*np.pi**5*c1**3*dA**3*n**5*z*np.cos(np.pi*n*zn) \
             + np.pi**7*c1**3*dA*n**7*z*np.cos(np.pi*n*zn)))/(c1_wo*dA*(H**2 - np.pi**2*c1*n**2)**2*(H**4 + 2*H**2*c1*(dA**2 - np.pi**2*n**2) + c1**2*(dA**2 + np.pi**2*n**2)**2)))

    out = sbar+spri(n0).sum(0)

    return out






def dsti_dz_dc2(z,c1,pars,w_on = 1): #derivative of dsti_dz
    # =============================================================================
    #     derivative of dsti_dz to c2
    # =============================================================================
    dA,H,B,n0 = pars

    dstb_dz_dc2 = -B*H*dA*(np.sin(z/np.sqrt(c1))*np.sinh(dA)*1/np.sin(H/np.sqrt(c1)) - np.sinh(dA*z/H))/(H**2 + c1*dA**2)
    dstp_dz_dc2 = 0
    
    dst_dz_dc2 = dstb_dz_dc2 + dstp_dz_dc2
    
    return dst_dz_dc2


def dsti_dz_dc3(z,c1,pars,w_on = 1):  #derivative of dsti_dz
    # =============================================================================
    #     derivative of dsti_dz to c3
    # =============================================================================
    dA,H,B,n0 = pars
    
    dsb_dz_dc3 = 0
    dsp_dz_dc3 = lambda n: np.pi*H*n*((-np.cos(np.pi*n) + np.cos(H/np.sqrt(c1)))*np.sin(z/np.sqrt(c1))*1/np.sin(H/np.sqrt(c1)) + np.cos(z/np.sqrt(c1)) - np.cos(np.pi*n*z/H))/(H**2 - np.pi**2*c1*n**2)
    
    ds_dz_dc3 = dsb_dz_dc3 + dsp_dz_dc3(n0)

    return ds_dz_dc3


def dsti_dz_dc4(z,c1,pars,w_on = 1):  #derivative of dsti_dz
    # =============================================================================
    #     derivative of dsti_dz to c3
    # =============================================================================
    dA,H,B,n0 = pars
    
    dsb_dz_dc4 = 0
    dsp_dz_dc4 = lambda n: H*(np.pi*B*H**7*n*np.sin(z/np.sqrt(c1))*np.cos(np.pi*n)*np.sinh(dA)*1/np.sin(H/np.sqrt(c1)) - np.pi*B*H**7*n*np.cos(np.pi*n*z/H)*np.sinh(dA*z/H) - np.pi*B*H**5*c1*dA**2*n*np.sin(z/np.sqrt(c1))*np.cos(np.pi*n)*np.sinh(dA)*1/np.sin(H/np.sqrt(c1)) \
                        + np.pi*B*H**5*c1*dA**2*n*np.cos(np.pi*n*z/H)*np.sinh(dA*z/H) - 3*np.pi**3*B*H**5*c1*n**3*np.sin(z/np.sqrt(c1))*np.cos(np.pi*n)*np.sinh(dA)*1/np.sin(H/np.sqrt(c1)) + 3*np.pi**3*B*H**5*c1*n**3*np.cos(np.pi*n*z/H)*np.sinh(dA*z/H) \
                        + 2*np.pi**3*B*H**3*c1**2*dA**2*n**3*np.sin(z/np.sqrt(c1))*np.cos(np.pi*n)*np.sinh(dA)*1/np.sin(H/np.sqrt(c1)) - 2*np.pi**3*B*H**3*c1**2*dA**2*n**3*np.cos(np.pi*n*z/H)*np.sinh(dA*z/H) + 3*np.pi**5*B*H**3*c1**2*n**5*np.sin(z/np.sqrt(c1))*np.cos(np.pi*n)*np.sinh(dA)*1/np.sin(H/np.sqrt(c1)) \
                        - 3*np.pi**5*B*H**3*c1**2*n**5*np.cos(np.pi*n*z/H)*np.sinh(dA*z/H) - np.pi**5*B*H*c1**3*dA**2*n**5*np.sin(z/np.sqrt(c1))*np.cos(np.pi*n)*np.sinh(dA)*1/np.sin(H/np.sqrt(c1)) + np.pi**5*B*H*c1**3*dA**2*n**5*np.cos(np.pi*n*z/H)*np.sinh(dA*z/H) \
                        - np.pi**7*B*H*c1**3*n**7*np.sin(z/np.sqrt(c1))*np.cos(np.pi*n)*np.sinh(dA)*1/np.sin(H/np.sqrt(c1)) + np.pi**7*B*H*c1**3*n**7*np.cos(np.pi*n*z/H)*np.sinh(dA*z/H) \
                        + B*H*dA*(H**2 + c1*(dA**2 + np.pi**2*n**2))*(H**2 - np.pi**2*c1*n**2)**2*np.sin(np.pi*n)*np.sin(z/np.sqrt(c1))*np.cosh(dA)*1/np.sin(H/np.sqrt(c1)) - B*H*dA*(H**2 - np.pi**2*c1*n**2)**2*(H**2 + c1*dA**2 + np.pi**2*c1*n**2)*np.sin(np.pi*n*z/H)*np.cosh(dA*z/H) \
                        - np.pi*H**7*dA*n*np.sin(z/np.sqrt(c1))*np.cos(np.pi*n)*1/np.sin(H/np.sqrt(c1)) - H**7*dA*np.sin(np.pi*n)*np.sin(z/np.sqrt(c1))*1/np.sin(H/np.sqrt(c1)) + H**7*dA*np.sin(np.pi*n*z/H) + np.pi*H**6*dA*n*z*np.cos(np.pi*n*z/H) \
                        - 2*np.pi*H**5*c1*dA**3*n*np.sin(z/np.sqrt(c1))*np.cos(np.pi*n)*1/np.sin(H/np.sqrt(c1)) - 2*H**5*c1*dA**3*np.sin(np.pi*n)*np.sin(z/np.sqrt(c1))*1/np.sin(H/np.sqrt(c1)) + 2*H**5*c1*dA**3*np.sin(np.pi*n*z/H) \
                        + 3*np.pi**3*H**5*c1*dA*n**3*np.sin(z/np.sqrt(c1))*np.cos(np.pi*n)*1/np.sin(H/np.sqrt(c1)) + np.pi**2*H**5*c1*dA*n**2*np.sin(np.pi*n)*np.sin(z/np.sqrt(c1))*1/np.sin(H/np.sqrt(c1)) - np.pi**2*H**5*c1*dA*n**2*np.sin(np.pi*n*z/H) + 2*np.pi*H**4*c1*dA**3*n*z*np.cos(np.pi*n*z/H) \
                        - 3*np.pi**3*H**4*c1*dA*n**3*z*np.cos(np.pi*n*z/H) - np.pi*H**3*c1**2*dA**5*n*np.sin(z/np.sqrt(c1))*np.cos(np.pi*n)*1/np.sin(H/np.sqrt(c1)) - H**3*c1**2*dA**5*np.sin(np.pi*n)*np.sin(z/np.sqrt(c1))*1/np.sin(H/np.sqrt(c1)) + H**3*c1**2*dA**5*np.sin(np.pi*n*z/H) \
                        - 4*np.pi**2*H**3*c1**2*dA**3*n**2*np.sin(np.pi*n)*np.sin(z/np.sqrt(c1))*1/np.sin(H/np.sqrt(c1)) + 4*np.pi**2*H**3*c1**2*dA**3*n**2*np.sin(np.pi*n*z/H) - 3*np.pi**5*H**3*c1**2*dA*n**5*np.sin(z/np.sqrt(c1))*np.cos(np.pi*n)*1/np.sin(H/np.sqrt(c1)) \
                        + np.pi**4*H**3*c1**2*dA*n**4*np.sin(np.pi*n)*np.sin(z/np.sqrt(c1))*1/np.sin(H/np.sqrt(c1)) - np.pi**4*H**3*c1**2*dA*n**4*np.sin(np.pi*n*z/H) + np.pi*H**2*c1**2*dA**5*n*z*np.cos(np.pi*n*z/H) + 3*np.pi**5*H**2*c1**2*dA*n**5*z*np.cos(np.pi*n*z/H) \
                        + np.pi**3*H*c1**3*dA**5*n**3*np.sin(z/np.sqrt(c1))*np.cos(np.pi*n)*1/np.sin(H/np.sqrt(c1)) - np.pi**2*H*c1**3*dA**5*n**2*np.sin(np.pi*n)*np.sin(z/np.sqrt(c1))*1/np.sin(H/np.sqrt(c1)) + np.pi**2*H*c1**3*dA**5*n**2*np.sin(np.pi*n*z/H) \
                        + 2*np.pi**5*H*c1**3*dA**3*n**5*np.sin(z/np.sqrt(c1))*np.cos(np.pi*n)*1/np.sin(H/np.sqrt(c1)) - 2*np.pi**4*H*c1**3*dA**3*n**4*np.sin(np.pi*n)*np.sin(z/np.sqrt(c1))*1/np.sin(H/np.sqrt(c1)) + 2*np.pi**4*H*c1**3*dA**3*n**4*np.sin(np.pi*n*z/H) \
                        + np.pi**7*H*c1**3*dA*n**7*np.sin(z/np.sqrt(c1))*np.cos(np.pi*n)*1/np.sin(H/np.sqrt(c1)) - np.pi**6*H*c1**3*dA*n**6*np.sin(np.pi*n)*np.sin(z/np.sqrt(c1))*1/np.sin(H/np.sqrt(c1)) + np.pi**6*H*c1**3*dA*n**6*np.sin(np.pi*n*z/H) - np.pi**3*c1**3*dA**5*n**3*z*np.cos(np.pi*n*z/H) \
                        - 2*np.pi**5*c1**3*dA**3*n**5*z*np.cos(np.pi*n*z/H) - np.pi**7*c1**3*dA*n**7*z*np.cos(np.pi*n*z/H))/(dA*(H**2 - np.pi**2*c1*n**2)**2*(H**4 + 2*H**2*c1*(dA**2 - np.pi**2*n**2) + c1**2*(dA**2 + np.pi**2*n**2)**2))
    
    ds_dz_dc4 = dsb_dz_dc4 + dsp_dz_dc4(n0)

    return ds_dz_dc4





def dsti_dz_old(z,c,pars,w_on = 1): #seems to work fine , but super slow!
    c1,c2,c3,c4 = c
    c3,c4 = c3*w_on , c4*w_on
    dA,H,B,n0 = pars
    
    #calculate some returning parts for better perfomance
    c1_wo   = np.sqrt(c1)
    sin_Hcw = np.sin(H/c1_wo)
    sin_zcw = np.sin(z/c1_wo)
    sinh_d  = np.sinh(dA)
    sinh_dzH= np.sinh(dA*z/H)

    
    sbar = (-B*H*c2*dA*(sin_zcw*sinh_d*1/sin_Hcw - sinh_dzH)/(H**2 + c1*dA**2))[0]#part without n
    
    def spri(n):
        sin_npz = np.sin(np.pi*n*z/H)
        cos_npz = np.cos(np.pi*n*z/H)
        cos_pn  = np.cos(np.pi*n)
        return -(-B*H*c2*dA*(sin_zcw*sinh_d*1/sin_Hcw - sinh_dzH)/(H**2 + c1*dA**2))[0] + H*(np.pi*B*H**9*c4*n*sin_zcw*cos_pn*sinh_d*1/sin_Hcw - np.pi*B*H**9*c4*n*cos_npz*sinh_dzH - B*H**8*c2*dA**2*sin_zcw*sinh_d*1/sin_Hcw + B*H**8*c2*dA**2*sinh_dzH \
                - 3*np.pi**3*B*H**7*c1*c4*n**3*sin_zcw*cos_pn*sinh_d*1/sin_Hcw + 3*np.pi**3*B*H**7*c1*c4*n**3*cos_npz*sinh_dzH - 2*B*H**6*c1*c2*dA**4*sin_zcw*sinh_d*1/sin_Hcw + 2*B*H**6*c1*c2*dA**4*sinh_dzH + 4*np.pi**2*B*H**6*c1*c2*dA**2*n**2*sin_zcw*sinh_d*1/sin_Hcw \
                - 4*np.pi**2*B*H**6*c1*c2*dA**2*n**2*sinh_dzH - np.pi*B*H**5*c1**2*c4*dA**4*n*sin_zcw*cos_pn*sinh_d*1/sin_Hcw + np.pi*B*H**5*c1**2*c4*dA**4*n*cos_npz*sinh_dzH - np.pi**3*B*H**5*c1**2*c4*dA**2*n**3*sin_zcw*cos_pn*sinh_d*1/sin_Hcw \
                + np.pi**3*B*H**5*c1**2*c4*dA**2*n**3*cos_npz*sinh_dzH + 3*np.pi**5*B*H**5*c1**2*c4*n**5*sin_zcw*cos_pn*sinh_d*1/sin_Hcw - 3*np.pi**5*B*H**5*c1**2*c4*n**5*cos_npz*sinh_dzH - B*H**4*c1**2*c2*dA**6*sin_zcw*sinh_d*1/sin_Hcw \
                + B*H**4*c1**2*c2*dA**6*sinh_dzH + 2*np.pi**2*B*H**4*c1**2*c2*dA**4*n**2*sin_zcw*sinh_d*1/sin_Hcw - 2*np.pi**2*B*H**4*c1**2*c2*dA**4*n**2*sinh_dzH - 6*np.pi**4*B*H**4*c1**2*c2*dA**2*n**4*sin_zcw*sinh_d*1/sin_Hcw + 6*np.pi**4*B*H**4*c1**2*c2*dA**2*n**4*sinh_dzH \
                + 2*np.pi**3*B*H**3*c1**3*c4*dA**4*n**3*sin_zcw*cos_pn*sinh_d*1/sin_Hcw - 2*np.pi**3*B*H**3*c1**3*c4*dA**4*n**3*cos_npz*sinh_dzH + 2*np.pi**5*B*H**3*c1**3*c4*dA**2*n**5*sin_zcw*cos_pn*sinh_d*1/sin_Hcw - 2*np.pi**5*B*H**3*c1**3*c4*dA**2*n**5*cos_npz*sinh_dzH \
                - np.pi**7*B*H**3*c1**3*c4*n**7*sin_zcw*cos_pn*sinh_d*1/sin_Hcw + np.pi**7*B*H**3*c1**3*c4*n**7*cos_npz*sinh_dzH + 2*np.pi**2*B*H**2*c1**3*c2*dA**6*n**2*sin_zcw*sinh_d*1/sin_Hcw - 2*np.pi**2*B*H**2*c1**3*c2*dA**6*n**2*sinh_dzH \
                + 2*np.pi**4*B*H**2*c1**3*c2*dA**4*n**4*sin_zcw*sinh_d*1/sin_Hcw - 2*np.pi**4*B*H**2*c1**3*c2*dA**4*n**4*sinh_dzH + 4*np.pi**6*B*H**2*c1**3*c2*dA**2*n**6*sin_zcw*sinh_d*1/sin_Hcw - 4*np.pi**6*B*H**2*c1**3*c2*dA**2*n**6*sinh_dzH \
                - np.pi**5*B*H*c1**4*c4*dA**4*n**5*sin_zcw*cos_pn*sinh_d*1/sin_Hcw + np.pi**5*B*H*c1**4*c4*dA**4*n**5*cos_npz*sinh_dzH - np.pi**7*B*H*c1**4*c4*dA**2*n**7*sin_zcw*cos_pn*sinh_d*1/sin_Hcw + np.pi**7*B*H*c1**4*c4*dA**2*n**7*cos_npz*sinh_dzH \
                + B*H*c4*dA*(H**2 + c1*dA**2)*(H**2 + c1*(dA**2 + np.pi**2*n**2))*(H**2 - np.pi**2*c1*n**2)**2*np.sin(np.pi*n)*sin_zcw*np.cosh(dA)*1/sin_Hcw - B*H*c4*dA*(H**2 + c1*dA**2)*(H**2 - np.pi**2*c1*n**2)**2*(H**2 + c1*dA**2 + np.pi**2*c1*n**2)*sin_npz*np.cosh(dA*z/H) \
                - np.pi**4*B*c1**4*c2*dA**6*n**4*sin_zcw*sinh_d*1/sin_Hcw + np.pi**4*B*c1**4*c2*dA**6*n**4*sinh_dzH - 2*np.pi**6*B*c1**4*c2*dA**4*n**6*sin_zcw*sinh_d*1/sin_Hcw + 2*np.pi**6*B*c1**4*c2*dA**4*n**6*sinh_dzH  - np.pi**8*B*c1**4*c2*dA**2*n**8*sin_zcw*sinh_d*1/sin_Hcw \
                + np.pi**8*B*c1**4*c2*dA**2*n**8*sinh_dzH - np.pi*H**9*c4*dA*n*sin_zcw*cos_pn*1/sin_Hcw - H**9*c4*dA*np.sin(np.pi*n)*sin_zcw*1/sin_Hcw + H**9*c4*dA*sin_npz - np.pi*H**8*c3*dA*n*sin_zcw*cos_pn*1/sin_Hcw + np.pi*H**8*c3*dA*n*sin_zcw*1/np.tan(H/c1_wo) \
                + np.pi*H**8*c3*dA*n*np.cos(z/c1_wo) - np.pi*H**8*c3*dA*n*cos_npz + np.pi*H**8*c4*dA*n*z*cos_npz - 3*np.pi*H**7*c1*c4*dA**3*n*sin_zcw*cos_pn*1/sin_Hcw - 3*H**7*c1*c4*dA**3*np.sin(np.pi*n)*sin_zcw*1/sin_Hcw + 3*H**7*c1*c4*dA**3*sin_npz \
                + 3*np.pi**3*H**7*c1*c4*dA*n**3*sin_zcw*cos_pn*1/sin_Hcw + np.pi**2*H**7*c1*c4*dA*n**2*np.sin(np.pi*n)*sin_zcw*1/sin_Hcw - np.pi**2*H**7*c1*c4*dA*n**2*sin_npz - 3*np.pi*H**6*c1*c3*dA**3*n*sin_zcw*cos_pn*1/sin_Hcw + 3*np.pi*H**6*c1*c3*dA**3*n*sin_zcw*1/np.tan(H/c1_wo) \
                + 3*np.pi*H**6*c1*c3*dA**3*n*np.cos(z/c1_wo) - 3*np.pi*H**6*c1*c3*dA**3*n*cos_npz + 3*np.pi**3*H**6*c1*c3*dA*n**3*sin_zcw*cos_pn*1/sin_Hcw - 3*np.pi**3*H**6*c1*c3*dA*n**3*sin_zcw*1/np.tan(H/c1_wo) - 3*np.pi**3*H**6*c1*c3*dA*n**3*np.cos(z/c1_wo) \
                + 3*np.pi**3*H**6*c1*c3*dA*n**3*cos_npz + 3*np.pi*H**6*c1*c4*dA**3*n*z*cos_npz - 3*np.pi**3*H**6*c1*c4*dA*n**3*z*cos_npz - 3*np.pi*H**5*c1**2*c4*dA**5*n*sin_zcw*cos_pn*1/sin_Hcw - 3*H**5*c1**2*c4*dA**5*np.sin(np.pi*n)*sin_zcw*1/sin_Hcw \
                + 3*H**5*c1**2*c4*dA**5*sin_npz + 3*np.pi**3*H**5*c1**2*c4*dA**3*n**3*sin_zcw*cos_pn*1/sin_Hcw - 3*np.pi**2*H**5*c1**2*c4*dA**3*n**2*np.sin(np.pi*n)*sin_zcw*1/sin_Hcw + 3*np.pi**2*H**5*c1**2*c4*dA**3*n**2*sin_npz - 3*np.pi**5*H**5*c1**2*c4*dA*n**5*sin_zcw*cos_pn*1/sin_Hcw \
                + np.pi**4*H**5*c1**2*c4*dA*n**4*np.sin(np.pi*n)*sin_zcw*1/sin_Hcw - np.pi**4*H**5*c1**2*c4*dA*n**4*sin_npz - 3*np.pi*H**4*c1**2*c3*dA**5*n*sin_zcw*cos_pn*1/sin_Hcw + 3*np.pi*H**4*c1**2*c3*dA**5*n*sin_zcw*1/np.tan(H/c1_wo) + 3*np.pi*H**4*c1**2*c3*dA**5*n*np.cos(z/c1_wo) \
                - 3*np.pi*H**4*c1**2*c3*dA**5*n*cos_npz + 3*np.pi**3*H**4*c1**2*c3*dA**3*n**3*sin_zcw*cos_pn*1/sin_Hcw - 3*np.pi**3*H**4*c1**2*c3*dA**3*n**3*sin_zcw*1/np.tan(H/c1_wo) - 3*np.pi**3*H**4*c1**2*c3*dA**3*n**3*np.cos(z/c1_wo) \
                + 3*np.pi**3*H**4*c1**2*c3*dA**3*n**3*cos_npz - 3*np.pi**5*H**4*c1**2*c3*dA*n**5*sin_zcw*cos_pn*1/sin_Hcw + 3*np.pi**5*H**4*c1**2*c3*dA*n**5*sin_zcw*1/np.tan(H/c1_wo) + 3*np.pi**5*H**4*c1**2*c3*dA*n**5*np.cos(z/c1_wo) \
                - 3*np.pi**5*H**4*c1**2*c3*dA*n**5*cos_npz + 3*np.pi*H**4*c1**2*c4*dA**5*n*z*cos_npz - 3*np.pi**3*H**4*c1**2*c4*dA**3*n**3*z*cos_npz + 3*np.pi**5*H**4*c1**2*c4*dA*n**5*z*cos_npz - np.pi*H**3*c1**3*c4*dA**7*n*sin_zcw*cos_pn*1/sin_Hcw \
                - H**3*c1**3*c4*dA**7*np.sin(np.pi*n)*sin_zcw*1/sin_Hcw + H**3*c1**3*c4*dA**7*sin_npz + np.pi**3*H**3*c1**3*c4*dA**5*n**3*sin_zcw*cos_pn*1/sin_Hcw - 5*np.pi**2*H**3*c1**3*c4*dA**5*n**2*np.sin(np.pi*n)*sin_zcw*1/sin_Hcw + 5*np.pi**2*H**3*c1**3*c4*dA**5*n**2*sin_npz \
                - np.pi**5*H**3*c1**3*c4*dA**3*n**5*sin_zcw*cos_pn*1/sin_Hcw - np.pi**4*H**3*c1**3*c4*dA**3*n**4*np.sin(np.pi*n)*sin_zcw*1/sin_Hcw + np.pi**4*H**3*c1**3*c4*dA**3*n**4*sin_npz + np.pi**7*H**3*c1**3*c4*dA*n**7*sin_zcw*cos_pn*1/sin_Hcw \
                - np.pi**6*H**3*c1**3*c4*dA*n**6*np.sin(np.pi*n)*sin_zcw*1/sin_Hcw + np.pi**6*H**3*c1**3*c4*dA*n**6*sin_npz - np.pi*H**2*c1**3*c3*dA**7*n*sin_zcw*cos_pn*1/sin_Hcw + np.pi*H**2*c1**3*c3*dA**7*n*sin_zcw*1/np.tan(H/c1_wo) + np.pi*H**2*c1**3*c3*dA**7*n*np.cos(z/c1_wo) \
                - np.pi*H**2*c1**3*c3*dA**7*n*cos_npz + np.pi**3*H**2*c1**3*c3*dA**5*n**3*sin_zcw*cos_pn*1/sin_Hcw - np.pi**3*H**2*c1**3*c3*dA**5*n**3*sin_zcw*1/np.tan(H/c1_wo) - np.pi**3*H**2*c1**3*c3*dA**5*n**3*np.cos(z/c1_wo) + np.pi**3*H**2*c1**3*c3*dA**5*n**3*cos_npz \
                - np.pi**5*H**2*c1**3*c3*dA**3*n**5*sin_zcw*cos_pn*1/sin_Hcw + np.pi**5*H**2*c1**3*c3*dA**3*n**5*sin_zcw*1/np.tan(H/c1_wo) + np.pi**5*H**2*c1**3*c3*dA**3*n**5*np.cos(z/c1_wo) - np.pi**5*H**2*c1**3*c3*dA**3*n**5*cos_npz \
                + np.pi**7*H**2*c1**3*c3*dA*n**7*sin_zcw*cos_pn*1/sin_Hcw - np.pi**7*H**2*c1**3*c3*dA*n**7*sin_zcw*1/np.tan(H/c1_wo) - np.pi**7*H**2*c1**3*c3*dA*n**7*np.cos(z/c1_wo) + np.pi**7*H**2*c1**3*c3*dA*n**7*cos_npz \
                + np.pi*H**2*c1**3*c4*dA**7*n*z*cos_npz - np.pi**3*H**2*c1**3*c4*dA**5*n**3*z*cos_npz + np.pi**5*H**2*c1**3*c4*dA**3*n**5*z*cos_npz - np.pi**7*H**2*c1**3*c4*dA*n**7*z*cos_npz + np.pi**3*H*c1**4*c4*dA**7*n**3*sin_zcw*cos_pn*1/sin_Hcw \
                - np.pi**2*H*c1**4*c4*dA**7*n**2*np.sin(np.pi*n)*sin_zcw*1/sin_Hcw + np.pi**2*H*c1**4*c4*dA**7*n**2*sin_npz + 2*np.pi**5*H*c1**4*c4*dA**5*n**5*sin_zcw*cos_pn*1/sin_Hcw - 2*np.pi**4*H*c1**4*c4*dA**5*n**4*np.sin(np.pi*n)*sin_zcw*1/sin_Hcw \
                + 2*np.pi**4*H*c1**4*c4*dA**5*n**4*sin_npz + np.pi**7*H*c1**4*c4*dA**3*n**7*sin_zcw*cos_pn*1/sin_Hcw - np.pi**6*H*c1**4*c4*dA**3*n**6*np.sin(np.pi*n)*sin_zcw*1/sin_Hcw + np.pi**6*H*c1**4*c4*dA**3*n**6*sin_npz + np.pi**3*c1**4*c3*dA**7*n**3*sin_zcw*cos_pn*1/sin_Hcw \
                - np.pi**3*c1**4*c3*dA**7*n**3*sin_zcw*1/np.tan(H/c1_wo) - np.pi**3*c1**4*c3*dA**7*n**3*np.cos(z/c1_wo) + np.pi**3*c1**4*c3*dA**7*n**3*cos_npz + 2*np.pi**5*c1**4*c3*dA**5*n**5*sin_zcw*cos_pn*1/sin_Hcw - 2*np.pi**5*c1**4*c3*dA**5*n**5*sin_zcw*1/np.tan(H/c1_wo) \
                - 2*np.pi**5*c1**4*c3*dA**5*n**5*np.cos(z/c1_wo) + 2*np.pi**5*c1**4*c3*dA**5*n**5*cos_npz + np.pi**7*c1**4*c3*dA**3*n**7*sin_zcw*cos_pn*1/sin_Hcw - np.pi**7*c1**4*c3*dA**3*n**7*sin_zcw*1/np.tan(H/c1_wo) - np.pi**7*c1**4*c3*dA**3*n**7*np.cos(z/c1_wo) + np.pi**7*c1**4*c3*dA**3*n**7*cos_npz \
                - np.pi**3*c1**4*c4*dA**7*n**3*z*cos_npz - 2*np.pi**5*c1**4*c4*dA**5*n**5*z*cos_npz - np.pi**7*c1**4*c4*dA**3*n**7*z*cos_npz)/(dA*(H**2 - np.pi**2*c1*n**2)**2*(H**6 + H**4*c1*(3*dA**2 - 2*np.pi**2*n**2) + H**2*c1**2*(3*dA**4 + np.pi**4*n**4) + c1**3*(dA**3 + np.pi**2*dA*n**2)**2))

    out = sbar+spri(n0).sum(0)
    
    return out

def sti_old(z,c,pars,w_on = 1): 
    # =============================================================================
    # calculate tidal salinity (complex notation, so \hat s_ti)
    # old version, this is the correct but slow version
    # =============================================================================
    c1,c2,c3,c4 = c
    c3,c4 = c3*w_on , c4*w_on
    dA,H,B,n0 = pars

    sbar = (-c2*(-B*H**2*np.cosh(dA*z/H) - B*H*np.sqrt(c1)*dA*np.cos(z/np.sqrt(c1))*np.sinh(dA)*1/np.sin(H/np.sqrt(c1)) + H**2 + c1*dA**2)/(H**2 + c1*dA**2))[0]

    spri = lambda n: -sbar + (-B*H**11*c4*np.sin(np.pi*n*z/H)*np.sinh(dA*z/H) - np.pi*B*H**10*np.sqrt(c1)*c4*n*np.cos(np.pi*n)*np.cos(z/np.sqrt(c1))*np.sinh(dA)*1/np.sin(H/np.sqrt(c1)) + B*H**9*np.sqrt(c1)*c2*dA**2*np.cos(z/np.sqrt(c1))*np.sinh(dA)*1/np.sin(H/np.sqrt(c1)) - 2*B*H**9*c1*c4*dA**2*np.sin(np.pi*n*z/H)*np.sinh(dA*z/H) \
        + 3*np.pi**2*B*H**9*c1*c4*n**2*np.sin(np.pi*n*z/H)*np.sinh(dA*z/H) + 3*np.pi**3*B*H**8*c1**(3/2)*c4*n**3*np.cos(np.pi*n)*np.cos(z/np.sqrt(c1))*np.sinh(dA)*1/np.sin(H/np.sqrt(c1)) + 2*B*H**7*c1**(3/2)*c2*dA**4*np.cos(z/np.sqrt(c1))*np.sinh(dA)*1/np.sin(H/np.sqrt(c1)) \
        - 4*np.pi**2*B*H**7*c1**(3/2)*c2*dA**2*n**2*np.cos(z/np.sqrt(c1))*np.sinh(dA)*1/np.sin(H/np.sqrt(c1)) - B*H**7*c1**2*c4*dA**4*np.sin(np.pi*n*z/H)*np.sinh(dA*z/H) + 5*np.pi**2*B*H**7*c1**2*c4*dA**2*n**2*np.sin(np.pi*n*z/H)*np.sinh(dA*z/H) - 3*np.pi**4*B*H**7*c1**2*c4*n**4*np.sin(np.pi*n*z/H)*np.sinh(dA*z/H) \
        + np.pi*B*H**6*c1**(5/2)*c4*dA**4*n*np.cos(np.pi*n)*np.cos(z/np.sqrt(c1))*np.sinh(dA)*1/np.sin(H/np.sqrt(c1)) + np.pi**3*B*H**6*c1**(5/2)*c4*dA**2*n**3*np.cos(np.pi*n)*np.cos(z/np.sqrt(c1))*np.sinh(dA)*1/np.sin(H/np.sqrt(c1)) \
        - 3*np.pi**5*B*H**6*c1**(5/2)*c4*n**5*np.cos(np.pi*n)*np.cos(z/np.sqrt(c1))*np.sinh(dA)*1/np.sin(H/np.sqrt(c1)) + B*H**5*c1**(5/2)*c2*dA**6*np.cos(z/np.sqrt(c1))*np.sinh(dA)*1/np.sin(H/np.sqrt(c1)) - 2*np.pi**2*B*H**5*c1**(5/2)*c2*dA**4*n**2*np.cos(z/np.sqrt(c1))*np.sinh(dA)*1/np.sin(H/np.sqrt(c1)) \
        + 6*np.pi**4*B*H**5*c1**(5/2)*c2*dA**2*n**4*np.cos(z/np.sqrt(c1))*np.sinh(dA)*1/np.sin(H/np.sqrt(c1)) + 2*np.pi**2*B*H**5*c1**3*c4*dA**4*n**2*np.sin(np.pi*n*z/H)*np.sinh(dA*z/H) - 4*np.pi**4*B*H**5*c1**3*c4*dA**2*n**4*np.sin(np.pi*n*z/H)*np.sinh(dA*z/H) + np.pi**6*B*H**5*c1**3*c4*n**6*np.sin(np.pi*n*z/H)*np.sinh(dA*z/H) \
        - 2*np.pi**3*B*H**4*c1**(7/2)*c4*dA**4*n**3*np.cos(np.pi*n)*np.cos(z/np.sqrt(c1))*np.sinh(dA)*1/np.sin(H/np.sqrt(c1)) - 2*np.pi**5*B*H**4*c1**(7/2)*c4*dA**2*n**5*np.cos(np.pi*n)*np.cos(z/np.sqrt(c1))*np.sinh(dA)*1/np.sin(H/np.sqrt(c1)) \
        + np.pi**7*B*H**4*c1**(7/2)*c4*n**7*np.cos(np.pi*n)*np.cos(z/np.sqrt(c1))*np.sinh(dA)*1/np.sin(H/np.sqrt(c1)) - 2*np.pi**2*B*H**3*c1**(7/2)*c2*dA**6*n**2*np.cos(z/np.sqrt(c1))*np.sinh(dA)*1/np.sin(H/np.sqrt(c1)) - 2*np.pi**4*B*H**3*c1**(7/2)*c2*dA**4*n**4*np.cos(z/np.sqrt(c1))*np.sinh(dA)*1/np.sin(H/np.sqrt(c1)) \
        - 4*np.pi**6*B*H**3*c1**(7/2)*c2*dA**2*n**6*np.cos(z/np.sqrt(c1))*np.sinh(dA)*1/np.sin(H/np.sqrt(c1)) - np.pi**4*B*H**3*c1**4*c4*dA**4*n**4*np.sin(np.pi*n*z/H)*np.sinh(dA*z/H) + np.pi**6*B*H**3*c1**4*c4*dA**2*n**6*np.sin(np.pi*n*z/H)*np.sinh(dA*z/H) \
        + np.pi**5*B*H**2*c1**(9/2)*c4*dA**4*n**5*np.cos(np.pi*n)*np.cos(z/np.sqrt(c1))*np.sinh(dA)*1/np.sin(H/np.sqrt(c1)) + np.pi**7*B*H**2*c1**(9/2)*c4*dA**2*n**7*np.cos(np.pi*n)*np.cos(z/np.sqrt(c1))*np.sinh(dA)*1/np.sin(H/np.sqrt(c1)) \
        + np.pi**4*B*H*c1**(9/2)*c2*dA**6*n**4*np.cos(z/np.sqrt(c1))*np.sinh(dA)*1/np.sin(H/np.sqrt(c1)) + 2*np.pi**6*B*H*c1**(9/2)*c2*dA**4*n**6*np.cos(z/np.sqrt(c1))*np.sinh(dA)*1/np.sin(H/np.sqrt(c1)) + np.pi**8*B*H*c1**(9/2)*c2*dA**2*n**8*np.cos(z/np.sqrt(c1))*np.sinh(dA)*1/np.sin(H/np.sqrt(c1)) \
        - B*np.sqrt(c1)*c4*dA*(H**2 + c1*dA**2)*(H**3 - np.pi**2*H*c1*n**2)**2*(H**2 + c1*dA**2 + np.pi**2*c1*n**2)*np.sin(np.pi*n)*np.cos(z/np.sqrt(c1))*np.cosh(dA)*1/np.sin(H/np.sqrt(c1)) + B*dA*(H**3 - np.pi**2*H*c1*n**2)**2*(2*np.pi*H*c1*c4*n*(H**2 + c1*dA**2)*np.cos(np.pi*n*z/H) \
        + c2*(H**4 + 2*H**2*c1*(dA**2 - np.pi**2*n**2) + c1**2*(dA**2 + np.pi**2*n**2)**2))*np.cosh(dA*z/H) + np.pi*H**10*np.sqrt(c1)*c4*dA*n*np.cos(np.pi*n)*np.cos(z/np.sqrt(c1))*1/np.sin(H/np.sqrt(c1)) + H**10*np.sqrt(c1)*c4*dA*np.sin(np.pi*n)*np.cos(z/np.sqrt(c1))*1/np.sin(H/np.sqrt(c1)) - H**10*c2*dA \
        - H**10*c3*dA*np.sin(np.pi*n*z/H) + H**10*c4*dA*z*np.sin(np.pi*n*z/H) + np.pi*H**9*np.sqrt(c1)*c3*dA*n*np.sin(z/np.sqrt(c1)) + np.pi*H**9*np.sqrt(c1)*c3*dA*n*np.cos(np.pi*n)*np.cos(z/np.sqrt(c1))*1/np.sin(H/np.sqrt(c1)) - np.pi*H**9*np.sqrt(c1)*c3*dA*n*np.cos(z/np.sqrt(c1))*1/np.tan(H/np.sqrt(c1)) \
        - 2*np.pi*H**9*c1*c4*dA*n*np.cos(np.pi*n*z/H) + 3*np.pi*H**8*c1**(3/2)*c4*dA**3*n*np.cos(np.pi*n)*np.cos(z/np.sqrt(c1))*1/np.sin(H/np.sqrt(c1)) + 3*H**8*c1**(3/2)*c4*dA**3*np.sin(np.pi*n)*np.cos(z/np.sqrt(c1))*1/np.sin(H/np.sqrt(c1)) \
        - 3*np.pi**3*H**8*c1**(3/2)*c4*dA*n**3*np.cos(np.pi*n)*np.cos(z/np.sqrt(c1))*1/np.sin(H/np.sqrt(c1)) - np.pi**2*H**8*c1**(3/2)*c4*dA*n**2*np.sin(np.pi*n)*np.cos(z/np.sqrt(c1))*1/np.sin(H/np.sqrt(c1)) - 3*H**8*c1*c2*dA**3 + 4*np.pi**2*H**8*c1*c2*dA*n**2 - 3*H**8*c1*c3*dA**3*np.sin(np.pi*n*z/H) \
        + 3*np.pi**2*H**8*c1*c3*dA*n**2*np.sin(np.pi*n*z/H) + 3*H**8*c1*c4*dA**3*z*np.sin(np.pi*n*z/H) - 3*np.pi**2*H**8*c1*c4*dA*n**2*z*np.sin(np.pi*n*z/H) + 3*np.pi*H**7*c1**(3/2)*c3*dA**3*n*np.sin(z/np.sqrt(c1)) + 3*np.pi*H**7*c1**(3/2)*c3*dA**3*n*np.cos(np.pi*n)*np.cos(z/np.sqrt(c1))*1/np.sin(H/np.sqrt(c1)) \
        - 3*np.pi*H**7*c1**(3/2)*c3*dA**3*n*np.cos(z/np.sqrt(c1))*1/np.tan(H/np.sqrt(c1)) - 3*np.pi**3*H**7*c1**(3/2)*c3*dA*n**3*np.sin(z/np.sqrt(c1)) - 3*np.pi**3*H**7*c1**(3/2)*c3*dA*n**3*np.cos(np.pi*n)*np.cos(z/np.sqrt(c1))*1/np.sin(H/np.sqrt(c1)) \
        + 3*np.pi**3*H**7*c1**(3/2)*c3*dA*n**3*np.cos(z/np.sqrt(c1))*1/np.tan(H/np.sqrt(c1)) - 6*np.pi*H**7*c1**2*c4*dA**3*n*np.cos(np.pi*n*z/H) + 4*np.pi**3*H**7*c1**2*c4*dA*n**3*np.cos(np.pi*n*z/H) + 3*np.pi*H**6*c1**(5/2)*c4*dA**5*n*np.cos(np.pi*n)*np.cos(z/np.sqrt(c1))*1/np.sin(H/np.sqrt(c1)) \
        + 3*H**6*c1**(5/2)*c4*dA**5*np.sin(np.pi*n)*np.cos(z/np.sqrt(c1))*1/np.sin(H/np.sqrt(c1)) - 3*np.pi**3*H**6*c1**(5/2)*c4*dA**3*n**3*np.cos(np.pi*n)*np.cos(z/np.sqrt(c1))*1/np.sin(H/np.sqrt(c1)) + 3*np.pi**2*H**6*c1**(5/2)*c4*dA**3*n**2*np.sin(np.pi*n)*np.cos(z/np.sqrt(c1))*1/np.sin(H/np.sqrt(c1)) \
        + 3*np.pi**5*H**6*c1**(5/2)*c4*dA*n**5*np.cos(np.pi*n)*np.cos(z/np.sqrt(c1))*1/np.sin(H/np.sqrt(c1)) - np.pi**4*H**6*c1**(5/2)*c4*dA*n**4*np.sin(np.pi*n)*np.cos(z/np.sqrt(c1))*1/np.sin(H/np.sqrt(c1)) - 3*H**6*c1**2*c2*dA**5 + 6*np.pi**2*H**6*c1**2*c2*dA**3*n**2 - 6*np.pi**4*H**6*c1**2*c2*dA*n**4 \
        - 3*H**6*c1**2*c3*dA**5*np.sin(np.pi*n*z/H) + 3*np.pi**2*H**6*c1**2*c3*dA**3*n**2*np.sin(np.pi*n*z/H) - 3*np.pi**4*H**6*c1**2*c3*dA*n**4*np.sin(np.pi*n*z/H) + 3*H**6*c1**2*c4*dA**5*z*np.sin(np.pi*n*z/H) - 3*np.pi**2*H**6*c1**2*c4*dA**3*n**2*z*np.sin(np.pi*n*z/H) + 3*np.pi**4*H**6*c1**2*c4*dA*n**4*z*np.sin(np.pi*n*z/H) \
        + 3*np.pi*H**5*c1**(5/2)*c3*dA**5*n*np.sin(z/np.sqrt(c1)) + 3*np.pi*H**5*c1**(5/2)*c3*dA**5*n*np.cos(np.pi*n)*np.cos(z/np.sqrt(c1))*1/np.sin(H/np.sqrt(c1)) - 3*np.pi*H**5*c1**(5/2)*c3*dA**5*n*np.cos(z/np.sqrt(c1))*1/np.tan(H/np.sqrt(c1)) - 3*np.pi**3*H**5*c1**(5/2)*c3*dA**3*n**3*np.sin(z/np.sqrt(c1)) \
        - 3*np.pi**3*H**5*c1**(5/2)*c3*dA**3*n**3*np.cos(np.pi*n)*np.cos(z/np.sqrt(c1))*1/np.sin(H/np.sqrt(c1)) + 3*np.pi**3*H**5*c1**(5/2)*c3*dA**3*n**3*np.cos(z/np.sqrt(c1))*1/np.tan(H/np.sqrt(c1)) + 3*np.pi**5*H**5*c1**(5/2)*c3*dA*n**5*np.sin(z/np.sqrt(c1)) \
        + 3*np.pi**5*H**5*c1**(5/2)*c3*dA*n**5*np.cos(np.pi*n)*np.cos(z/np.sqrt(c1))*1/np.sin(H/np.sqrt(c1)) - 3*np.pi**5*H**5*c1**(5/2)*c3*dA*n**5*np.cos(z/np.sqrt(c1))*1/np.tan(H/np.sqrt(c1)) - 6*np.pi*H**5*c1**3*c4*dA**5*n*np.cos(np.pi*n*z/H) - 2*np.pi**5*H**5*c1**3*c4*dA*n**5*np.cos(np.pi*n*z/H) \
        + np.pi*H**4*c1**(7/2)*c4*dA**7*n*np.cos(np.pi*n)*np.cos(z/np.sqrt(c1))*1/np.sin(H/np.sqrt(c1)) + H**4*c1**(7/2)*c4*dA**7*np.sin(np.pi*n)*np.cos(z/np.sqrt(c1))*1/np.sin(H/np.sqrt(c1)) - np.pi**3*H**4*c1**(7/2)*c4*dA**5*n**3*np.cos(np.pi*n)*np.cos(z/np.sqrt(c1))*1/np.sin(H/np.sqrt(c1)) \
        + 5*np.pi**2*H**4*c1**(7/2)*c4*dA**5*n**2*np.sin(np.pi*n)*np.cos(z/np.sqrt(c1))*1/np.sin(H/np.sqrt(c1)) + np.pi**5*H**4*c1**(7/2)*c4*dA**3*n**5*np.cos(np.pi*n)*np.cos(z/np.sqrt(c1))*1/np.sin(H/np.sqrt(c1)) + np.pi**4*H**4*c1**(7/2)*c4*dA**3*n**4*np.sin(np.pi*n)*np.cos(z/np.sqrt(c1))*1/np.sin(H/np.sqrt(c1)) \
        - np.pi**7*H**4*c1**(7/2)*c4*dA*n**7*np.cos(np.pi*n)*np.cos(z/np.sqrt(c1))*1/np.sin(H/np.sqrt(c1)) + np.pi**6*H**4*c1**(7/2)*c4*dA*n**6*np.sin(np.pi*n)*np.cos(z/np.sqrt(c1))*1/np.sin(H/np.sqrt(c1)) - H**4*c1**3*c2*dA**7 + 4*np.pi**2*H**4*c1**3*c2*dA**5*n**2 - 4*np.pi**4*H**4*c1**3*c2*dA**3*n**4 \
        + 4*np.pi**6*H**4*c1**3*c2*dA*n**6 - H**4*c1**3*c3*dA**7*np.sin(np.pi*n*z/H) + np.pi**2*H**4*c1**3*c3*dA**5*n**2*np.sin(np.pi*n*z/H) - np.pi**4*H**4*c1**3*c3*dA**3*n**4*np.sin(np.pi*n*z/H) + np.pi**6*H**4*c1**3*c3*dA*n**6*np.sin(np.pi*n*z/H) + H**4*c1**3*c4*dA**7*z*np.sin(np.pi*n*z/H) \
        - np.pi**2*H**4*c1**3*c4*dA**5*n**2*z*np.sin(np.pi*n*z/H) + np.pi**4*H**4*c1**3*c4*dA**3*n**4*z*np.sin(np.pi*n*z/H) - np.pi**6*H**4*c1**3*c4*dA*n**6*z*np.sin(np.pi*n*z/H) + np.pi*H**3*c1**(7/2)*c3*dA**7*n*np.sin(z/np.sqrt(c1)) + np.pi*H**3*c1**(7/2)*c3*dA**7*n*np.cos(np.pi*n)*np.cos(z/np.sqrt(c1))*1/np.sin(H/np.sqrt(c1)) \
        - np.pi*H**3*c1**(7/2)*c3*dA**7*n*np.cos(z/np.sqrt(c1))*1/np.tan(H/np.sqrt(c1)) - np.pi**3*H**3*c1**(7/2)*c3*dA**5*n**3*np.sin(z/np.sqrt(c1)) - np.pi**3*H**3*c1**(7/2)*c3*dA**5*n**3*np.cos(np.pi*n)*np.cos(z/np.sqrt(c1))*1/np.sin(H/np.sqrt(c1)) \
        + np.pi**3*H**3*c1**(7/2)*c3*dA**5*n**3*np.cos(z/np.sqrt(c1))*1/np.tan(H/np.sqrt(c1)) + np.pi**5*H**3*c1**(7/2)*c3*dA**3*n**5*np.sin(z/np.sqrt(c1)) + np.pi**5*H**3*c1**(7/2)*c3*dA**3*n**5*np.cos(np.pi*n)*np.cos(z/np.sqrt(c1))*1/np.sin(H/np.sqrt(c1)) \
        - np.pi**5*H**3*c1**(7/2)*c3*dA**3*n**5*np.cos(z/np.sqrt(c1))*1/np.tan(H/np.sqrt(c1)) - np.pi**7*H**3*c1**(7/2)*c3*dA*n**7*np.sin(z/np.sqrt(c1)) - np.pi**7*H**3*c1**(7/2)*c3*dA*n**7*np.cos(np.pi*n)*np.cos(z/np.sqrt(c1))*1/np.sin(H/np.sqrt(c1)) \
        + np.pi**7*H**3*c1**(7/2)*c3*dA*n**7*np.cos(z/np.sqrt(c1))*1/np.tan(H/np.sqrt(c1)) - 2*np.pi*H**3*c1**4*c4*dA**7*n*np.cos(np.pi*n*z/H) - 4*np.pi**3*H**3*c1**4*c4*dA**5*n**3*np.cos(np.pi*n*z/H) - 2*np.pi**5*H**3*c1**4*c4*dA**3*n**5*np.cos(np.pi*n*z/H) \
        - np.pi**3*H**2*c1**(9/2)*c4*dA**7*n**3*np.cos(np.pi*n)*np.cos(z/np.sqrt(c1))*1/np.sin(H/np.sqrt(c1)) + np.pi**2*H**2*c1**(9/2)*c4*dA**7*n**2*np.sin(np.pi*n)*np.cos(z/np.sqrt(c1))*1/np.sin(H/np.sqrt(c1)) - 2*np.pi**5*H**2*c1**(9/2)*c4*dA**5*n**5*np.cos(np.pi*n)*np.cos(z/np.sqrt(c1))*1/np.sin(H/np.sqrt(c1)) \
        + 2*np.pi**4*H**2*c1**(9/2)*c4*dA**5*n**4*np.sin(np.pi*n)*np.cos(z/np.sqrt(c1))*1/np.sin(H/np.sqrt(c1)) - np.pi**7*H**2*c1**(9/2)*c4*dA**3*n**7*np.cos(np.pi*n)*np.cos(z/np.sqrt(c1))*1/np.sin(H/np.sqrt(c1)) + np.pi**6*H**2*c1**(9/2)*c4*dA**3*n**6*np.sin(np.pi*n)*np.cos(z/np.sqrt(c1))*1/np.sin(H/np.sqrt(c1)) \
        + 2*np.pi**2*H**2*c1**4*c2*dA**7*n**2 + np.pi**4*H**2*c1**4*c2*dA**5*n**4 + 2*np.pi**6*H**2*c1**4*c2*dA**3*n**6 - np.pi**8*H**2*c1**4*c2*dA*n**8 + np.pi**2*H**2*c1**4*c3*dA**7*n**2*np.sin(np.pi*n*z/H) + 2*np.pi**4*H**2*c1**4*c3*dA**5*n**4*np.sin(np.pi*n*z/H) + np.pi**6*H**2*c1**4*c3*dA**3*n**6*np.sin(np.pi*n*z/H) \
        - np.pi**2*H**2*c1**4*c4*dA**7*n**2*z*np.sin(np.pi*n*z/H) - 2*np.pi**4*H**2*c1**4*c4*dA**5*n**4*z*np.sin(np.pi*n*z/H) - np.pi**6*H**2*c1**4*c4*dA**3*n**6*z*np.sin(np.pi*n*z/H) - np.pi**3*H*c1**(9/2)*c3*dA**7*n**3*np.sin(z/np.sqrt(c1)) - np.pi**3*H*c1**(9/2)*c3*dA**7*n**3*np.cos(np.pi*n)*np.cos(z/np.sqrt(c1))*1/np.sin(H/np.sqrt(c1)) \
        + np.pi**3*H*c1**(9/2)*c3*dA**7*n**3*np.cos(z/np.sqrt(c1))*1/np.tan(H/np.sqrt(c1)) - 2*np.pi**5*H*c1**(9/2)*c3*dA**5*n**5*np.sin(z/np.sqrt(c1)) - 2*np.pi**5*H*c1**(9/2)*c3*dA**5*n**5*np.cos(np.pi*n)*np.cos(z/np.sqrt(c1))*1/np.sin(H/np.sqrt(c1)) \
        + 2*np.pi**5*H*c1**(9/2)*c3*dA**5*n**5*np.cos(z/np.sqrt(c1))*1/np.tan(H/np.sqrt(c1)) - np.pi**7*H*c1**(9/2)*c3*dA**3*n**7*np.sin(z/np.sqrt(c1)) - np.pi**7*H*c1**(9/2)*c3*dA**3*n**7*np.cos(np.pi*n)*np.cos(z/np.sqrt(c1))*1/np.sin(H/np.sqrt(c1)) \
        + np.pi**7*H*c1**(9/2)*c3*dA**3*n**7*np.cos(z/np.sqrt(c1))*1/np.tan(H/np.sqrt(c1)) - np.pi**4*c1**5*c2*dA**7*n**4 - 2*np.pi**6*c1**5*c2*dA**5*n**6 - np.pi**8*c1**5*c2*dA**3*n**8)/(dA*(H**2 + c1*dA**2)*(H**2 - np.pi**2*c1*n**2)**2*(H**4 + 2*H**2*c1*(dA**2 - np.pi**2*n**2) + c1**2*(dA**2 + np.pi**2*n**2)**2))
            
    out = sbar+spri(n0).sum(0)
    
    return out
