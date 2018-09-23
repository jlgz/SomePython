# -*- coding: utf-8 -*-
"""

@author: Jose Luis Guillen Zafra
"""
import numpy as np
import math
import matplotlib.pyplot as plt
def main():
    print 'si esta ejecutando sobre una terminal de python cierre las ventanas'
    print 'emergentes de los graficos para continuar con la ejecucion del programa'
    k = raw_input('pulsa una tecla para continuar ')
    plt.ion()
    menu1()
 #FSK([0,1,1,0,0,1,0])
 #B8ZS([1,1,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,1,0])
 #HDB3([1,1,1,0,0,0,0,0,0,0,0,0,1,0,1])
def menu1():
    o = 0
    seq = []
    valid = False
    while((o != 3 and not valid)or(o!=5 and valid)):
        print 'seq actual: ', seq
        print '1-Introducir nueva secuencia'
        print '2-Introducir secuencia aleatoria'
        if not valid:
            print '3-Salir'
        else:
            print '3-Modular secuencia'
            print '4-Codificar secuencia'
            print '5-Salir'
        o = raw_input('Selecciona una opcion ')
        try: o = int(o)
        except ValueError: o = 0
        if (not valid and o>3) or o<1 or(valid and o >5): 
            print 'Entra una opcion valida!'
        if o== 1: 
            seq = input_seq()
            valid = True
        if o == 2:
            seq = random_b_seq()
            valid = True
        if valid and o == 3: menu2(seq)
        if valid and o == 4: menu3(seq)
def menu2(seq):
    o = 0
    while(True):
        print '1-Modulacion ASK'
        print '2-Modulacion FSK'
        print '3-Modulacion PSK'
        print '4-Atras'
        o = raw_input('Selecciona una opcion ')
        try: o = int(o)
        except ValueError: o = 0
        if o>4 or o<1: print 'Entra una opcion valida!'
        else:
            if o == 4: break
            print seq
            if o == 1: ASK(seq)
            if o == 2: FSK(seq)
            if o == 3: PSK(seq)
            k = raw_input('pulsa una tecla para continuar ')
def menu3(seq):
    o = 0
    while(True):
        print '1-Codificalion NRZ'
        print '2-Codificalion NRZ-L'
        print '3-Codificalion NRZI'
        print '4-Codificalion bipolar AMI'
        print '5-Codificalion pseudoternaria'
        print '6-Codificalion B8ZS'
        print '7-Codificalion HDB3'
        print '8-Codificalion Manchester'
        print '9-Codificalion Manchester diferencial'
        print '10-Atras'
        o = raw_input('Selecciona una opcion ')
        try: o = int(o)
        except ValueError: o = 0
        if o>10 or o<1: print 'Entra una opcion valida!'
        else:
            if o == 10: break
            print seq
            if o == 1: NRZ(list(seq))
            if o == 2: NRZ_L(list(seq))
            if o == 3: NRZI(list(seq))
            if o == 4: bipolar_AMI(list(seq))
            if o == 5: pseudoternary(list(seq))
            if o == 6: B8ZS(list(seq))
            if o == 7: HDB3(list(seq))
            if o == 8: manchester(list(seq))
            if o == 9: manchester_D(list(seq))
            k = raw_input('pulsa una tecla para continuar ')
def random_b_seq():
    s = 1    
    while(s<10):
        s = raw_input('entra el tamanio de la secuencia(10 o mas) ')
        try: s = int(s)
        except ValueError: s = 1
    return np.random.randint(2,size = s)
def input_seq():
    size = 1    
    while(size<10):
        size = raw_input('entra el tamanio de la secuencia(10 o mas) ')
        try: size = int(size)
        except ValueError: size = 1
    seq = []
    for i in range(size):
        b = 3
        while(b not in [0,1]):
            print seq
            b = raw_input('entra el siguiente bit ')
            try: b = int(b)
            except ValueError: b = b
        seq = seq +[b]
    return seq
def wave_form(A,f,t,p = 0):
    return A * np.sin(2*math.pi*f*t +p)
def NRZ(seq):
    plt.show()
    x = np.arange(0, len(seq)+1)
    for i in range(len(seq)): seq[i] -= 0.5
    y = np.array([0]+seq)
    plt.xlim(0, len(seq))
    plt.ylim(-1, 1)
    plt.step(x, y)
    for i in x: plt.axvline(i,color='k',linestyle='--')
    plt.draw()
    plt.pause(0.01)
    plt.show(block = True)
def NRZ_L(seq):
    plt.show()
    x = np.arange(0, len(seq)+1)
    for i in range(len(seq)): 
        seq[i] -= 0.5
        seq[i] *= -1
    y = np.array([0]+seq)
    plt.xlim(0, len(seq))
    plt.ylim(-1, 1)
    plt.step(x, y)
    for i in x: plt.axvline(i,color='k',linestyle='--')
    plt.draw()
    plt.pause(0.01)
    plt.show(block = True)
    
def NRZI(seq):
    plt.show()
    x = np.arange(0, len(seq)+1)
    seq[0] -= 0.5
    for i in range(1,len(seq)): 
        if(seq[i] == 1): seq[i] = -seq[i-1]
        else: seq[i] = seq[i-1]
    y = np.array([0]+seq)
    plt.xlim(0, len(seq))
    plt.ylim(-1, 1)
    plt.step(x, y)
    for i in x: plt.axvline(i,color='k',linestyle='--')
    plt.draw()
    plt.pause(0.01)
    plt.show(block = True)
def bipolar_AMI(seq):
    plt.show()
    up = True
    x = np.arange(0, len(seq)+1)
    for i in range(len(seq)): 
        if seq[i] == 1:
            seq[i] -= 0.5
            if not up: seq[i] *=-1
            up = not up
    y = np.array([0]+seq)
    plt.xlim(0, len(seq))
    plt.ylim(-1, 1)
    plt.step(x, y)
    for i in x: plt.axvline(i,color='k',linestyle='--')
    plt.draw()
    plt.pause(0.01)
    plt.show(block = True)
def pseudoternary(seq):
    plt.show()
    for i in range(len(seq)):
        if seq[i]== 1: seq[i] = 0
        else: seq[i] = 1
    bipolar_AMI(seq)
def manchester(seq):
    plt.show()
    x = np.arange(0.0, len(seq)+ 0.5,0.5)
    y = [0] * len(x)
    y[1] = -(seq[0] -0.5)
    j=2
    for i in range(1,len(seq)):
       y[j] =  -y[j-1]
       y[j+1] = -(seq[i] -0.5)
       j+=2
    y[j] = -y[j-1]
    y = np.array(y)
    plt.xlim(0, len(seq))
    plt.ylim(-1, 1)
    plt.step(x, y)
    for i in np.arange(0,len(seq)+1): plt.axvline(i,color='k',linestyle='--')
    plt.draw()
    plt.pause(0.01) 
    plt.show(block = True)
def manchester_D(seq):
    plt.show()
    x = np.arange(0.0, len(seq)+0.5,0.5)
    y = [0.5] * len(x)
    y[1] = seq[0] -0.5
    j=2
    for i in range(1,len(seq)):
       y[j] =  -y[j-1]
       if seq[i] == 1: y[j+1] = y[j]
       else: y[j+1] = -y[j]
       j+=2
    y[j] = -y[j-1]
    y = np.array(y)
    plt.xlim(-0.2, len(seq))
    plt.ylim(-1, 1)
    plt.step(x, y)
    for i in np.arange(0,len(seq)+1): plt.axvline(i,color='k',linestyle='--')
    plt.draw()
    plt.pause(0.01)
    plt.show(block = True)
def B8ZS(seq):
    plt.show()
    up = True
    zcont = 0
    x = np.arange(0, len(seq)+1)
    for i in range(len(seq)): 
        if seq[i] == 1:
            zcont = 0
            seq[i] -= 0.5
            if not up: seq[i] *=-1
            up = not up
        else: 
            zcont +=1
            if zcont == 8:
                mult = 1
                if not up: mult = -1
                seq[i]= -0.5 *mult
                seq[i-1] = 0.5 *mult
                seq[i-3] = 0.5 *mult
                seq[i-4] = -0.5 *mult
                zcont = 0
    y = np.array([0]+seq)
    plt.xlim(0, len(seq))
    plt.ylim(-1, 1)
    plt.step(x, y)
    for i in np.arange(0,len(seq)+1): plt.axvline(i,color='k',linestyle='--')
    plt.draw()
    plt.pause(0.01)
    plt.show(block = True)
def HDB3(seq):
    plt.show()
    up = True
    zcont = 0
    ucont = 0
    x = np.arange(0, len(seq)+1)
    for i in range(len(seq)): 
        if seq[i] == 1:
            zcont = 0
            seq[i] -= 0.5
            if not up: seq[i] *=-1
            up = not up
            ucont +=1
        else: 
            zcont +=1
            if zcont == 4:
                mult = 1
                if not up: mult = -1
                if ucont %2 == 0:
                    seq[i]= 0.5 *mult
                    seq[i-3]= 0.5*mult
                else: seq[i] = 0.5 * mult * -1
                zcont = 0
                ucont = 0
    y = np.array([0]+seq)
    plt.xlim(0, len(seq))
    plt.ylim(-1, 1)
    plt.step(x, y)
    for i in np.arange(0,len(seq)+1): plt.axvline(i,color='k',linestyle='--')
    plt.draw()
    plt.pause(0.01)
    plt.show(block = True)
def ASK(seq):
    plt.show()
    x = np.arange(0,len(seq)+0.01,0.01) 
    y = []
    for i in seq:
        if i== 1: y += [wave_form(0.5,1,j) for j in np.arange(0.0,1,0.01)]
        else: y += [0]*100
    y +=[0]
    y = np.array(y)
    for i in np.arange(0,len(seq)+1): plt.axvline(i,color='k',linestyle='--')
    plt.plot(x,y)
    plt.draw()
    plt.pause(0.01)
    plt.show(block = True)
def FSK(seq):
    plt.show()
    x = np.arange(0,len(seq)+0.01,0.01) 
    y = []
    for i in seq:
        if i== 1: y += [wave_form(0.5,1,j) for j in np.arange(0.0,1,0.01)]
        else: y += [wave_form(0.5,3,j) for j in np.arange(0.0,1,0.01)]
    y +=[0]
    y = np.array(y)
    for i in np.arange(0,len(seq)+1): plt.axvline(i,color='k',linestyle='--')
    plt.plot(x,y)
    plt.draw()
    plt.pause(0.01)
    plt.show(block = True)
def PSK(seq):
    plt.show()
    x = np.arange(0,len(seq)+0.01,0.01) 
    y = []
    p = 0
    for i in seq:
        if i== 1: 
            p+= math.pi
            y += [wave_form(0.5,2,j,p) for j in np.arange(0.0,1,0.01)]
        else: y += [wave_form(0.5,2,j,p) for j in np.arange(0.0,1,0.01)]
    y +=[0]
    y = np.array(y)
    for i in np.arange(0,len(seq)+1): plt.axvline(i,color='k',linestyle='--')
    plt.plot(x,y)
    plt.draw()
    plt.pause(0.01)
    plt.show(block = True)
main()