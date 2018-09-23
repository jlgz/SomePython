 #%reset -f
#load data
from nltk.stem.porter import PorterStemmer
import pandas as pd
import numpy as np
import string
import math
#Retorna el número total de notícies. 
def count_news(df):
    return len(df.index)
#Retorna una Series que compte el número de notícies per a cadascun dels tópics
def count_topic_news(df):
    return df.groupby("Topic_2digit").count()["Title"]
#stemming
# Aquesta funció ha de construir un diccionari que contingui totes les paraules que s'han trobat indicant 
# el total de cops que ha aparegut i el nombre de notícies on apareix
def lower_no_point(s): 
    s=s.lower()
    stmp = ''
    for i in s: 
        if i not in string.punctuation:stmp+= i
    return stmp

def count_words(df):
    porter_stemmer = PorterStemmer()
    word_dicc=dict()
    for i in df.index:
        st = df.loc[i]["Title"]
        st = st.split()
        ss =  df.loc[i]["Summary"]
        st = st + ss.split()
        for i in set(st):
            k = lower_no_point(i) #quitar signos de puntuacion
            try: k=porter_stemmer.stem(k) #stemming
            except UnicodeDecodeError: None
            if k in word_dicc:
                word_dicc[k]["n_ocur"] += st.count(i)
                word_dicc[k]["n_news"] += 1
            else: 
                word_dicc[k] = {"n_ocur": st.count(i),"n_news": 1}
    return word_dicc
#Compta la freqüència de les paraules per a un tòpic determinat
def count_words_topic(df):
    dictop = dict()
    for i in df["Topic_2digit"].unique():
        dictop[i] = count_words(df[df.Topic_2digit == i]) 
    return dictop
#Calcula les N parules més representativa de cada tòpic . La sortida ha de 
# ser un diccionari on tenim tantes entrades com tòpics
# el valors de les entrades ha de ser una llista amb les paraules seleccionades.
def basic_words_d():
    l = basic_words=['about','after','again','air','all','along','also','an','and','another','any','are','around','as','at','away','back','be','because','been','before','below','between','both','but','by','came','can','come','could','day','did','different','do','does','down','each','end','even','every','few','find','first','for','found','from','get','give','go','good','great','had','has','have','he','help','her','here','him','his','home','house','how','I','if','in','into','is','it','its','just','know','large','last','left','like','line','little','long','look','made','make','man','many','may','me','men','might','more','most','Mr.','must','my','name','never','new','next','no','not','now','number','of','off','old','on','one','only','or','other','our','out','over','own','part','people','place','put','read','right','said','same','saw','say','see','she','should','show','small','so','some','something','sound','still','such','take','tell','than','that','the','them','then','there','these','they','thing','think','this','those','thought','three','through','time','to','together','too','two','under','up','us','use','very','want','water','way','we','well','went','were','what','when','where','which','while','who','why','will','with','word','work','world','would','write','year','you','your','was']
    d = dict()
    for i in l: d[i] = i
    return d
def topNwords(df,words_topics,N):
    bw = basic_words_d()
    topcont = count_topic_news(data)
    fita = 0.1
    top_words=dict()
    for i in words_topics:
        l =['0']*N
        f = [0] *N  
        for j in words_topics[i]:
            ntopic = words_topics[i][j]["n_ocur"]                  
            ntopicnews = words_topics[i][j]["n_news"]
            #ntotal = dicc_text[j]["n_ocur"]
            #ntotalnews = dicc_text[j]["n_news"]
            nnewstop = topcont.loc[i]
            freq_n = float(ntopicnews) / nnewstop
            #freq_t = float(ntotalnews) / len(df.index)
            ind = f.index(min(f))                
            if(j not in bw and freq_n> f[ind]): #falta probar añadir al if 0.1<ft <0.5
                    l[ind] = j              #sino usamos esto mejor no usar el dicc_text
                    f[ind] = freq_n
        top_words[i] = l
    return top_words
# Crea el vector de característiques necessari per a l'entrenament del classificador Naive Bayes
# selected_words: ha de ser el diccionari que retorna topNWords.
# train_data : conté totes les notícies del conjunt d'entrenament
# Rertorna un diccionari que conté un np.array per a cadascuna de les notícies amb el vector de característiques corresponent.
def create_features(train_data,selected_words):
    porter_stemmer = PorterStemmer()
    dict_feat_vector=dict()
    words = []
    for i in selected_words:
        words+= selected_words[i]
    words = list(set(words))
    dict_feat_vector["clasific"] = words
    for i in train_data.index:
        car_v = [0]* len(words)
        st=train_data.loc[i]["Title"]
        st = st.split()
        ss =  train_data.loc[i]["Summary"]
        st = st + ss.split()
        for j in set(st):
            k = lower_no_point(j)
            try: k=porter_stemmer.stem(k)
            except UnicodeDecodeError: None
            if k in words:
                car_v[words.index(k)] = 1 #st.count(j) no boleana
        dict_feat_vector[i] = np.array(car_v)
    return dict_feat_vector
#Mètode que implementa el clasificador Naive_Bayes.Ha de mostrar el resultat obtingut per pantalla
def naive_bayes(df,topcont,feature_vector, topic_words):
    topcont2 = count_topic_news(df)
    tok =sorted(df["Topic_2digit"].unique())
    d = dict()
    error = 0
    sizecl =len(feature_vector['clasific'])
    for k in tok:
        p = [0]*sizecl
        for j in  range(sizecl):
            if feature_vector['clasific'][j] in topic_words[k]: f = (topic_words[k][feature_vector['clasific'][j]]["n_news"] +1)/float(topcont.loc[k]+len(topcont) )  
            else: f = 1.0 / (topcont.loc[k] + len(topcont))
            p[j] = [math.log(1-f),math.log(f)]
        d[k] = p
    for s in tok:
        aciertos = 0
        for i in df[df.Topic_2digit == s].index.tolist():
            t = [0]*len(topcont)
            l = [0]*len(topcont)
            cont = 0
            for k in tok:
                suma = 0
                for j in  range(sizecl): #probabilidades con correccion de laplace
                        suma += d[k][j][feature_vector[i][j]]#suma de log para evitar underflow
                l[cont] = suma
                t[cont] = k
                cont += 1
            if t[l.index(max(l))] == s: aciertos += 1
            else: error += 1
        print 'topico ', s, ' aciertos: ', aciertos, ' / Noticias: ', topcont2.loc[s], " / ", 100 * aciertos / topcont2.loc[s], " %"    
    print 'error naive-bayes ', float(error) / df.index.size * 100, '%'
    return 0
def nfold_asign(df,n,i,s):
    sampled_ids = np.random.choice(df[df.fold==0].index,
                                   size=s,
                                   replace=False)
    df.loc[sampled_ids, 'fold'] = i
    return df

#Mètode per avaluar el classificador mitjançant la tècnica n-fold validation. n és el nombre de folds  
#Ha de mostrar per pantalla el resultat obtingut 
def n_fold(df,feature_vector,n): #aqui no se si habria que definir otro vector de caracteristicas para cada iteracion
        df['fold'] = 0          #de conjunto de entrenamiento
        sizef = int(float(df.index.size) / n)
        for i in range(n):
            df = nfold_asign(df,n,i+1,sizef)
        for i in range(n):
            df_train = df[df.fold != i +1]
            df_test = df[df.fold == i+1]
            words_topics=count_words_topic(df_train)
            print 'fold :', i+1
            naive_bayes(df_test,count_topic_news(df_train),feature_vector, words_topics)
        return 0
# Main. Es criden a totes les funcions per a la correcte execució del programa.   
def main():
    data=pd.read_csv('Boydstun_NYT_FrontPage_Dataset_1996-2006_0.csv')
    #dicc_text = count_words(data)
    N = 20 # Aquest parametre el podem canviar i fer proves per avaluar quin és el millor valor. 
    words_topics=count_words_topic(data)
    top_words=topNwords(data,words_topics,N)
    feature_vectors = create_features(data,top_words) 
    naive_bayes(data,count_topic_news(data),feature_vectors,words_topics) #fins aquí error d'entrenament. Fem servir totes les dades.
    n_fold(data, feature_vectors,3)
data=pd.read_csv('Boydstun_NYT_FrontPage_Dataset_1996-2006_0.csv',index_col="Article_ID")
main()