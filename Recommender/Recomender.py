import pandas as pd
import numpy as np

#support funtion to generate testing and training sets
def assign_to_set(df):
    sampled_ids = np.random.choice(df.index,
                                   size=np.int64(np.ceil(df.index.size * 0.2)),
                                   replace=False)
    #deprecated df.ix[sampled_ids, 'for_testing'] = True
    df.loc[sampled_ids, 'for_testing'] = True
    return df

#definitions to evaluate recomender performance
def compute_rmse(y_pred, y_true):
    return np.sqrt(np.mean(np.power(y_pred - y_true, 2)))

def evaluate(estimate,test):
    ids_to_estimate = zip(test['user_id'], test['movie_id'])
    estimated = np.array([estimate(u,i) for (u,i) in ids_to_estimate])
    real = test.rating.values
    nans = np.isnan(estimated)
    return compute_rmse(estimated[~nans], real[~nans])

# Returns the euclidean distance of two vectors
def dist_euclid(x, y, ind):
    #x = x.ix[ind]
    #y = y.ix[ind]
    x = x[ind]
    y = y[ind]
    return np.sqrt(np.sum((x-y)**2))

# Returns the Pearson correlation of two vectors 
def coef_pearson(x, y,ind):
    x = x[ind]
    y = y[ind]
    #dt = pd.DataFrame({'x': x, 'y': y})
    #return dt.corr(method='pearson', min_periods=1).ix[0]['y']
    corr = np.corrcoef(x,y)[0][1]
    return corr

# Returns a distance-based similarity score for person1 and person2 based on euclidean distance
def sim_euclid(data_frame, row1, row2):
    #x = data_frame.ix[row1]
    #y = data_frame.ix[row2]
    x = data_frame.loc[row1]
    y = data_frame.loc[row2]
    #ind = [i for i in x.index if not(np.isnan(x.ix[i]) or np.isnan(y.ix[i]))] #coincidencias en ratings
    ind = [i for i in x.index if not(np.isnan(x.loc[i]) or np.isnan(y.loc[i]))] #coincidencias en ratings
    if len(ind) < 1: return 0 #no tiene sentido compararlos
    else: return 1/(1+dist_euclid(x,y,ind)) #semejanza del 0 al 1

# Returns a distance-based similarity score for person1 and person2 based on pearson distance
def sim_pearson(data_frame, row1, row2):
    #x = data_frame.ix[row1]
    #y = data_frame.ix[row2]
    x = data_frame.loc[row1]
    y = data_frame.loc[row2]
    #ind = [i for i in x.index if not(np.isnan(x.ix[i]) or np.isnan(y.ix[i]))] #coincidencias en ratings
    ind = [i for i in x.index if not(np.isnan(x.loc[i]) or np.isnan(y.loc[i]))] #coincidencias en ratings
    if len(ind) < 1: return 0 #no tiene sentido compararlos
    else:
        corr = coef_pearson(x,y,ind) #semejanza del -1 al 1 mas semejante cuanto mayor sea el valor absoluto
        if np.isnan(corr): return  1/(1+dist_euclid(x,y,ind)) #aparecen nans..
        else: return corr

# return the N most similar users to a given user based on euclidean distance
def get_best_euclid(data_frame, user, n):
    dt = data.pivot_table('rating',index= 'user_id', columns ='movie_id')
    comp = data.user_id.unique() #ids users
    maxi = [-float("inf")]*n      #maximos n valores de semejanza encontrados
    users_id = [None]*n        #ids de los n usuarios mas parecidos
    for i in range(len(comp)):
        if(user != comp[i]): #no tiene sentido comparar el usuario consigo mismo
            tmp =sim_euclid(dt,comp[i],user) #semejanza basada en distancia euclidea
            ind = maxi.index(min(maxi))    #indice del minimo de los maximos valores de semejanza encontrados 
            if(tmp> maxi[ind]):  
                     maxi[ind] = tmp
                     users_id[ind] = comp[i] #id del nuevo  usuario con  uno de los mayor valores de semejanza encontrados
    return users_id

# return the N most similar users to a given user based on pearson correlation
def get_best_pearson(data_frame, user, n):
    dt = data.pivot_table('rating',index= 'user_id', columns ='movie_id') #mismo algoritmo que con la euclidea
    comp = data.user_id.unique()
    maxi = [-float("inf")]*n
    users_id = [None]*n                  
    for i in range(len(comp)):
        if(user != comp[i]):
            tmp = abs(sim_pearson(dt,comp[i],user)) # semejanza tanto para correlacion porsitiva como negativa
            ind = maxi.index(min(maxi)) 
            if(tmp>maxi[ind]): 
                     maxi[ind] = tmp
                     users_id[ind] = comp[i]
    return users_id

class CollaborativeFiltering(object):
    """ Collaborative filtering using a custom sim(u,u'). """
    
    def __init__(self, data, M, similarity=sim_pearson):
        """ Constructor """
        self.sim_method = similarity # Gets recommendations for a person by using a weighted average
        self.df = data 
        self.sim = pd.DataFrame(0, index=M, columns=M, dtype = 'float')
        
    def precompute(self):
        """Prepare data structures for estimation. Compute similarity matrix self.sim"""
        comp = self.df.index.tolist()
        for i in range(1,len(comp)):
            for j in range(i+1,len(comp)): #evitamos comparar 2 veces el  mismo par de items/usuarios
                value = self.sim_method(self.df,comp[i],comp[j])
                self.sim[comp[i]][comp[j]] = value #llenamos la matriz de semejanzas
                self.sim[comp[j]][comp[i]] = value
#Calcula la lista de usuarios a los que se le puede estimar almenos una peli
#print [k for k in self.df.index if len([i for i in self.df.index if  len([j for j in self.df.columns if not np.isnan(self.df.ix[i][j]) and not np.isnan(self.sim[i][k]) and self.sim[i][k] != 0 and np.isnan(self.df.ix[k][j])]) != 0]) !=0]
        return 0
    
    def estimate(self, row, col):
        """ Given an row (user_id in 6, movie_id in 7) and a column (movie_id in 6, user_id in 7) 
            returns the estimated rating """
        n=10 #numero de usuarios similares para realizar la estimacion
        comp = self.df.index.tolist()
        if not row  in comp: return np.nan
        if not col  in self.df.loc[row].index.tolist(): return np.nan
        maxi = [-float("inf")]*n #maximos valores de semejanza en valor absoluto encontrados
        rows_id = [None]*n #ids items/usuarios
        for i in comp:
                if (i != row) and not np.isnan(self.df.loc[i].loc[col]): #no comparar consigo mismo y solo con los que tengan valor
                        tmp =abs(self.sim[row][i])                         #en la columna dada
                        ind = maxi.index(min(maxi)) 
                        if(tmp> maxi[ind]): 
                            maxi[ind] = tmp
                            rows_id[ind] = i
        suma = 0.0   #calculo de la media ponderada  del rating para la columna dada de los rows semejanes
        div = 0.0
        # print maxi permite observar cuando retorna nans
        for i in rows_id:
            if(i != None): #puede ser que no hubiera encontrado n maximos por falta de coincidencias
                nota = self.df.loc[i][col]
                coef = self.sim.loc[i][row]
                if(coef<0): #coef de pearson < 0 : votan de manera contraria 
                    coef = -coef
                    nota = 6 - nota
                suma += coef*nota
                div += coef
        if div == 0: return np.nan
        return suma/div
    def get_recomendations(self,rows,cols, n): #busqueda de las mejores n estimaciones
        nestim = [-float("inf")]*n 
        nind =[None] *n
        for i in rows:
            for j in cols: 
                tmp = self.estimate(i,j)
                ind = nestim.index(min(nestim))
                if(tmp> nestim[ind]):
                    nestim[ind] = tmp
                    nind[ind] = (i,j)
        return nind

class UserRecomender(CollaborativeFiltering):
    """ Recomender using Collaborative filtering with a User similarity (u,u'). """
    
    def __init__(self, data_train, similarity=sim_pearson):
        """ Constructor """
        
        # You should do any transformation to data_train (grouping/pivot/...) here, if needed
        transformed_data = data_train.pivot_table('rating',index = 'user_id',columns = 'movie_id')
        
        super(UserRecomender, self).__init__(transformed_data, data_train.user_id.unique(), similarity)

                
    def estimate(self, user_id, movie_id):
        """ Given an user_id and a movie_id returns the estimated rating for such movie """
        return super(UserRecomender, self).estimate(user_id, movie_id)
    def get_recomendations(self,user_id, n):
        if not user_id in self.df.index.tolist(): return np.nan
        cols = [i for i in self.df.columns.tolist() if np.isnan(self.df.loc[user_id][i])] #peliculas no votadas
        return [i[1] for i in super(UserRecomender,self).get_recomendations([user_id], cols,n) if i != None] #lista de ids_pelis

class ItemRecomender(CollaborativeFiltering):
    """ Recomender using Collaborative filtering with a Item similarity (i,i'). """
    
    def __init__(self,data_train, similarity=sim_pearson):
        """ Constructor """
        
        # You should do any transformation to data_train (grouping/pivot/...) here, if needed
        transformed_data = data_train.pivot_table('rating',index = 'movie_id',columns = 'user_id')
        
        super(ItemRecomender, self).__init__(transformed_data, data_train.movie_id.unique(), similarity)
      
    def estimate(self, user_id, movie_id):
        """ Given an user_id and a movie_id returns the estimated rating for such movie """
        return super(ItemRecomender, self).estimate(movie_id, user_id)
    #recomienda n pelis(no votadas) para el usuario
    def get_recomendations(self,user_id, n):
        if not user_id in self.df.columns.tolist(): return np.nan
        rows = [i for i in self.df.index.tolist() if np.isnan(self.df.loc[i][user_id])]
        return [i[0] for i in super(ItemRecomender,self).get_recomendations(rows,[user_id], n) if i != None]    #lista de ids_pelis
    
#reading data:
unames = ['user_id', 'gender', 'age', 'occupation', 'zip']
users = pd.read_table('ml-1m/users.dat', sep='::', header=None, names=unames, engine='python')
rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_table('ml-1m/ratings.dat', sep='::', header=None, names=rnames, engine='python')
mnames = ['movie_id', 'title', 'genres']
movies = pd.read_table('ml-1m/movies.dat', sep='::', header=None, names=mnames, engine='python')
data = pd.merge(pd.merge(ratings, users), movies)
data = data[data.user_id < 100]
data = data[data.movie_id < 100]

#generating training and testing fields
data['for_testing'] = False
grouped = data.groupby('user_id', group_keys=False).apply(assign_to_set)
movielens_train = data[grouped.for_testing == False]
movielens_test = data[grouped.for_testing == True]

dt = data.pivot_table('rating',index= 'user_id', columns ='movie_id')

#print results     
user_reco = UserRecomender(movielens_train)
user_reco.precompute()
print "basado en usuarios"
print "Error :"
print evaluate(user_reco.estimate, movielens_test)
print
print "las 5 pelis mas recomendadas al user 2" 
print user_reco.get_recomendations(2, 5)
print


item_reco = ItemRecomender(movielens_train)
item_reco.precompute()
print "basado en items"
print "Error :"
print evaluate(item_reco.estimate, movielens_test)
print "las 5 pelis mas recomendadas al user 2" 
print item_reco.get_recomendations(2, 5)