import numpy as np
'''
import mlpython.learners.generic as mlgen 
import mlpython.learners.classification as mlclass
import mlpython.mlproblems.generic as mlpb 
import mlpython.mlproblems.classification as mlpbclass
'''
import pdb

class RBM():#class RBM(mlgen.Learner):
    """
    Restricted Boltzmann Machine trained with unsupervised learning.

    Option ``lr`` is the learning rate.

    Option ``hidden_size`` is the size of the hidden layer.

    Option ``CDk`` is the number of Gibbs sampling steps used
    by contrastive divergence.

    Option ``seed`` is the seed of the random number generator.
    
    Option ``n_epochs`` number of training epochs.
    """

    def __init__(self, 
                 lr,             # learning rate
                 hidden_size,    # hidden layer size
                 CDk=4,          # nb. of Gibbs sampling steps
                 seed=1234,      # seed for random number generator
                 n_epochs=10     # nb. of training iterations
                 ):
        self.n_epochs = n_epochs
        self.hidden_size = hidden_size
        self.lr = lr
        self.CDk = CDk
        self.seed = seed
        self.rng = np.random.mtrand.RandomState(self.seed)   # create random number generator

    def train(self,trainset):
        """
        Train RBM for ``self.n_epochs`` iterations.
        """
        # Initialize parameters
        #input_size = trainset.metadata['input_size']
        input_size=len(trainset[0])

        # Parameter initialization
        self.W = (self.rng.rand(input_size,self.hidden_size)-0.5)/(max(input_size,self.hidden_size))
        self.b = np.zeros((self.hidden_size,))
        self.c = np.zeros((input_size,))
        
        self.grad_W=np.zeros(np.shape(self.W))
        self.grad_b=np.zeros(np.shape(self.b))
        self.grad_c=np.zeros(np.shape(self.c))
        
        self.px_list=[]
        
        self.h=np.zeros(self.hidden_size)
        
        for it in range(self.n_epochs):
            for input in trainset:
                # Perform CD-k
                # - you must use the matrix self.W and the bias vectors self.b and self.c

                "PUT CODE HERE"
                firstTerm, secondTerm=np.zeros(np.shape(self.W)),np.zeros(np.shape(self.W))
                ph=np.zeros(self.hidden_size)#for h(x)
                h_binary_states=np.zeros(self.hidden_size)#for h binary states               
                
                px=np.zeros(input_size)     
                x_binary_states=np.zeros(input_size)        
                #hidden layer computation           
                
                for k in range(self.CDk+1): 
                    print(px," ",h_binary_states," ",ph)
                    self.px_list.append(px)

                    '''Gibbs loop start'''                    
                    gibbsX=x_binary_states  
                    if(k==0):gibbsX=input
                    Wx=np.dot(gibbsX,self.W)
                    for i in range(len(ph)):
                        ph[i]=1/(1+np.exp(-(Wx[i]+self.b[i])))
                        if(ph[i]>np.random.rand()):
                            h_binary_states[i]=1   
                                           
                    #sampling visible layer computation
                    Wh=np.dot(self.W,h_binary_states)
                    for i in range(len(px)):                        
                        px[i]=1/(1+np.exp(-(Wh[i]+self.c[i])))
                        if(px[i]>np.random.rand()):
                            x_binary_states[i]=1                                          
                    
                    if k==0:               
                        for m in range(np.shape(firstTerm)[0]):
                            for n in range(np.shape(firstTerm)[1]):
                                firstTerm[m][n]=-input[m]*ph[n] #h[n] of gibbs sampling at k=0
                                
                        self.grad_b+=ph  
                        self.grad_c+=gibbsX
                    
                    '''Gibbs loop end'''
                for m in range(np.shape(secondTerm)[0]):
                    for n in range(np.shape(secondTerm)[1]):
                        secondTerm[m][n]=- x_binary_states[m]*ph[n]  #h[n] of gibbs sampling at k=K                
                self.grad_b-=ph                
                self.grad_c-=gibbsX
                
                self.W+=self.lr*self.grad_W
                self.b+=self.lr*self.grad_b
                self.c+=self.lr*self.grad_c                
                
                
    def show_filters(self):
        from matplotlib.pylab import show, draw, ion
        import mlpython.misc.visualize as mlvis
        mlvis.show_filters(0.5*self.W.T,
                           200,
                           16,
                           8,
                           10,20,2)
        show()
