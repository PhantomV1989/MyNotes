
from mlpython.learners.generic import Learner
import numpy as np
import pdb

class LinearChainCRF(Learner):
    """
    Linear chain conditional random field. The contex window size
    has a radius of 1.
 
    Option ``lr`` is the learning rate.
 
    Option ``dc`` is the decrease constante for the learning rate.
 
    Option ``L2`` is the L2 regularization weight (weight decay).
 
    Option ``L1`` is the L1 regularization weight (weight decay).
 
    Option ``n_epochs`` number of training epochs.
 
    **Required metadata:**
 
    * ``'input_size'``: Size of the input.
    * ``'targets'``:    Set of possible targets.
 
    """
    
    def __init__(self,
                 lr=0.008,
                 dc=1e-10,
                 L2=0.001,
                 L1=0,
                 n_epochs=10):
        self.lr=lr
        self.dc=dc
        self.L2=L2
        self.L1=L1
        self.n_epochs=n_epochs

        # internal variable keeping track of the number of training iterations since initialization
        self.epoch = 0 

    def initialize(self,input_size,n_classes):
        """
        This method allocates memory for the fprop/bprop computations
        and initializes the parameters of the CRF to 0 (DONE)
        """

        self.n_classes = n_classes
        self.input_size = input_size

        # Can't allocate space for the alpha/beta tables of
        # belief propagation (forward-backward), since their size
        # depends on the input sequence size, which will change from
        # one example to another.

        self.alpha = np.zeros((0,0))
        self.beta = np.zeros((0,0))
        
        ###########################################
        # Allocate space for the linear chain CRF #
        ###########################################
        # - self.weights[0] are the connections with the image at the current position
        # - self.weights[-1] are the connections with the image on the left of the current position
        # - self.weights[1] are the connections with the image on the right of the current position
        self.weights = [np.zeros((self.input_size,self.n_classes)),
                        np.zeros((self.input_size,self.n_classes)),
                        np.zeros((self.input_size,self.n_classes))]
        # - self.bias is the bias vector of the output at the current position
        self.bias = np.zeros((self.n_classes))

        # - self.lateral_weights are the linear chain connections between target at adjacent positions
        self.lateral_weights = np.zeros((self.n_classes,self.n_classes))
        
        self.reset()
        
                    
        #########################
        # Initialize parameters #
        #########################

        # Since the CRF log factors are linear in the parameters,
        # the optimization is convex and there's no need to use a random
        # initialization.

        self.n_updates = 0 # To keep track of the number of updates, to decrease the learning rate
        
        self.randomizeValues()
        #self.initCustom()
        
        

    def forget(self):
        """
        Resets the neural network to its original state (DONE)
        """
        self.initialize(self.input_size,self.n_classes)
        self.epoch = 0
        
    def train(self,trainset):
        """
        Trains the neural network until it reaches a total number of
        training epochs of ``self.n_epochs`` since it was
        initialize. (DONE)

        Field ``self.epoch`` keeps track of the number of training
        epochs since initialization, so training continues until 
        ``self.epoch == self.n_epochs``.
        
        If ``self.epoch == 0``, first initialize the model.
        """
        
        

 
        
        for it in range(self.epoch,self.n_epochs):
            print("Epoch:", it)
            for input,target in trainset:  
                self.fprop(input)
                self.training_loss(target)  
                self.bprop(input,target)
                self.update()
               
        self.epoch = self.n_epochs
        
        
    def fprop(self,input):
        """
        Forward propagation: 
        - computes the value of the unary log factors for the target given the input (the field
          self.target_unary_log_factors should be assigned accordingly)
        - computes the alpha and beta tables using the belief propagation (forward-backward) 
          algorithm for linear chain CRF (the field ``self.alpha`` and ``self.beta`` 
          should be allocated and filled accordingly)
        - returns the training loss, i.e. the 
          regularized negative log-likelihood for this (``input``,``target``) pair
        Argument ``input`` is a Numpy 2D array where the number of
        rows if the sequence size and the number of columns is the
        input size. 
        Argument ``target`` is a Numpy 1D array of integers between 
        0 and nb. of classe - 1. Its size is the same as the number of
        rows of argument ``input``.
        """

        ## PUT CODE HERE ##
        # (your code should call belief_propagation and training_loss)
        """             1                """#ToBeChecked manually with random W
        lastIndex=len(input)-1
        self.target_unary_log_factors=np.zeros([len(input),self.n_classes])              
        for k in range(lastIndex+1):               
            if (k==0):  #first one
                for i in range(0,2): 
                    self.target_unary_log_factors[k:k+1]+=(np.dot(input[k+i],self.weights[i])+self.bias)                  
            elif (k==len(input)-1):#last one                
                for i in range(-1,1):                
                    self.target_unary_log_factors[k:k+1]+=(np.dot(input[k+i],self.weights[i])+self.bias)                 
            else:
                for i in range(-1,2):                                     
                    self.target_unary_log_factors[k:k+1]+=(np.dot(input[k+i],self.weights[i])+self.bias) 
        
        """             2                """      
        self.belief_propagation(input)#fills alpha beta tables        
        self.belief_propagation_log_space(input)
        """             3                """ 
       
            
        
        #raise NotImplementedError()

    def belief_propagation(self,input):
        """
        Returns the alpha/beta tables (i.e. the factor messages) using
        belief propagation (which is equivalent to forward-backward in HMMs).
        """

        ## PUT CODE HERE ##        
        lastIndex=len(input)-1
        self.alpha = np.zeros((len(input),self.n_classes))
        self.beta = np.zeros((len(input),self.n_classes))        
        """       Alpha table        """   
        for k in range(lastIndex):            
            for kP1Class in range(self.n_classes):                               
                for kClass in range(self.n_classes):
                    if(k==0):                        
                        self.alpha[k,kP1Class]+=np.exp(self.target_unary_log_factors[k,kClass]+self.lateral_weights[kClass,kP1Class])                       
                    else:                                       
                        self.alpha[k,kP1Class]+=np.exp(self.target_unary_log_factors[k,kClass]+self.lateral_weights[kClass,kP1Class])* self.alpha[k-1,kClass]                           
        for lastClass in range(self.n_classes):            
            self.alpha[-1,lastClass]=np.exp(self.target_unary_log_factors[-1,lastClass])* self.alpha[-2,lastClass]
        #Z(X) for alpha is sum of last col,np.sum(myObj.alpha[-1:])
        #alpha table checked
      
        """       Beta table        """    
        for k in range(lastIndex,0,-1):            
            for kM1Class in range(self.n_classes):
                for kClass in range(self.n_classes):                   
                    if(k==lastIndex):                        
                        self.beta[k,kM1Class]+=np.exp(self.target_unary_log_factors[k,kClass]+self.lateral_weights[kM1Class,kClass])                    
                    else:                 
                        self.beta[k,kM1Class]+=np.exp(self.target_unary_log_factors[k,kClass]+self.lateral_weights[kM1Class,kClass])*self.beta[k+1,kClass]    
        for firstClass in range(self.n_classes):
            self.beta[0,firstClass]=np.exp(self.target_unary_log_factors[0,firstClass])*self.beta[1,firstClass] 
        
        #Z(X for beta is sum of 1st col), np.sum(myObj.beta[:1])
        #beta table checked
        #raise NotImplementedError()
        
    def belief_propagation_log_space(self,input):
        """
        Returns the alpha/beta tables (i.e. the factor messages) using
        belief propagation (which is equivalent to forward-backward in HMMs).
        """
        ## PUT CODE HERE ##     
        lastIndex=len(input)-1
        self.log_alpha = np.zeros((len(input),self.n_classes))
        self.log_beta = np.zeros((len(input),self.n_classes))
        """       Alpha table        """        
        for k in range(lastIndex+1):            
            for kP1Class in range(self.n_classes):
                tempHolder=np.zeros([self.n_classes])
                if(k==lastIndex):
                    self.log_alpha[k,kP1Class]=self.target_unary_log_factors[k,kP1Class]+self.log_alpha[k-1,kP1Class]                    
                    continue
                for kClass in range(self.n_classes):                   
                    if(k==0):
                        tempHolder[kClass]=self.target_unary_log_factors[k,kClass]+self.lateral_weights[kClass,kP1Class]                        
                    elif(k<lastIndex):
                        tempHolder[kClass]=self.target_unary_log_factors[k,kClass]+self.lateral_weights[kClass,kP1Class]+self.log_alpha[k-1,kClass]                        
                    if(kClass<self.n_classes):
                        sumX=0;                        
                        for i in range(len(tempHolder)):
                            sumX+=(np.exp(tempHolder[i]-max(tempHolder)))
                        self.log_alpha[k,kP1Class]=max(tempHolder)+np.log(sumX)
                                        
                      
            
        
        #Z(X) for alpha is sum of last col
       #log_alpha table checked         
        """       Beta table        """    
        for k in range(lastIndex,-1,-1):            
            for kM1Class in range(self.n_classes):
                tempHolder=np.zeros([self.n_classes])
                if(k==0):
                    self.log_beta[k,kM1Class]=self.target_unary_log_factors[k,kM1Class]+self.log_beta[k+1,kM1Class]
                    continue
                for kClass in range(self.n_classes):
                    if(k==lastIndex):
                        tempHolder[kClass]=self.target_unary_log_factors[k,kClass]+self.lateral_weights[kM1Class,kClass] 
                    elif(k>0):
                        tempHolder[kClass]=self.target_unary_log_factors[k,kClass]+self.lateral_weights[kM1Class,kClass]+self.log_beta[k+1,kClass]
                    if(kClass>0):                        
                        sumX=0;
                        for i in range(len(tempHolder)):
                            sumX+=(np.exp(tempHolder[i]-max(tempHolder)))                       
                        self.log_beta[k,kM1Class]=max(tempHolder)+np.log(sumX)                
                   
        #Z(X for beta is sum of 1st col)
        #beta table checked
      

        #raise NotImplementedError()
       
    
    
    def training_loss(self,target):
        """
        Computation of the loss:
        - returns the regularized negative log-likelihood loss associated with the
          given the true target, the unary log factors of the target space and alpha/beta tables
        """

        ## PUT CODE HERE ##
        #loss = -log(p(y|X))
        #where p(y|X)
        self.pairwise_log_factors=0
        self.unarySum,self.pairwiseSum=0,0
        for k in range(len(target)):
            self.unarySum+=self.target_unary_log_factors[k,target[k]]          
            if(k<len(target)-1):
                self.pairwiseSum+=self.lateral_weights[target[k],target[k+1]]              
        loss=np.exp(self.unarySum+self.pairwiseSum)/np.sum(self.alpha[-1:]) #add regularizer        
        loss=-np.log(loss)  
        self.loss=loss
        print("loss:",loss)
        return loss

        #raise NotImplementedError()

    def bprop(self,input,target):
        """
        Backpropagation:
        - fills in the CRF gradients of the weights, lateral weights and bias 
          in self.grad_weights, self.grad_lateral_weights and self.grad_bias
        - returns nothing
        Argument ``input`` is a Numpy 2D array where the number of
        rows if the sequence size and the number of columns is the
        input size. 
        Argument ``target`` is a Numpy 1D array of integers between 
        0 and nb. of classe - 1. Its size is the same as the number of
        rows of argument ``input``.
        """

        ## PUT CODE HERE ##
        self.reset()
        self.pykX=np.zeros([len(input),self.n_classes])
        self.pykykP1X=np.zeros([len(input)-1,self.n_classes,self.n_classes])
        uLF=self.target_unary_log_factors
        lW=self.lateral_weights
        self.intermediateGrad=np.zeros([len(input),self.n_classes])
        self.marginalPykArray,self.marginalPykykP1Array=np.zeros([len(input),self.n_classes]),np.zeros([len(input),self.n_classes])
        ''' For unary (marginal table checked)'''
        for k in range(len(input)):
            denom=0            
            def eachMarginalNominator():return 0
            if(k==0):#first one must exclude log_alpha 
                eachMarginalNominator=lambda i: np.exp(uLF[k,i]+self.log_beta[k+1,i])
            elif(k==(len(input)-1)):#last one must exclude log_beta
                eachMarginalNominator=lambda i:np.exp(uLF[k,i]+self.log_alpha[k-1,i])  
            else:
                eachMarginalNominator=lambda i:np.exp(uLF[k,i]+self.log_alpha[k-1,i]+self.log_beta[k+1,i])
            for i in range(self.n_classes):#fill Z(X) first   
                self.pykX[k,i]=eachMarginalNominator(i)
                denom+=eachMarginalNominator(i)             
            self.pykX[k]=self.pykX[k]/denom   
            '''fills up intermediate values'''
            for classNo in range(self.n_classes):
                trueClass=target[k]
                if(trueClass==classNo):
                    self.intermediateGrad[k,classNo]=-(1-self.pykX[k][classNo])
                else:
                    self.intermediateGrad[k,classNo]=-(0-self.pykX[k][classNo])  
        
        '''For pairwise (joint marginal table checked)'''
        for k in range(len(input)-1):
            denomP=0
            def eachPairwiseMarginalNominator():return 0
            if(k==0):#1st one excludes log_alpha
                eachPairwiseMarginalNominator=lambda i,j:np.exp(uLF[k,i]+lW[i,j]+uLF[k+1,j]+self.log_beta[k+2,j]) 
            elif(k==(len(input)-2)):#last one exludes log_beta, pairwise sums until K-2     
                eachPairwiseMarginalNominator=lambda i,j:np.exp(uLF[k,i]+lW[i,j]+uLF[k+1,j]+self.log_alpha[k-1,i])
            else:                
                eachPairwiseMarginalNominator=lambda i,j:np.exp(uLF[k,i]+lW[i,j]+uLF[k+1,j]+self.log_alpha[k-1,i]+self.log_beta[k+2,j])
            for i in range(self.n_classes):#for each possible class at position k
                for j in range(self.n_classes):                                      
                    self.pykykP1X[k,i,j]=eachPairwiseMarginalNominator(i,j)
                    denomP+=eachPairwiseMarginalNominator(i,j)
            self.pykykP1X[k]=self.pykykP1X[k]/denomP
                         
 
        
        '''Fills up grad_weights'''
        for k in range(len(input)):#this one for summing over K values            
            for classNo in range(self.n_classes):                
                self.grad_bias[classNo]+=self.intermediateGrad[k,classNo]
                for n in range(self.input_size):#this one for running through each position k in input                                    
                        if(k>0):self.grad_weights[-1][n,classNo]+=np.dot(self.intermediateGrad[k,classNo],input[k-1][n])#self.n_classes x len(input)
                        self.grad_weights[0][n,classNo]+=np.dot(self.intermediateGrad[k,classNo],input[k][n])#self.n_classes x len(input)
                        if(k<len(input)-1):self.grad_weights[1][n,classNo]+=np.dot(self.intermediateGrad[k,classNo],input[k+1][n])#self.n_classes x len(input)
        
        '''Fills up grad_lateral_weights'''
        pairTruthTable=np.zeros([self.n_classes,self.n_classes])
        pairPredictedTable=np.zeros([self.n_classes,self.n_classes])
        for k in range(len(input)-1):
            pairTruthTable[target[k],target[k+1]]+=1            
            pairPredictedTable+=self.pykykP1X[k]
        self.grad_lateral_weights=-(pairTruthTable-pairPredictedTable)
        #raise NotImplementedError()

    def update(self):
        """
        Stochastic gradient update:
        - performs a gradient step update of the CRF parameters self.weights,
          self.lateral_weights and self.bias, using the gradients in 
          self.grad_weights, self.grad_lateral_weights and self.grad_bias
        """

        ## PUT CODE HERE ##
        self.bias=np.add(self.bias,-self.grad_bias*self.lr)       
        self.weights=np.add(np.add(self.weights,-np.multiply(self.grad_weights,self.lr)),-np.multiply(self.L2_grad(),self.L2))
        self.lateral_weights=np.add(self.lateral_weights,-np.multiply(self.grad_lateral_weights,self.lr))

        #raise NotImplementedError()
           
    def use(self,dataset):
        """
        Computes and returns the outputs of the Learner for
        ``dataset``:
        - the outputs should be a list of size ``len(dataset)``, containing
          a Numpy 1D array that gives the class prediction for each position
          in the sequence, for each input sequence in ``dataset``
        Argument ``dataset`` is an MLProblem object.
        """
         ## PUT CODE HERE ##    
        self.useOutput=np.zeros([len(dataset),len(dataset[0])])
        self.finalTarget=np.zeros([len(dataset),len(dataset[0]),self.n_classes])
        self.useConfidence=np.zeros([len(dataset)])
        for i in range(len(dataset)):            
            self.fprop(dataset[i])            
            for k in range(len(self.target_unary_log_factors)):#length of input i
                if (k==0):                    
                    self.finalTarget[i,k]=self.target_unary_log_factors[k]
                    self.useOutput[i,k]=self.getClass(self.target_unary_log_factors[k]) #only first one no need previous class
                else:                    
                    self.finalTarget[i,k]=self.target_unary_log_factors[k]+self.lateral_weights[int(self.useOutput[i,k-1])]                    
                    self.useOutput[i,k]=self.getClass(self.target_unary_log_factors[k]+self.lateral_weights[int(self.useOutput[i,k-1])])#use pairwise with previous class output[i,k-1]
            
            ''' Calculating Confidence'''
            tempUnarySum,tempPairwiseSum=0,0
            for k in range(len(self.useOutput)):                
                tempUnarySum+=self.finalTarget[0][int(self.useOutput[i][k])]  
                if(k<len(self.useOutput)-1):
                    tempPairwiseSum+=self.lateral_weights[int(self.useOutput[i][k]),int(self.useOutput[i][k+1])]             
            confidence=np.exp(self.unarySum+self.pairwiseSum)/np.sum(self.alpha[-1:]) #add regularizer        
            self.useConfidence[i]=(confidence)
          

    
       
        
        #raise NotImplementedError()
            
        #return output
        
    def test(self,dataset):
        """
        Computes and returns the outputs of the Learner as well as the errors of the
        CRF for ``dataset``:
        - the errors should be a list of size ``len(dataset)``, containing
          a pair ``(classif_errors,nll)`` for each examples in ``dataset``, where 
            - ``classif_errors`` is a Numpy 1D array that gives the class prediction error 
              (0/1) at each position in the sequence
            - ``nll`` is a positive float giving the regularized negative log-likelihood of the target given
              the input sequence
        Argument ``dataset`` is an MLProblem object.
        """
        
        outputs = self.use(dataset)
        errors = []

        ## PUT CODE HERE ##
        raise NotImplementedError()
            
        return outputs, errors
 
 
    def getClass(self,output):
        maxP=0
        classX=0
        for i in range(len(output)):
            if (output[i]>maxP):
                maxP=output[i]
                classX=i
        return classX
        
    
    def verify_gradients(self):
        """
        Verifies the implementation of the fprop and bprop methods
        using a comparison with a finite difference approximation of
        the gradients.
        
        Note:
            bias gradient need to recheck
        """
        
        print ('WARNING: calling verify_gradients reinitializes the learner')
  
        rng = np.random.mtrand.RandomState(1234)
  
        self.initialize(10,3)
        example = (rng.rand(4,10),np.array([0,1,1,2]))
        input,target = example
        epsilon=1e-6
        self.lr = 0.1
        self.decrease_constant = 0

        self.weights = [0.01*rng.rand(self.input_size,self.n_classes),
                        0.01*rng.rand(self.input_size,self.n_classes),
                        0.01*rng.rand(self.input_size,self.n_classes)]
        self.bias = 0.01*rng.rand(self.n_classes)
        self.lateral_weights = 0.01*rng.rand(self.n_classes,self.n_classes)
        
        self.fprop(input,target)
        self.bprop(input,target) # compute gradients

        import copy
        emp_grad_weights = copy.deepcopy(self.weights)
  
        for h in range(len(self.weights)):
            for i in range(self.weights[h].shape[0]):
                for j in range(self.weights[h].shape[1]):
                    self.weights[h][i,j] += epsilon
                    a = self.fprop(input,target)
                    self.weights[h][i,j] -= epsilon
                    
                    self.weights[h][i,j] -= epsilon
                    b = self.fprop(input,target)
                    self.weights[h][i,j] += epsilon
                    
                    emp_grad_weights[h][i,j] = (a-b)/(2.*epsilon)


        print ('grad_weights[-1] diff.:',np.sum(np.abs(self.grad_weights[-1].ravel()-emp_grad_weights[-1].ravel()))/self.weights[-1].ravel().shape[0])
        print ('grad_weights[0] diff.:',np.sum(np.abs(self.grad_weights[0].ravel()-emp_grad_weights[0].ravel()))/self.weights[0].ravel().shape[0])
        print ('grad_weights[1] diff.:',np.sum(np.abs(self.grad_weights[1].ravel()-emp_grad_weights[1].ravel()))/self.weights[1].ravel().shape[0])
  
        emp_grad_lateral_weights = copy.deepcopy(self.lateral_weights)
  
        for i in range(self.lateral_weights.shape[0]):
            for j in range(self.lateral_weights.shape[1]):
                self.lateral_weights[i,j] += epsilon
                a = self.fprop(input,target)
                self.lateral_weights[i,j] -= epsilon

                self.lateral_weights[i,j] -= epsilon
                b = self.fprop(input,target)
                self.lateral_weights[i,j] += epsilon
                
                emp_grad_lateral_weights[i,j] = (a-b)/(2.*epsilon)


        print ('grad_lateral_weights diff.:',np.sum(np.abs(self.grad_lateral_weights.ravel()-emp_grad_lateral_weights.ravel()))/self.lateral_weights.ravel().shape[0])

        emp_grad_bias = copy.deepcopy(self.bias)
        for i in range(self.bias.shape[0]):
            self.bias[i] += epsilon
            a = self.fprop(input,target)
            self.bias[i] -= epsilon
            
            self.bias[i] -= epsilon
            b = self.fprop(input,target)
            self.bias[i] += epsilon
            
            emp_grad_bias[i] = (a-b)/(2.*epsilon)

        print ('grad_bias diff.:',np.sum(np.abs(self.grad_bias.ravel()-emp_grad_bias.ravel()))/self.bias.ravel().shape[0])


    
    def L2_regularizer(self):
        return np.sum(np.square(self.weights))
    
    def L2_grad(self):
        return np.multiply(2,self.weights)
        
      
            
    def reset(self):
        self.grad_weights = [np.zeros((self.input_size,self.n_classes)),
                        np.zeros((self.input_size,self.n_classes)),
                        np.zeros((self.input_size,self.n_classes))]
        self.grad_bias = np.zeros((self.n_classes))
        self.grad_lateral_weights = np.zeros((self.n_classes,self.n_classes))
       
    
    def randomizeValues(self):
        rng = np.random.mtrand.RandomState(1234)
        self.weights = [0.01*rng.rand(self.input_size,self.n_classes),
                        0.01*rng.rand(self.input_size,self.n_classes),
                        0.01*rng.rand(self.input_size,self.n_classes)]
        self.bias = 0.01*rng.rand(self.n_classes)
        self.lateral_weights = 0.01*rng.rand(self.n_classes,self.n_classes)
        
    def onesAll(self):
        self.weights=[np.ones([self.input_size,self.n_classes]),
                      np.ones([self.input_size,self.n_classes]),
                      np.ones([self.input_size,self.n_classes])]      
        #self.bias=np.ones(self.n_classes)
        self.lateral_weights = np.ones([self.n_classes,self.n_classes])

    def initCustom(self):
        self.weights[0]=np.array([[1,4],[2,5],[3,6]])
        self.weights[1]=np.array([[7,10],[8,11],[9,12]])
        self.weights[-1]=np.array([[13,16],[14,17],[15,18]])
        
        self.bias[0]=0.3
        self.bias[1]=0.5
                 
        self.lateral_weights[0,0]=1.5#along k=0,k+1=0
        self.lateral_weights[1,0]=2.5#along k=1,k+1=0
        self.lateral_weights[0,1]=3.5
        self.lateral_weights[1,1]=4.5
                            
                            
                            
                            