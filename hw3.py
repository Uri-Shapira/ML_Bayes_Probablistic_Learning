import numpy as np
np.random.seed(42)

####################################################################################################
#                                            Part A
####################################################################################################
def split_dataset(dataset, class_value):
    class_value_dataset = []
    for row in dataset:
        if row[-1] == class_value:
            class_value_dataset.append(row)
    return np.asarray(class_value_dataset)   

def get_mu_vector(class_value_dataset):
    mean_vector = []
    temperature_mean = np.mean(class_value_dataset[:,0])
    humidity_mean = np.mean(class_value_dataset[:,1])
    mean_vector.append(temperature_mean)
    mean_vector.append(humidity_mean)
    return mean_vector

def get_prior(dataset, class_value):
    count = 0
    length = len(dataset)
    for row in dataset:
        if row[-1] == class_value:
            count += 1
    return count / length

def get_instance_posterior(prior, likelihood):
    return prior * likelihood

class NaiveNormalClassDistribution():

    def __init__(self, dataset, class_value):
        """
        A class which encapsulate the relevant parameters(mean, std) for a class conditinoal normal distribution.
        The mean and std are computed from a given data set.
        
        Input
        - dataset: The dataset from which to compute the mean and mu (Numpy Array).
        - class_value : The class to calculate the mean and mu for.
        """
        self.dataset = dataset
        self.class_value = class_value
        self.class_value_dataset = split_dataset(dataset, class_value)
        self.mu_vector = get_mu_vector(self.class_value_dataset)
        self.sigma_vector = self.get_sigma_vector()
                                                                                             
    def get_sigma_vector(self):
        sigma_vector = []
        temperature_sigma = np.std(self.class_value_dataset[:,0])
        humidity_sigma = np.std(self.class_value_dataset[:,1])
        sigma_vector.append(temperature_sigma)
        sigma_vector.append(humidity_sigma)
        return sigma_vector
    
    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        return get_prior(self.dataset, self.class_value)
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihhod porbability of the instance under the class according to the dataset distribution.
        """
        instance_likelihood = 1
        for i in range(2):
            likelihood = normal_pdf(x[i], self.mu_vector[i], self.sigma_vector[i])
            instance_likelihood *= likelihood
        return instance_likelihood
    
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        return get_instance_posterior(self.get_prior(), self.get_instance_likelihood(x))

    
class MultiNormalClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which encapsulate the relevant parameters(mean, cov matrix) for a class conditinoal multi normal distribution.
        The mean and cov matrix (You can use np.cov for this!) will be computed from a given data set.
        
        Input
        - dataset: The dataset from which to compute the mean and mu (Numpy Array).
        - class_value : The class to calculate the mean and mu for.
        """
        self.dataset = dataset
        self.class_value = class_value
        self.class_value_dataset = split_dataset(dataset, class_value)
        self.mu_vector = get_mu_vector(self.class_value_dataset)
        self.cov = self.get_cov_matrix()
        
    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        return get_prior(self.dataset, self.class_value)
    
    def get_cov_matrix(self):
        transposed_matrix = (np.transpose(np.delete(self.class_value_dataset, 2, 1)))
        return np.cov(transposed_matrix)
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihhod porbability of the instance under the class according to the dataset distribution.
        """
        likelihood = multi_normal_pdf(x, self.mu_vector, self.cov)
        return likelihood  
    
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        return get_instance_posterior(self.get_prior(), self.get_instance_likelihood(x))
       
def normal_pdf(x, mean, std):
    """
    Calculate normal density function for a given x, mean and standrad deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean value of the distribution.
    - std:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mean and var for the given x.    
    """
    denominator = np.sqrt(2 * np.pi * np.square(std))
    first_part = 1 / denominator
    exponent = -(np.square(x - mean)) / (2*np.square(std))
    second_part = np.exp(exponent)
    normal_pdf = first_part * second_part
    return normal_pdf
    
def multi_normal_pdf(x, mean, cov):
    """
    Calculate multi variante normal density function for a given x, mean and covarince matrix.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean value of the distribution.
    - std:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mean and var for the given x.    
    """
    x = np.delete(x,2)
    first_part = np.power(np.square(2*np.pi)*np.square(np.linalg.det(cov)), -1/2)
    matrix_mult1 = np.matmul(np.transpose(x - mean) , np.linalg.inv(cov))
    matrix_mult2 = np.matmul(matrix_mult1, x - mean)
    exponent = (-1/2) * matrix_mult2
    multi_normal_pdf = first_part * np.exp(exponent)
    return multi_normal_pdf


####################################################################################################
#                                            Part B
####################################################################################################
EPSILLON = 1e-6 # == 0.000001 It could happen that a certain value will only occur in the test set.
                # In case such a thing occur the probability for that value will EPSILLON.

class DiscreteNBClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which computes and encapsulate the relevant probabilites for a discrete naive bayes 
        distribution for a specific class. The probabilites are computed with la place smoothing.
        
        Input
        - dataset: The dataset from which to compute the probabilites (Numpy Array).
        - class_value : Compute the relevant parameters only for instances from the given class.
        """
        self.dataset = dataset
        self.class_value = class_value
        self.class_value_dataset = split_dataset(dataset, class_value)
      
    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        return get_prior(self.dataset, self.class_value)
    
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihhod porbability of the instance under the class according to the dataset distribution.
        """
        likelihood = 1
        for i in range(0,1):
            size_vj = len(np.unique(self.dataset[:,i])) 
            n_i_j = np.sum(self.class_value_dataset[:,i] == x[i])
            n_i = len(self.class_value_dataset)
            if n_i_j == 0:
                likelihood *= EPSILLON
            else: 
                likelihood *= (n_i_j + 1) / (n_i + size_vj)
        return likelihood
    
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        return get_instance_posterior(self.get_prior(), self.get_instance_likelihood(x))


    
####################################################################################################
#                                            General
####################################################################################################            
class MAPClassifier():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum a postreiori classifier. 
        This class will hold 2 class distribution, one for class 0 and one for class 1, and will predicit and instance
        by the class that outputs the highest posterior probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1
    
    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
        
        Input
            - An instance to predict.
            
        Output
            - 0 if the posterior probability of class 0 is higher 1 otherwise.
        """
        if self.ccd0.get_instance_posterior(x) > self.ccd1.get_instance_posterior(x): 
            return 0
        else:
            return 1
    
def compute_accuracy(testset, map_classifier):
    """
    Compute the accuracy of a given a testset and using a map classifier object.
    
    Input
        - testset: The test for which to compute the accuracy (Numpy array).
        - map_classifier : A MAPClassifier object capable of prediciting the class for each instance in the testset.
        
    Ouput
        - Accuracy = #Correctly Classified / #testset size
    """
    correct_predictions = 0
    for row in testset:
        if map_classifier.predict(row) == row[-1]:
            correct_predictions +=1
    return correct_predictions / len(testset)        