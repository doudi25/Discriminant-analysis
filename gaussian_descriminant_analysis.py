import numpy as np
import scipy.stats as st
class discriminant_analysis:
    def __init__(self):
        pass
    def fit(self,x_train,y_train):
        n=x_train.shape[0]
        n_features=x_train.shape[1]
        class_labels=len(np.unique(y_train))
        mean_matrix=np.zeros((class_labels,n_features))
        cov_tensor=np.zeros((class_labels,n_features,n_features))
        phi=np.zeros(class_labels)
        self.class_labels=class_labels
        for label in range(class_labels):
            indices=(y_train==label)
            phi[label]=float(np.sum(indices))/n
            mean_matrix[label]=np.mean(x_train[indices,:],axis=0)
            cov_tensor[label] = np.cov(x_train[indices, :], rowvar=False) + 1e-6 * np.eye(n_features)
        self.mean_vectors=mean_matrix
        self.covariance_matrix=cov_tensor
        self.phi=phi
        return self.phi,self.mean_vectors,self.covariance_matrix
    def predict(self,x_tests):
        predict_mat=np.zeros((x_tests.shape[0],self.class_labels))
        for label in range(self.class_labels):
            prob_dist=st.multivariate_normal(self.mean_vectors[label],self.covariance_matrix[label])
            for i,x_test in enumerate(x_tests):
                predict_mat[i,label]=np.log(self.phi[label])+prob_dist.logpdf(x_test)
        return predict_mat.argmax(axis=1)
