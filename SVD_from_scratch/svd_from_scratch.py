import numpy as np


class SVD:
    def __init__(self,n_components=None,eps=1e-10):
        """
        n_components: keep only the top k singular vector/value
        eps: tolerance for numerical stability
        """
        self.n_components= n_components
        self.eps = eps
        self.U = None
        self.VT = None
        self.S = None # Sigma as in the formula
        self.rank = None

    def fit(self, A):
        A = np.array( A , dtype=float)
        m, n = A.shape  

        # step1:compute A^T A
        ATA = A.T @ A

        # step2; eigen-decomposition of symetric matrix 
        eigenval , eigenvect =np.linalg.eigh(ATA)

        #step3 : sort eigenvlaues in decending  order
        sor = np.argsort(eigenval)[::-1]
        eigenval = eigenval[sor]
        eigenvect = eigenvect[:, sor]

        #step4: singular value are sqrt of eigenvalues
        singular_value = np.sqrt(np.clip(eigenval , 0 ,None))

        #Remove near-zero singular values for rank calculation
        nonzero = singular_value >self.eps
        self.rank = int(np.sum(nonzero))

        # keep only requested number of components
        k = self.n_components if self.n_components is not None else min(m,n)
        k = min(k,len(singular_value))

        singular_value = singular_value[:k]
        V =eigenvect[:,:k]

        #step5 : compute U columns as u_i =A v_i / sigma_i_
        U_cols = []
        for i in range(k):
            singma = singular_value[i]
            v  = V[:, i]

            if singma >self.eps:
                u = (A @ v) / singma
                # normalize to aviold numerical drift
                u_norm =np.linalg.norm(u)
                if u_norm>self.eps:
                    u = u / u_norm
                U_cols.append(u)

            else:
                # fro zero singular value ,stop building meaningfull deirections
                break

        self.U = np.column_stack(U_cols) if U_cols else np.empty((m,0))     
        self.S = singular_value[:self.U.shape[1]]  
        self.VT = V[:, :self.U.shape[1]].T

        return self
        
    def transform(self, A):
            """
            project A onto the learned right-singular sapce.
            Useful fro dimenctionality reduction
            """
            if self.VT is None:
                raise ValueError ("Call the fit first")
            
            A= np.array(A, dtype=float)

            return A @ self.VT.T
        
    def reconstruct(self):
            """
            Recostruct the matrix using SVD componect.
            """
            if self.U is None or self.VT is None or self.S is None:
                raise ValueError("first call the fit")
            
            return self.U @ np.diag(self.S) @ self.VT

    def reank_k_approximation(self, k=None):
            """
            return rank-k approximation using the top -k singular value.
            """

            if self.U is None or self.S is None or self.VT is None:
                raise ValueError("call the fit first.")

            if k is None:
                k=len(self.S)

            k = min (k,len(self.S))
            U_k =self.U[:, :k]
            S_k = self.S[:k]
            VT_k = self.VT[:k , :]

            return U_k @ np.diag(S_k) @ VT_k    