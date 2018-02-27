#ifndef MY_GSL_UTILITIES_HPP
#define MY_GSL_UTILITIES_HPP

const double myINFINITY = 1.0e100;
const double M_LNPI = gsl_sf_log(M_PI);

/************************ Misc. Utility functions ************************/

inline double my_min(double a, double b) {
	return((a<b)?a:b);
}

inline double my_max(double a, double b) {
	return((a<b)?b:a);
}

inline double my_exp(double val) {
	if(val>700.0) {
		return(gsl_sf_exp(700.0));
	} else if(val<-690.775528) {
		return(1e-300);
	} else {
		return(gsl_sf_exp(val));
	}
}

inline double my_sigmoid(double val) {
	if(val>0) {
		if(val>23.0259) {
			val = 23.0259;
		}
		return(1.0/(1.0+my_exp(-val)));
	} else {
		if(val<-23.0259) {
			val = -23.0259;
		}
		double aux = my_exp(val);
		return(aux/(1.0+aux));
	}
}

inline double my_pow2(double val) {
	return(val*val);
}

inline double my_pow3(double val) {
	return(val*val*val);
}

inline double my_log(double val) {
	return (val<=0?-myINFINITY:gsl_sf_log(val));
}

inline double my_logit(double val) {
	if(val>=1.0-1.0e-6) {
		val = 1.0-1.0e-6;
	} else if(val<1.0e-6) {
		val = 1.0e-6;
	}
	return(my_log(val)-my_log(1-val));
}

inline double my_logsumexp(double *x, int N1) {
	double maxim = -myINFINITY;
	double suma = 0.0;
	for(int n1=0; n1<N1; n1++) {
		if(x[n1]>maxim) {
			maxim = x[n1];
		}
	}
	for(int n1=0; n1<N1; n1++) {
		suma += my_exp(x[n1]-maxim);
	}
	return(maxim+my_log(suma));
}

inline double my_gsl_ran_gamma(const gsl_rng *r, double shape, double scale) {
	double aux = gsl_ran_gamma(r,shape,scale);
	return((aux<1e-300)?1e-300:aux);
}

inline double gsl_ran_invgamma(const gsl_rng *r, double shape, double scale) {
	return(1.0/my_gsl_ran_gamma(r,shape,1.0/scale));
}

inline double my_gsl_ran_beta(const gsl_rng *r, double p1, double p2) {
	double aux = gsl_ran_beta(r,p1,p2);
	aux = (aux<1e-300)?1e-300:aux;
	return((aux>1.0-1e-300)?1.0-1e-300:aux);
}

inline double my_gsl_ran_gammaE(const gsl_rng *r, double shape, double mu) {
	return(my_gsl_ran_gamma(r,shape,mu/shape));
}

inline void my_gsl_ran_dirichlet(const gsl_rng *r, size_t K, const double alpha[], double theta[]) {
	gsl_ran_dirichlet(r,K,alpha,theta);
	for(unsigned int k=0; k<K; k++) {
		if(theta[k]<1e-300) {
			theta[k] = 1e-300;
		} else if(theta[k]>1.0-1e-300) {
			theta[k] = 1.0-1e-300;
		}
	}
}

inline double my_loggamma(double val) {
	return((val<1e-300)?gsl_sf_lngamma(1e-300):gsl_sf_lngamma(val));
}

inline double my_logfactorial(double val) {
	return((val==0.0)?0.0:my_loggamma(val+1.0));
}

inline double my_digamma(double val) {
	return((val<1e-300)?gsl_sf_psi(1e-300):gsl_sf_psi(val));
}

inline double my_trigamma(double val) {
	return((val<1e-300)?gsl_sf_psi_1(1e-300):gsl_sf_psi_1(val));
}

inline double my_softplus(double val) {
	if(val>0) {
		return(my_log(1.0+my_exp(-val))+val);
	} else {
		return(my_log(1.0+my_exp(val)));
	}
}

inline void my_softmax(double *vec, int N) {
	double maximo = -myINFINITY;
	double aux;
	double suma = 0.0;
	// Find the maximum
	for(int nn=0; nn<N; nn++) {
		if(vec[nn]>maximo) {
			maximo = vec[nn];
		}
	}
	// Take exp safely
	for(int nn=0; nn<N; nn++) {
		aux = my_exp(vec[nn]-maximo);
		vec[nn] = aux;
		suma += aux;
	}
	// Renormalize
	for(int nn=0; nn<N; nn++) {
		vec[nn] /= suma;
	}
}


/*************************** cuda functions ***************************/

template<typename T>
inline void d_allocate(T **d_p, int size) {
	cudaError_t result = cudaMalloc(d_p, size*sizeof(T));
	if(result!=cudaSuccess) {
		std::cerr << "[ERR] Failed to allocate device memory (cuda error: " << cudaGetErrorString(result) << ")\n";
		assert(0);
	}
}

template<typename T>
inline void copy_h2d(T *d_p, const T *h_p, int size) {
	cudaError_t result = cudaMemcpy(d_p, h_p, size*sizeof(T), cudaMemcpyHostToDevice);
	if(result!=cudaSuccess) {
		std::cerr << "[ERR] Failed to copy to device memory (cuda error: " << cudaGetErrorString(result) << ")\n";
		assert(0);
	}
}

template<typename T>
inline void copy_d2h(T *h_p, const T *d_p, int size) {
	cudaError_t result = cudaMemcpy(h_p, d_p, size*sizeof(T), cudaMemcpyDeviceToHost);
	if(result!=cudaSuccess) {
		std::cerr << "[ERR] Failed to copy to host memory (cuda error: " << cudaGetErrorString(result) << ")\n";
		assert(0);
	}
}

inline void d_sync() {
	cudaError_t result = cudaDeviceSynchronize();
	if(result!=cudaSuccess) {
		std::cerr << "[ERR] Failed to synchronize with device (cuda error: " << cudaGetErrorString(result) << ")\n";
		assert(0);
	}
}

__global__ void matMulKern_TransB(double *A, double *B, double *C, int N, int M, int K) {
	// Size [N M] = [N K] x [M K]^T
	int row_id = blockIdx.x*blockDim.x + threadIdx.x;
	int col_id = blockIdx.y*blockDim.y + threadIdx.y;

	double tmpSum = 0.0;

	// Each thread computes one element of the block sub-matrix
	if(row_id<N && col_id<M) {
		// Accumulate the sum
		for(int k=0; k<K; k++) {
			tmpSum += A[row_id*K+k] * B[col_id*K+k];
		}
		// Write result into output matrix
		C[row_id*M+col_id] = tmpSum;
	}
}

#endif

