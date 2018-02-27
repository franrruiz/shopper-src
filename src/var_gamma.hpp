#ifndef REP_VAR_GAMMA
#define REP_VAR_GAMMA

const int SHP_AUGMENT = 10;	// Number of shape augmentation steps (*must* be >=1)
const bool USE_GCORR = false;	// Use g^cor?

/* Gamma distribution */
class var_gamma {
private:
	double Gt_rhoP1;
	double Gt_rhoP2;
	double df_dz;
	double df_dz_neg;
	std::mutex my_mutex;

public:
	double z;
	double ee;
	double sum_log_u;
	double shape;
	double u_shape;
	double mean;
	double u_mean;
	double rate;

	// Constructor
	var_gamma() {
		Gt_rhoP1 = 0.0;
		Gt_rhoP2 = 0.0;
		df_dz = 0.0;
		df_dz_neg = 0.0;
	}

	// Destructor: free memory
	~var_gamma() { }

	// Copy operator (prevents errors due to mutex)
	var_gamma & operator=(const var_gamma &rhs) {
		// Check for self-assignment!
		if(this!=&rhs) {
			z = rhs.z;
			ee = rhs.ee;
			sum_log_u = rhs.sum_log_u;
			shape = rhs.shape;
			u_shape = rhs.u_shape;
			mean = rhs.mean;
			u_mean = rhs.u_mean;
			rate = rhs.rate;
			Gt_rhoP1 = rhs.Gt_rhoP1;
			Gt_rhoP2 = rhs.Gt_rhoP2;
			df_dz = rhs.df_dz;
			df_dz_neg = rhs.df_dz_neg;
		}
		return *this;
	}

	// Initialize randomly
	void initialize_random(gsl_rng *semilla, double val_shp, double val_rte, double offset) {
		double shp = val_shp + gsl_ran_flat(semilla,-offset*val_shp,offset*val_shp);
		double rte = val_rte + gsl_ran_flat(semilla,-offset*val_rte,offset*val_rte);
		initialize(shp,shp/rte);
	}

	// Initialize
	inline void initialize(double ss, double mm) {
		shape = ss;
		u_shape = my_log(my_exp(shape)-1.0);
		mean = mm;
		u_mean = my_log(my_exp(mm)-1.0);
		rate = ss/mm;
	}

	// Generate samples following Marsaglia and Tsang's method
	void sample(gsl_rng *semilla) {
		sum_log_u = 0.0;
		double u;

		z = aux_sample_above1(semilla,shape+SHP_AUGMENT)/rate;
		for(int b=0; b<SHP_AUGMENT; b++) {
			do {
				u = gsl_ran_flat(semilla,0.0,1.0);
			} while(u==0.0);
			z *= pow(u,1.0/(shape+b));
			sum_log_u += my_log(u)/my_pow2(shape+b);
		}
	}

	// Set to mean
	void set_to_mean() {
		// Warning: ee and sum_log_u are not updated
		z = mean;
	}

	double aux_sample_above1(gsl_rng *semilla, double aa) {
		double aux = 9.0*aa-3.0;
		double d = aux/9.0;
		double c = 1.0/sqrt(aux);
		double aux_cmp = -1.0/c;
		bool stop = false;
		double v;
		double u_rej;
		while(!stop) {
			ee = gsl_ran_ugaussian(semilla);
			if(ee>=aux_cmp) {
				v = my_pow3(1.0+c*ee);
				u_rej = gsl_ran_flat(semilla,0.0,1.0);
				if(u_rej<1.0-0.0331*my_pow2(my_pow2(ee)) || \
				   my_log(u_rej)<0.5*my_pow2(ee)+d*(1.0-v+my_log(v))) {
					stop = true;
				}
			}
		}
		return d*v;
	}

	// Compute and follow gradient
	void update_param_var(double f, double eta, int step_schedule) {
		// Obtain gradients
		double grad_shp;
		double grad_mean;
		double stepsize;
		// WARNING: df_dz is assumed to be preconditioned (i.e., it must be z*df_dz instead of df_dz)

		// Gradient wrt mean (only g_rep)
		grad_mean = df_dz/mean;

		// Gradient wrt alpha (g_rep+g_cor)
		double aux = 9.0*(shape+SHP_AUGMENT)-3.0;
		double sqrt_aux = sqrt(aux);
		// Compute (the pre-conditioned) dh_dalpha(eps;alpha+B)/z:
		double dzaux_dalpha = 9.0/aux-13.5*ee/(my_pow3(sqrt_aux)+ee*aux);
		double g_cor = 0.0;
		if(USE_GCORR) {
			// First term of g_cor
			double auz_z = (aux/9.0)*my_pow3(1.0+ee/sqrt_aux);
			g_cor = (shape+SHP_AUGMENT-1.0-auz_z)*dzaux_dalpha + \
					my_log(auz_z)-my_digamma(shape+SHP_AUGMENT);
			// Second term of g_cor (grad-log-jacobian)
			g_cor += 4.5/aux-9.0*ee/(my_pow3(sqrt_aux)+ee*aux);
		}
		// Final gradient wrt alpha (g_rep+g_cor)
		double dh_dalpha = dzaux_dalpha-1.0/shape-sum_log_u;
		grad_shp = df_dz*dh_dalpha + f*g_cor;

		// Add analytic gradient of entropy
		grad_shp += 1.0+(1.0-shape)*my_trigamma(shape)-1.0/shape;
		grad_mean += 1.0/mean;

		// Convert to gradient in unconstr. space
		grad_shp *= (1.0-my_exp(-shape));
		grad_mean *= (1.0-my_exp(-mean));

		// Update Gt
		Gt_rhoP1 = var_stepsize::update_G(grad_shp, Gt_rhoP1, step_schedule);
		Gt_rhoP2 = var_stepsize::update_G(grad_mean, Gt_rhoP2, step_schedule);

		// Follow gradient wrt shape
		stepsize = var_stepsize::get_stepsize(eta, Gt_rhoP1, step_schedule);
		u_shape += stepsize*grad_shp;
		u_shape = (u_shape<-6.907)?-6.907:u_shape;
		shape = my_log(1.0+my_exp(u_shape));

		// Follow gradient wrt mean
		stepsize = var_stepsize::get_stepsize(eta, Gt_rhoP2, step_schedule);
		u_mean += stepsize*grad_mean;
		u_mean = (u_mean<-9.2103)?-9.2103:u_mean;
		mean = my_log(1.0+my_exp(u_mean));

		// Compute new rate
		rate = shape/mean;
	}

	// Set gradient to 0
	inline void set_grad_to_zero() {
		// Preconditioned grad_log_q
		df_dz = 0.0;
		df_dz_neg = 0.0;
	}

	// Initialize for a new gradient descent algorithm
	inline void initialize_iterations() {
		Gt_rhoP1 = 0.0;
		Gt_rhoP2 = 0.0;
		df_dz = 0.0;
		df_dz_neg = 0.0;
	}

	// Set gradient to prior
	inline double set_grad_to_prior(double aa, double bb) {
		double aux = bb*z;
		// Preconditioned grad_log_q
		df_dz = (aa-1.0)-aux;
		df_dz_neg = 0.0;
		// Return
		return(aa*my_log(bb)-my_loggamma(aa)+(aa-1.0)*my_log(z)-aux);
	}

	// Increase gradient of the model
	inline void increase_grad(double val) {
		my_mutex.lock();
		df_dz += val*z;
		my_mutex.unlock();
	}

	// Increase gradient (already preconditioned)
	inline void increase_grad_precond(double val) {
		// The input 'val' here must be z*df_dz instead of df_dz
		my_mutex.lock();
		df_dz += val;
		my_mutex.unlock();
	}

	// Increase gradient for the negatives
	inline void increase_grad_neg(double val) {
		my_mutex.lock();
		df_dz_neg += val*z;
		my_mutex.unlock();
	}

	// Increase gradient for the negatives (already preconditioned)
	inline void increase_grad_neg_precond(double val) {
		// The input 'val' here must be z*df_dz instead of df_dz
		my_mutex.lock();
		df_dz_neg += val;
		my_mutex.unlock();
	}

	// Scale gradient for the negatives and add to df_dz
	inline void scale_add_grad_neg(double val) {
		my_mutex.lock();
		df_dz_neg *= val;
		df_dz += df_dz_neg;
		my_mutex.unlock();
	}

	// Return the gradient
	inline double get_grad() {
		return df_dz;
	}

	// Compute logq
	inline double logq() {
		return(shape*my_log(rate)-my_loggamma(shape)+(shape-1.0)*my_log(z)-rate*z);
	}
};

#endif
