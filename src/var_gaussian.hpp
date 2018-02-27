#ifndef REP_VAR_GAUSSIAN
#define REP_VAR_GAUSSIAN

/* Gaussian distribution */
class var_gaussian {
private:
	double Gt_rhoP1;
	double Gt_rhoP2;
	double df_dz;
	double df_dz_neg;
	std::mutex my_mutex;

public:
	double z;
	double ee;
	double mean;
	double sigma;
	double u_sigma;

	// Constructor
	var_gaussian() {
		Gt_rhoP1 = 0.0;
		Gt_rhoP2 = 0.0;
		df_dz = 0.0;
		df_dz_neg = 0.0;
	}

	// Destructor: free memory
	~var_gaussian() { }

	// Copy operator (prevents errors due to mutex)
	var_gaussian & operator=(const var_gaussian &rhs) {
		// Check for self-assignment!
		if(this!=&rhs) {
			z = rhs.z;
			ee = rhs.ee;
			mean = rhs.mean;
			sigma = rhs.sigma;
			u_sigma = rhs.u_sigma;
			Gt_rhoP1 = rhs.Gt_rhoP1;
			Gt_rhoP2 = rhs.Gt_rhoP2;
			df_dz = rhs.df_dz;
			df_dz_neg = rhs.df_dz_neg;
		}
		return *this;
	}

	// Initialize randomly
	void initialize_random(gsl_rng *semilla, double val_mean, double val_sigma, double offset) {
		double mm;
		double ss;

		if(val_mean==0.0) {
			val_mean = 0.01;
		}

		mm = val_mean + gsl_ran_flat(semilla,-val_mean*offset,val_mean*offset);
		ss = val_sigma + gsl_ran_flat(semilla,-val_sigma*offset,val_sigma*offset);
		initialize(mm,ss);
	}

	// Initialize
	inline void initialize(double mm, double ss) {
		sigma = ss;
		u_sigma = my_log(my_exp(sigma)-1.0);
		mean = mm;
	}

	// Generate samples
	void sample(gsl_rng *semilla) {
		ee = gsl_ran_ugaussian(semilla);
		z = mean+sigma*ee;
	}

	// Set to mean
	void set_to_mean() {
		z = mean;
		ee = 0.0;
	}

	// Compute and follow gradient
	void update_param_var(double f, double eta, int step_schedule) {
		// Obtain gradients
		double grad_mean;
		double grad_sigma;
		double stepsize;

		// Gradient wrt mean (only g_rep)
		grad_mean = df_dz;

		// Gradient wrt sigma (only g_rep)
		grad_sigma = df_dz*ee;

		// Add analytic gradient of entropy
		grad_sigma += 1.0/sigma;

		// Convert to gradient w.r.t. sigma in unconstr. space
		grad_sigma *= (1.0-my_exp(-sigma));

		// Update Gt
		Gt_rhoP1 = var_stepsize::update_G(grad_mean, Gt_rhoP1, step_schedule);
		Gt_rhoP2 = var_stepsize::update_G(grad_sigma, Gt_rhoP2, step_schedule);

		// Follow gradient wrt mean
		stepsize = var_stepsize::get_stepsize(eta, Gt_rhoP1, step_schedule);
		mean += stepsize*grad_mean;

		// Follow gradient wrt sigma
		stepsize = var_stepsize::get_stepsize(eta, Gt_rhoP2, step_schedule);
		u_sigma += stepsize*grad_sigma;
		u_sigma = (u_sigma<-9.2103)?-9.2103:u_sigma;
		sigma = my_log(1.0+my_exp(u_sigma));
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
	inline double set_grad_to_prior(double mm, double ss) {
		double ss2 = my_pow2(ss);
		double aux = -(z-mm)/ss2;
		df_dz = aux;
		df_dz_neg = 0.0;
		// Return
		return(-0.5*(M_LN2+M_LNPI+my_log(ss2)-aux*(z-mm)));
	}

	inline double set_grad_to_prior(double ss) {
		return set_grad_to_prior(0.0,ss);
	}

	// Increase gradient of the model
	inline void increase_grad(double val) {
		my_mutex.lock();
		df_dz += val;
		my_mutex.unlock();
	}

	// Increase gradient for the negatives
	inline void increase_grad_neg(double val) {
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
		return(-my_log(sigma)-0.5*(M_LN2+M_LNPI+my_pow2((z-mean)/sigma)));
	}
};

#endif
