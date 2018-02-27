#ifndef PEMB_INFERENCE_HPP
#define PEMB_INFERENCE_HPP

/*
__global__ void lookAhead_kern(double *d_eta_lookahead, double *d_prod_rho_alpha, 
							   double *d_eta_base, int i,
							   int *d_elem_context, int context_size, int Nitems, 
							   bool flag_avgContext) {
	int k = blockIdx.x*blockDim.x + threadIdx.x;

	// Each thread computes one element
	if(k<Nitems) {
		double denomAvgContext = context_size+1.0;
		if(!flag_avgContext) {
			denomAvgContext = 1.0;
		}

		bool found = false;
		int j;
		// Check if target element
		if(k==i) {
			d_eta_lookahead[k] = -myINFINITY;
			found = true;
		}
		// Check if element in context
		if(!found) {
			for(int idx_j=0; idx_j<context_size; idx_j++) {
				j = d_elem_context[idx_j];
				if(k==j) {
					found = true;
					d_eta_lookahead[k] = -myINFINITY;
				}
			}
		}
		// Compute eta_target
		if(!found) {
			double aux = d_eta_base[k];
			for(int idx_j=0; idx_j<context_size; idx_j++) {
				j = d_elem_context[idx_j];
				aux += d_prod_rho_alpha[k*Nitems+j]/denomAvgContext;
			}
			aux += d_prod_rho_alpha[k*Nitems+i]/denomAvgContext;
			d_eta_lookahead[k] = aux;
		}
	}
}
*/

void increase_gradients_t(int thread_id, my_data *data, const my_hyper *hyper, const my_param *param, my_pvar *pvar, \
						  int t, int batchsize, gsl_rng *semilla, std::mutex *semilla_mutex, \
						  double *logp, std::mutex *logp_mutex);

void compute_val_likelihood_ll(int thread_id, my_data *data, const my_hyper *hyper, const my_param *param, my_pvar *pvar, \
							   int i_checkout, int ll, double *llh);

void compute_test_likelihood_ll(int thread_id, my_data *data, const my_hyper *hyper, const my_param *param, my_pvar *pvar, \
								int i_checkout, int ll, double *llh, Matrix1D<bool> *test_valid_lines);

void compute_test_baskets_likelihood_tt(int thread_id, my_data *data, const my_hyper *hyper, const my_param *param, my_pvar *pvar, \
										int i_checkout, int t, double *llh_checkout, double *llh_nocheckout, \
										Matrix1D<bool> *test_valid_lin, Matrix1D<int> *test_valid_lines_per_trans);

class my_infer {
public:

	static void inference_step(my_data &data, const my_hyper &hyper, const my_param &param, my_pvar &pvar, gsl_rng *semilla) {
		double logp;

		// 0. Subsample transactions at random
		int batchsize = my_min(param.batchsize,data.Ntrans);
		if(batchsize<=0) {
			batchsize = data.Ntrans;
		}
		std::vector<int> transaction_all(data.Ntrans);
		for(int t=0; t<data.Ntrans; t++) {
			transaction_all.at(t) = t;
		}
		std::vector<int> transaction_list(batchsize);
		gsl_ran_choose(semilla,transaction_list.data(),batchsize,transaction_all.data(),data.Ntrans,sizeof(int));

		// 0b. Sample from everything
		sample_all(semilla,data,hyper,param,pvar);

		// 1a. Compute the sum of the alpha's
		if(param.flag_shuffle==-1) {
			std::cout << "  computing sum of alphas..." << endl;
			compute_sum_alpha(data,hyper,param,pvar,transaction_list);
		}

		// 1b. Compute products rho*alpha, theta*alpha, etc.
		compute_prod_all(data,hyper,param,pvar,transaction_list);

		// 2. Initialize all gradients to prior
		std::cout << "  initializing gradients..." << endl;
		logp = set_grad_to_prior(data,hyper,param,pvar);

		// 3. Increase the gradients
		std::cout << "  increasing gradients..." << endl;
		logp += increase_gradients(data,hyper,param,pvar,transaction_list,semilla);

		// 4. Take gradient step
		std::cout << "  taking grad step..." << endl;
		take_grad_step(logp,data,hyper,param,pvar);

		// 5. Output the objective function to a file
		my_output::write_objective_function(param,logp);
	}

	static void compute_prod_all(my_data &data, const my_hyper &hyper, const my_param &param, my_pvar &pvar, std::vector<int> &transaction_list) {
		// First, create active_users vector
		std::set<int> active_users_set;
		int u;
		for(int &t : transaction_list) {
			u = data.user_per_trans.get_object(t);
			active_users_set.insert(u);
		}
		std::vector<int> active_users(active_users_set.begin(), active_users_set.end());

		// Compute rho*alpha
		if(!param.flag_symmetricRho) {
			compute_prod_one(pvar.rho, pvar.alpha, pvar.prod_rho_alpha, pvar.d_prod_rho_alpha);
		} else {
			compute_prod_one(pvar.alpha, pvar.alpha, pvar.prod_rho_alpha, pvar.d_prod_rho_alpha);
		}

		// Compute theta*alpha
		if(param.flag_userVec==1) {
			compute_prod_one(pvar.theta, pvar.rho, pvar.prod_theta_alpha);
		} else if(param.flag_userVec==3) {
			compute_prod_one(pvar.theta, pvar.alpha, pvar.prod_theta_alpha);
		} else if(param.flag_userVec!=0) {
			std::cerr << "[ERR] Not implemented function for userVec=" << std::to_string(param.flag_userVec) << endl;
			assert(0);
		}

		// Compute gamma*beta
		if(param.flag_price>0) {
			compute_prod_one(pvar.gamma, pvar.beta, pvar.prod_gamma_beta);
		}

		// Compute delta*mu
		if(param.flag_day>0) {
			compute_prod_one(pvar.delta, pvar.mu, pvar.prod_delta_mu);
		}
	}

	static void compute_prod_one(double *h_M1, double *h_M2, Matrix2D<double> &prod_result, 
								 double *d_ptr, int N, int M, int K) {
		// Allocate memory (host)
		double *h_R = new double[N*M];

		// Allocate memory (device)
		double *d_M1;
		double *d_M2;
		double *d_R;
		d_allocate(&d_M1, N*K);
		d_allocate(&d_M2, M*K);
		if(d_ptr==nullptr) {
			d_allocate(&d_R, M*N);
		} else {
			d_R = d_ptr;
		}

		// Copy to device
		copy_h2d(d_M1, h_M1, N*K);
		copy_h2d(d_M2, h_M2, M*K);
		copy_h2d(d_R, h_R, M*N);

		// Launch kernel to compute matrix product
		dim3 n_blocks(ceil(N/2.0),ceil(M/256.0),1);
		dim3 n_threads_per_block(2,256,1);
		matMulKern_TransB<<<n_blocks, n_threads_per_block>>>(d_M1, d_M2, d_R, N, M, K);
		d_sync();

		// Copy to host
		copy_d2h(h_R, d_R, M*N);
		if(h_R[0]==0.0 && h_R[1]==0.0 && h_R[N*M-1]==0) {
			std::cerr << "[ERR] The cuda matmul function was not executed properly" << endl;
			assert(0);
		}

		// Copy host pointer to matrix
		copy_host_pointer_to_matrix(prod_result, h_R);

		// Free memory
		cudaFree(d_M1);
		cudaFree(d_M2);
		if(d_ptr==nullptr) {
			cudaFree(d_R);
		}
		delete [] h_R;
	}

	static void compute_prod_one(Matrix2D<var_pointmass> &M1, Matrix2D<var_pointmass> &M2, Matrix2D<double> &prod_result, 
								 double *d_ptr) {
		int N = M1.get_size1();
		int M = M2.get_size1();
		int K = M1.get_size2();
		double *h_M1 = new double[N*K];
		double *h_M2 = new double[M*K];
		copy_matrix_to_host_pointer(M1, h_M1);
		copy_matrix_to_host_pointer(M2, h_M2);

		compute_prod_one(h_M1, h_M2, prod_result, d_ptr, N, M, K);

		delete [] h_M1;
		delete [] h_M2;
	}

	static void compute_prod_one(Matrix2D<var_gaussian> &M1, Matrix2D<var_gaussian> &M2, Matrix2D<double> &prod_result, 
								 double *d_ptr) {
		int N = M1.get_size1();
		int M = M2.get_size1();
		int K = M1.get_size2();
		double *h_M1 = new double[N*K];
		double *h_M2 = new double[M*K];
		copy_matrix_to_host_pointer(M1, h_M1);
		copy_matrix_to_host_pointer(M2, h_M2);

		compute_prod_one(h_M1, h_M2, prod_result, d_ptr, N, M, K);

		delete [] h_M1;
		delete [] h_M2;
	}

	static void compute_prod_one(Matrix2D<var_gamma> &M1, Matrix2D<var_gamma> &M2, Matrix2D<double> &prod_result, 
								 double *d_ptr) {
		int N = M1.get_size1();
		int M = M2.get_size1();
		int K = M1.get_size2();
		double *h_M1 = new double[N*K];
		double *h_M2 = new double[M*K];
		copy_matrix_to_host_pointer(M1, h_M1);
		copy_matrix_to_host_pointer(M2, h_M2);

		compute_prod_one(h_M1, h_M2, prod_result, d_ptr, N, M, K);

		delete [] h_M1;
		delete [] h_M2;
	}

	static void compute_prod_one(Matrix2D<var_pointmass> &M1, Matrix2D<var_pointmass> &M2, Matrix2D<double> &prod_result) {
		compute_prod_one(M1, M2, prod_result, nullptr);
	}

	static void compute_prod_one(Matrix2D<var_gaussian> &M1, Matrix2D<var_gaussian> &M2, Matrix2D<double> &prod_result) {
		compute_prod_one(M1, M2, prod_result, nullptr);
	}

	static void compute_prod_one(Matrix2D<var_gamma> &M1, Matrix2D<var_gamma> &M2, Matrix2D<double> &prod_result) {
		compute_prod_one(M1, M2, prod_result, nullptr);
	}

	static void copy_matrix_to_host_pointer(Matrix2D<var_pointmass> &M, double *h_p) {
		int K = M.get_size2();
		for(int i=0; i<M.get_size1(); i++) {
			for(int j=0; j<K; j++) {
				h_p[i*K+j] = M.get_object(i,j).z;
			}
		}
	}

	static void copy_matrix_to_host_pointer(Matrix2D<var_gaussian> &M, double *h_p) {
		int K = M.get_size2();
		for(int i=0; i<M.get_size1(); i++) {
			for(int j=0; j<K; j++) {
				h_p[i*K+j] = M.get_object(i,j).z;
			}
		}
	}

	static void copy_matrix_to_host_pointer(Matrix2D<var_gamma> &M, double *h_p) {
		int K = M.get_size2();
		for(int i=0; i<M.get_size1(); i++) {
			for(int j=0; j<K; j++) {
				h_p[i*K+j] = M.get_object(i,j).z;
			}
		}
	}

	static void copy_host_pointer_to_matrix(Matrix2D<double> &M, double *h_p) {
		int K = M.get_size2();
		for(int i=0; i<M.get_size1(); i++) {
			for(int j=0; j<K; j++) {
				M.set_object(i, j, h_p[i*K+j]);
			}
		}
	}

	static void sample_all(gsl_rng *semilla, my_data &data, const my_hyper &hyper, const my_param &param, my_pvar &pvar) {
		if(!param.flag_symmetricRho) {
			sample_mat(semilla, pvar.rho);
		}
		sample_mat(semilla, pvar.alpha);
		if(param.flag_itemIntercept) {
			sample_mat(semilla, pvar.lambda0);
		}
		if(param.flag_userVec>0) {
			sample_mat(semilla, pvar.theta);
		}
		if(param.flag_price>0) {
			sample_mat(semilla, pvar.gamma);
			sample_mat(semilla, pvar.beta);
		}
		if(param.flag_day>0) {
			sample_mat(semilla, pvar.delta);
			sample_mat(semilla, pvar.mu);
		}
	}

	static void set_to_mean_all(my_data &data, const my_hyper &hyper, const my_param &param, my_pvar &pvar) {
		if(!param.flag_symmetricRho) {
			set_to_mean_mat(pvar.rho);
		}
		set_to_mean_mat(pvar.alpha);
		if(param.flag_itemIntercept) {
			set_to_mean_mat(pvar.lambda0);
		}
		if(param.flag_userVec>0) {
			set_to_mean_mat(pvar.theta);
		}
		if(param.flag_price>0) {
			set_to_mean_mat(pvar.gamma);
			set_to_mean_mat(pvar.beta);
		}
		if(param.flag_day>0) {
			set_to_mean_mat(pvar.delta);
			set_to_mean_mat(pvar.mu);
		}
	}

	static double set_grad_to_prior(my_data &data, const my_hyper &hyper, const my_param &param, my_pvar &pvar) {
		double logp = 0.0;
		if(!param.flag_symmetricRho) {
			logp += set_grad_to_prior_mat(pvar.rho,0.0,hyper.s2rho);
		}
		logp += set_grad_to_prior_mat(pvar.alpha,0.0,hyper.s2alpha);
		if(param.flag_itemIntercept) {
			logp += set_grad_to_prior_mat(pvar.lambda0,0.0,hyper.s2lambda);
		}
		if(param.flag_userVec>0) {
			logp += set_grad_to_prior_mat(pvar.theta,0.0,hyper.s2theta);
		}
		if(param.flag_price>0) {
			logp += set_grad_to_prior_mat(pvar.gamma,hyper.shp_gamma,hyper.rte_gamma);
			logp += set_grad_to_prior_mat(pvar.beta,hyper.shp_beta,hyper.rte_beta);
		}
		if(param.flag_day>0) {
			logp += set_grad_to_prior_mat(pvar.delta,0.0,hyper.s2delta);
			logp += set_grad_to_prior_mat(pvar.mu,0.0,hyper.s2mu);
		}
		return logp;
	}

	static void sample_mat(gsl_rng *semilla, Matrix1D<var_pointmass> &M) {
		for(int i=0; i<M.get_size1(); i++) {
			M.get_object(i).sample(semilla);
		}
	}

	static void sample_mat(gsl_rng *semilla, Matrix1D<var_gaussian> &M) {
		for(int i=0; i<M.get_size1(); i++) {
			M.get_object(i).sample(semilla);
		}
	}

	static void sample_mat(gsl_rng *semilla, Matrix1D<var_gamma> &M) {
		for(int i=0; i<M.get_size1(); i++) {
			M.get_object(i).sample(semilla);
		}
	}

	static void sample_mat(gsl_rng *semilla, Matrix2D<var_pointmass> &M) {
		for(int i=0; i<M.get_size1(); i++) {
			for(int j=0; j<M.get_size2(); j++) {
				M.get_object(i,j).sample(semilla);
			}
		}
	}

	static void sample_mat(gsl_rng *semilla, Matrix2D<var_gaussian> &M) {
		for(int i=0; i<M.get_size1(); i++) {
			for(int j=0; j<M.get_size2(); j++) {
				M.get_object(i,j).sample(semilla);
			}
		}
	}

	static void sample_mat(gsl_rng *semilla, Matrix2D<var_gamma> &M) {
		for(int i=0; i<M.get_size1(); i++) {
			for(int j=0; j<M.get_size2(); j++) {
				M.get_object(i,j).sample(semilla);
			}
		}
	}

	static void set_to_mean_mat(Matrix1D<var_pointmass> &M) {
		for(int i=0; i<M.get_size1(); i++) {
			M.get_object(i).set_to_mean();
		}
	}

	static void set_to_mean_mat(Matrix1D<var_gaussian> &M) {
		for(int i=0; i<M.get_size1(); i++) {
			M.get_object(i).set_to_mean();
		}
	}

	static void set_to_mean_mat(Matrix1D<var_gamma> &M) {
		for(int i=0; i<M.get_size1(); i++) {
			M.get_object(i).set_to_mean();
		}
	}

	static void set_to_mean_mat(Matrix2D<var_pointmass> &M) {
		for(int i=0; i<M.get_size1(); i++) {
			for(int j=0; j<M.get_size2(); j++) {
				M.get_object(i,j).set_to_mean();
			}
		}
	}

	static void set_to_mean_mat(Matrix2D<var_gaussian> &M) {
		for(int i=0; i<M.get_size1(); i++) {
			for(int j=0; j<M.get_size2(); j++) {
				M.get_object(i,j).set_to_mean();
			}
		}
	}

	static void set_to_mean_mat(Matrix2D<var_gamma> &M) {
		for(int i=0; i<M.get_size1(); i++) {
			for(int j=0; j<M.get_size2(); j++) {
				M.get_object(i,j).set_to_mean();
			}
		}
	}

	static double set_grad_to_prior_mat(Matrix1D<var_pointmass> &M, double mm, double ss2) {
		double logp = 0.0;
		for(int i=0; i<M.get_size1(); i++) {
			logp += M.get_object(i).set_grad_to_prior(mm,ss2);
		}
		return logp;
	}

	static double set_grad_to_prior_mat(Matrix1D<var_gaussian> &M, double mm, double ss2) {
		double logp = 0.0;
		for(int i=0; i<M.get_size1(); i++) {
			logp += M.get_object(i).set_grad_to_prior(mm,ss2);
		}
		return logp;
	}

	static double set_grad_to_prior_mat(Matrix1D<var_gamma> &M, double mm, double ss2) {
		double logp = 0.0;
		for(int i=0; i<M.get_size1(); i++) {
			logp += M.get_object(i).set_grad_to_prior(mm,ss2);
		}
		return logp;
	}

	static double set_grad_to_prior_mat(Matrix2D<var_pointmass> &M, double mm, double ss2) {
		double logp = 0.0;
		for(int i=0; i<M.get_size1(); i++) {
			for(int j=0; j<M.get_size2(); j++) {
				logp += M.get_object(i,j).set_grad_to_prior(mm,ss2);
			}
		}
		return logp;
	}

	static double set_grad_to_prior_mat(Matrix2D<var_gamma> &M, double mm, double ss2) {
		double logp = 0.0;
		for(int i=0; i<M.get_size1(); i++) {
			for(int j=0; j<M.get_size2(); j++) {
				logp += M.get_object(i,j).set_grad_to_prior(mm,ss2);
			}
		}
		return logp;
	}

	static double set_grad_to_prior_mat(Matrix2D<var_gaussian> &M, double mm, double ss2) {
		double logp = 0.0;
		for(int i=0; i<M.get_size1(); i++) {
			for(int j=0; j<M.get_size2(); j++) {
				logp += M.get_object(i,j).set_grad_to_prior(mm,ss2);
			}
		}
		return logp;
	}

	static double set_grad_to_zero(my_data &data, const my_hyper &hyper, const my_param &param, my_pvar &pvar) {
		if(!param.flag_symmetricRho) {
			set_grad_to_zero_mat(pvar.rho);
		}
		set_grad_to_zero_mat(pvar.alpha);
		if(param.flag_itemIntercept) {
			set_grad_to_zero_mat(pvar.lambda0);
		}
		if(param.flag_userVec>0) {
			set_grad_to_zero_mat(pvar.theta);
		}
		if(param.flag_price>0) {
			set_grad_to_zero_mat(pvar.gamma);
			set_grad_to_zero_mat(pvar.beta);
		}
		if(param.flag_day>0) {
			set_grad_to_zero_mat(pvar.mu);
			set_grad_to_zero_mat(pvar.delta);
		}
		return 0.0;
	}

	static void set_grad_to_zero_mat(Matrix1D<var_pointmass> &M) {
		for(int i=0; i<M.get_size1(); i++) {
			M.get_object(i).set_grad_to_zero();
		}
	}

	static void set_grad_to_zero_mat(Matrix1D<var_gaussian> &M) {
		for(int i=0; i<M.get_size1(); i++) {
			M.get_object(i).set_grad_to_zero();
		}
	}

	static void set_grad_to_zero_mat(Matrix1D<var_gamma> &M) {
		for(int i=0; i<M.get_size1(); i++) {
			M.get_object(i).set_grad_to_zero();
		}
	}

	static void set_grad_to_zero_mat(Matrix2D<var_pointmass> &M) {
		for(int i=0; i<M.get_size1(); i++) {
			for(int j=0; j<M.get_size2(); j++) {
				M.get_object(i,j).set_grad_to_zero();
			}
		}
	}

	static void set_grad_to_zero_mat(Matrix2D<var_gaussian> &M) {
		for(int i=0; i<M.get_size1(); i++) {
			for(int j=0; j<M.get_size2(); j++) {
				M.get_object(i,j).set_grad_to_zero();
			}
		}
	}

	static void set_grad_to_zero_mat(Matrix2D<var_gamma> &M) {
		for(int i=0; i<M.get_size1(); i++) {
			for(int j=0; j<M.get_size2(); j++) {
				M.get_object(i,j).set_grad_to_zero();
			}
		}
	}

	static void compute_sum_alpha(my_data &data, const my_hyper &hyper, const my_param &param, my_pvar &pvar) {
		std::vector<int> transaction_all(data.Ntrans);
		for(int t=0; t<data.Ntrans; t++) {
			transaction_all.at(t) = t;
		}
		compute_sum_alpha(data,hyper,param,pvar,transaction_all);
	}

	static void compute_sum_alpha(my_data &data, const my_hyper &hyper, const my_param &param, my_pvar &pvar, std::vector<int> &transaction_list) {
		// Loop over all transactions
		for(int &t : transaction_list) {
			double aux;
			int i;
			double y = 1.0;
			for(int k=0; k<param.K; k++) {
				aux = 0.0;
				// Loop over all lines of transaction t
				for(const int &ll : data.lines_per_trans.get_object(t)) {
					i = data.obs.y_item[ll];
					aux += y*pvar.alpha.get_object(i,k).z;
				}
				pvar.sum_alpha.set_object(t,k,aux);
			}
		}
	}

	static double increase_gradients(my_data &data, const my_hyper &hyper, const my_param &param, my_pvar &pvar,
									 std::vector<int> &transaction_list, gsl_rng *semilla) {
		double logp = 0.0;
		std::mutex logp_mutex;
		std::mutex semilla_mutex;
		int batchsize = transaction_list.size();
		ctpl::thread_pool my_threads(param.Nthreads);

		// For each transaction
		for(int &t : transaction_list) {
			my_threads.push(increase_gradients_t, &data, &hyper, &param, &pvar, t, batchsize, semilla, &semilla_mutex, &logp, &logp_mutex);
		}
		my_threads.stop(true);
		return logp;
	}

	static void increase_gradients_t_i(my_data *data, const my_hyper *hyper, const my_param *param, my_pvar *pvar,
									   double *suma_input, int i, int u, int s, int t, double price,
									   double dL_deta, std::vector<int> *context_items, int argmax) {
		double deta_dz;
		double y_j;
		double y_j_prime;
		int g_i;
		int g_argmax = -1;
		double *suma = new double[param->K];

		// Divide suma by the corresponding value
		double denomAvgContext = 0.0;
		if(context_items->size()>0) {
			if(!param->flag_avgContext) {
				denomAvgContext = 1.0;
			} else {
				denomAvgContext = static_cast<double>(context_items->size());
			}
			for(int k=0; k<param->K; k++) {
				suma[k] = suma_input[k]/denomAvgContext;
			}
		} else {
			set_to_zero(suma, param->K);
		}

		// Get g_i
		if(param->flag_price>0 || param->flag_day>0) {
			g_i = data->group_per_item.get_object(i);
			if(argmax!=-1) {
				g_argmax = data->group_per_item.get_object(argmax);
			}
		}

		// Increase gradient of rho
		for(int k=0; k<param->K; k++) {
			if(param->flag_userVec==0) {
				deta_dz = suma[k];
			} else if(param->flag_userVec==1) {
				deta_dz = suma[k]+pvar->theta.get_object(u,k).z;
			} else if(param->flag_userVec==2) {
				deta_dz = suma[k]*pvar->theta.get_object(u,k).z;
			} else if(param->flag_userVec==3) {
				deta_dz = suma[k];
			}
			if(!param->flag_symmetricRho) {
				pvar->rho.get_object(i,k).increase_grad(dL_deta*deta_dz);
			} else {
				pvar->alpha.get_object(i,k).increase_grad(dL_deta*deta_dz);
			}
			// (lookahead)
			if(param->flag_lookahead && argmax!=-1) {
				double aux_suma_k;
				if(!param->flag_avgContext) {
					aux_suma_k = (suma_input[k]+pvar->alpha.get_object(i,k).z);
				} else {
					aux_suma_k = (suma_input[k]+pvar->alpha.get_object(i,k).z)/(1.0+denomAvgContext);
				}
				if(param->flag_userVec==0) {
					deta_dz = aux_suma_k;
				} else if(param->flag_userVec==1) {
					deta_dz = aux_suma_k+pvar->theta.get_object(u,k).z;
				} else if(param->flag_userVec==2) {
					std::cerr << "[ERR] Not implemented" << endl;
					assert(0);
				} else if(param->flag_userVec==3) {
					deta_dz = aux_suma_k;
				}
				if(!param->flag_symmetricRho) {
					pvar->rho.get_object(argmax,k).increase_grad(dL_deta*deta_dz);
				} else {
					pvar->alpha.get_object(argmax,k).increase_grad(dL_deta*deta_dz);
				}
			}
		}

		// Increase gradient of alpha for all items in the context
		for(const int &j : *context_items) {
			// Scale down y_j to account for the averaging of elements in context
			y_j = 1.0/denomAvgContext;
			if(param->flag_lookahead && argmax!=-1) {
				if(!param->flag_avgContext) {
					y_j_prime = 1.0;
				} else {
					y_j_prime = 1.0/(denomAvgContext+1.0);
				}
			}
			// Increase gradient
			for(int k=0; k<param->K; k++) {
				double aux_k;
				if(!param->flag_symmetricRho) {
					aux_k = y_j*pvar->rho.get_object(i,k).z;
				} else {
					aux_k = y_j*pvar->alpha.get_object(i,k).z;
				}
				if(param->flag_userVec==0) {
					deta_dz = aux_k;
				} else if(param->flag_userVec==1) {
					deta_dz = aux_k;
				} else if(param->flag_userVec==2) {
					deta_dz = aux_k*pvar->theta.get_object(u,k).z;
				} else if(param->flag_userVec==3) {
					deta_dz = aux_k;
				}
				// (lookahead)
				if(param->flag_lookahead && argmax!=-1) {
					if(!param->flag_symmetricRho) {
						deta_dz += y_j_prime*pvar->rho.get_object(argmax,k).z;
					} else {
						deta_dz += y_j_prime*pvar->alpha.get_object(argmax,k).z;
					}
				}
				// increase grad
				pvar->alpha.get_object(j,k).increase_grad(dL_deta*deta_dz);
			}
		}
		// (lookahead, as alpha_i is also in the context)
		if(param->flag_lookahead && argmax!=-1) {
			if(!param->flag_avgContext) {
				y_j_prime = 1.0;
			} else {
				y_j_prime = 1.0/(denomAvgContext+1.0);
			}
			for(int k=0; k<param->K; k++) {
				if(!param->flag_symmetricRho) {
					deta_dz = y_j_prime*pvar->rho.get_object(argmax,k).z;
				} else {
					deta_dz = y_j_prime*pvar->alpha.get_object(argmax,k).z;
				}
				pvar->alpha.get_object(i,k).increase_grad(dL_deta*deta_dz);
			}
		}

		// Increase gradient of alpha_i (only if param->flag_userVec==3)
		if(param->flag_userVec==3) {
			for(int k=0; k<param->K; k++) {
				deta_dz = pvar->theta.get_object(u,k).z;
				pvar->alpha.get_object(i,k).increase_grad(dL_deta*deta_dz);
			}
			// (lookahead)
			if(param->flag_lookahead && argmax!=-1) {
				for(int k=0; k<param->K; k++) {
					deta_dz = pvar->theta.get_object(u,k).z;
					pvar->alpha.get_object(argmax,k).increase_grad(dL_deta*deta_dz);
				}
			}
		}

		// Increase gradient of lambda0
		if(param->flag_itemIntercept) {
			pvar->lambda0.get_object(i).increase_grad(dL_deta);
			// (lookahead)
			if(param->flag_lookahead && argmax!=-1) {
				pvar->lambda0.get_object(argmax).increase_grad(dL_deta);
			}
		}

		// Increase gradient of theta
		if(param->flag_userVec>0) {
			for(int k=0; k<param->K; k++) {
				if(param->flag_userVec==1) {
					deta_dz = pvar->rho.get_object(i,k).z;
				} else if(param->flag_userVec==2) {
					deta_dz = suma[k]*pvar->rho.get_object(i,k).z;
				} else if(param->flag_userVec==3) {
					deta_dz = pvar->alpha.get_object(i,k).z;
				}
				// (lookahead)
				if(param->flag_lookahead && argmax!=-1) {
					if(param->flag_userVec==1) {
						deta_dz += pvar->rho.get_object(argmax,k).z;
					} else if(param->flag_userVec==3) {
						deta_dz += pvar->alpha.get_object(argmax,k).z;
					}
				}
				// increase grad
				pvar->theta.get_object(u,k).increase_grad(dL_deta*deta_dz);
			}
		}

		// Increase gradient of the price vectors
		for(int k=0; k<param->flag_price; k++) {
			// gamma
			deta_dz = -price*pvar->beta.get_object(g_i,k).z;
			// (lookahead)
			if(param->flag_lookahead && argmax!=-1) {
				deta_dz += -price*pvar->beta.get_object(g_argmax,k).z;
			}
			// increase grad
			pvar->gamma.get_object(u,k).increase_grad(dL_deta*deta_dz);

			// beta
			deta_dz = -price*pvar->gamma.get_object(u,k).z;
			pvar->beta.get_object(g_i,k).increase_grad(dL_deta*deta_dz);
			// (lookahead)
			if(param->flag_lookahead && argmax!=-1) {
				deta_dz = -price*pvar->gamma.get_object(u,k).z;
				pvar->beta.get_object(g_argmax,k).increase_grad(dL_deta*deta_dz);
			}
		}

		// Increase gradient of seasonal effect parameters
		for(int k=0; k<param->flag_day; k++) {
			int d = data->day_per_session.get_object(s);

			// delta
			deta_dz = pvar->mu.get_object(g_i,k).z;
			// (lookahead)
			if(param->flag_lookahead && argmax!=-1) {
				deta_dz += pvar->mu.get_object(g_argmax,k).z;
			}
			// increase grad
			pvar->delta.get_object(d,k).increase_grad(dL_deta*deta_dz);

			// mu
			deta_dz = pvar->delta.get_object(d,k).z;
			pvar->mu.get_object(g_i,k).increase_grad(dL_deta*deta_dz);
			// (lookahead)
			if(param->flag_lookahead && argmax!=-1) {
				deta_dz = pvar->delta.get_object(d,k).z;
				pvar->mu.get_object(g_argmax,k).increase_grad(dL_deta*deta_dz);
			}
		}

		// Free memory
		delete [] suma;
	}

	static int take_negative_sample(int i, gsl_rng *semilla, my_data &data, const my_param &param) {
		int i_prime;
		if(param.flag_nsFreq<=1) {
			i_prime = gsl_ran_discrete(semilla, data.negsampling_dis);
		} else if(param.flag_nsFreq>=2) {
			int g_i = data.group_per_item.get_object(i);
			i_prime = gsl_ran_discrete(semilla, data.negsampling_dis_per_group.at(g_i));
		} else {
			std::cerr << "[ERR] Wrong value for -nsFreq: " << std::to_string(param.flag_nsFreq) << endl;
			assert(0);
		}
		return i_prime;
	} 

	static int take_negative_sample(int i, gsl_rng *semilla, std::mutex *semilla_mutex, my_data *data, const my_param *param) {
		int i_prime;
		if(param->flag_nsFreq<=1) {
			semilla_mutex->lock();
			i_prime = gsl_ran_discrete(semilla, data->negsampling_dis);
			semilla_mutex->unlock();
		} else if(param->flag_nsFreq>=2) {
			int g_i = data->group_per_item.get_object(i);
			semilla_mutex->lock();
			i_prime = gsl_ran_discrete(semilla, data->negsampling_dis_per_group.at(g_i));
			semilla_mutex->unlock();
		} else {
			std::cerr << "[ERR] Wrong value for -nsFreq: " << std::to_string(param->flag_nsFreq) << endl;
			assert(0);
		}
		return i_prime;
	} 

	static bool elem_in_vector(std::vector<int> &vec, int val) {
		for(const int &i : vec) {
			if(i==val) {
				return true;
			}
		}
		return false;
	}

	static bool elem_in_vector(std::vector<int> *vec, int val) {
		for(const int &i : *vec) {
			if(i==val) {
				return true;
			}
		}
		return false;
	}

	static void compute_eta_base(my_data &data, const my_hyper &hyper, const my_param &param, my_pvar &pvar, int t, int u, int s, Matrix1D<double> &eta_base) {
		double eta;
		double price;
		int g_i;
		int d;
		if(param.flag_day>0) {
			d = data.day_per_session.get_object(s);
		}

		for(int i=0; i<data.Nitems; i++) {
			eta = 0.0;
			price = data.get_price(i,s,param);

			// Get group of item i
			if(param.flag_day>0 || param.flag_price>0) {
				g_i = data.group_per_item.get_object(i);
			}

			// Add intercept term
			if(param.flag_itemIntercept) {
				eta += pvar.lambda0.get_object(i).z;
			}
			// Add user vectors
			if(param.flag_userVec>0) {
				eta += pvar.prod_theta_alpha.get_object(u,i);
			}
			// Add price term
			if(param.flag_price>0) {
				eta -= pvar.prod_gamma_beta.get_object(u,g_i)*price;
			}
			// Add seasonal effect
			if(param.flag_day>0) {
				eta += pvar.prod_delta_mu.get_object(d,g_i);
			}

			eta_base.set_object(i,eta);
		}
	}

	static void compute_eta_base(my_data *data, const my_hyper *hyper, const my_param *param, my_pvar *pvar, int t, int u, int s, double *eta_base) {
		double eta;
		double price;
		int g_i;
		int d;
		if(param->flag_day>0) {
			d = data->day_per_session.get_object(s);
		}

		for(int i=0; i<data->Nitems; i++) {
			eta = 0.0;
			price = data->get_price(i,s,param);

			// Get group of item i
			if(param->flag_day>0 || param->flag_price>0) {
				g_i = data->group_per_item.get_object(i);
			}

			// Add intercept term
			if(param->flag_itemIntercept) {
				eta += pvar->lambda0.get_object(i).z;
			}
			// Add user vectors
			if(param->flag_userVec>0) {
				eta += pvar->prod_theta_alpha.get_object(u,i);
			}
			// Add price term
			if(param->flag_price>0) {
				eta -= pvar->prod_gamma_beta.get_object(u,g_i)*price;
			}
			// Add seasonal effect
			if(param->flag_day>0) {
				eta += pvar->prod_delta_mu.get_object(d,g_i);
			}

			eta_base[i] = eta;
		}
	}

	static double compute_mean(my_data &data, const my_param &param, my_pvar &pvar, int t, int i, int u, int s,
							   std::vector<int> &elem_context, int &argmax, Matrix1D<double> eta_base) {
		double eta = 0.0;
		double denomAvgContext = static_cast<double>(elem_context.size());
		if(!param.flag_avgContext) {
			denomAvgContext = 1.0;
		}
		argmax = -1;

		// Get checkout item
		int i_checkout = -1;
		if(param.flag_checkout) {
			unsigned long long mid = ULLONG_MAX;
			i_checkout = data.item_ids.find(mid)->second;
		}

		// Eta base
		eta = eta_base.get_object(i);
		// Add elements in context
		for(int &j : elem_context) {
			eta += pvar.prod_rho_alpha.get_object(i,j)/denomAvgContext;
		}

		// Look-ahead
		if(param.flag_lookahead>0 && i!=i_checkout) {
			double maximo = -myINFINITY;
			
			// NoCuda: From here
			double aux;
			for(int k=0; k<data.Nitems; k++) {
				if(k!=i && !elem_in_vector(elem_context,k)) {
					aux = eta_base.get_object(k);
					for(int &j : elem_context) {
						if(!param.flag_avgContext) {
							aux += pvar.prod_rho_alpha.get_object(k,j);
						} else {
							aux += pvar.prod_rho_alpha.get_object(k,j)/(denomAvgContext+1.0);
						}
					}
					if(!param.flag_avgContext) {
						aux += pvar.prod_rho_alpha.get_object(k,i);
					} else {
						aux += pvar.prod_rho_alpha.get_object(k,i)/(denomAvgContext+1.0);
					}
					if(aux>maximo) {
						maximo = aux;
						argmax = k;
					}
				}
			}
			// NoCuda: Up to here
			
			// Cuda: From here
			/*
			int *d_elem_context;
			d_allocate(&d_elem_context, elem_context.size());
			if(elem_context.size()>0) {
				copy_h2d(d_elem_context, elem_context.data(), elem_context.size());
			}
			double *h_eta_lookahead = new double[data.Nitems];
			double *d_eta_lookahead;
			d_allocate(&d_eta_lookahead, data.Nitems);
			double *d_eta_base;
			d_allocate(&d_eta_base, data.Nitems);
			copy_h2d(d_eta_base, eta_base.get_pointer(0), data.Nitems);
			int n_blocks = ceil(data.Nitems/512.0);
			int n_threads_per_block = 512;
			lookAhead_kern<<<n_blocks, n_threads_per_block>>>(d_eta_lookahead, pvar.d_prod_rho_alpha, d_eta_base, i, d_elem_context, elem_context.size(), data.Nitems, param.flag_avgContext);
			d_sync();
			copy_d2h(h_eta_lookahead, d_eta_lookahead, data.Nitems);
			thrust::device_ptr<double> t_eta_lookahead = thrust::device_pointer_cast(d_eta_lookahead);
			thrust::device_ptr<double> t_maximo = thrust::max_element(t_eta_lookahead, t_eta_lookahead+data.Nitems);
			argmax = t_maximo - t_eta_lookahead;
			maximo = h_eta_lookahead[argmax];
			cudaFree(d_elem_context);
			cudaFree(d_eta_base);
			cudaFree(d_eta_lookahead);
			delete [] h_eta_lookahead;
			*/
			// Cuda: Up to here

			eta += maximo;
		}

		// Return
		return eta;
	}

	static double compute_mean(my_data *data, const my_param *param, my_pvar *pvar, int t, int i, int u, int s,
							   std::vector<int> *elem_context, int *argmax, double *eta_base) {
		double eta = 0.0;
		double denomAvgContext = static_cast<double>(elem_context->size());
		if(!param->flag_avgContext) {
			denomAvgContext = 1.0;
		}
		*argmax = -1;

		// Get checkout item
		int i_checkout = -1;
		if(param->flag_checkout) {
			unsigned long long mid = ULLONG_MAX;
			i_checkout = data->item_ids.find(mid)->second;
		}

		// Eta base
		eta = eta_base[i];
		// Add elements in context
		for(int &j : *elem_context) {
			eta += pvar->prod_rho_alpha.get_object(i,j)/denomAvgContext;
		}

		// Look-ahead
		if(param->flag_lookahead>0 && i!=i_checkout) {
			double maximo = -myINFINITY;
			
			// NoCuda: From here
			double aux;
			for(int k=0; k<data->Nitems; k++) {
				if(k!=i && !elem_in_vector(elem_context,k)) {
					aux = eta_base[k];
					for(int &j : *elem_context) {
						if(!param->flag_avgContext) {
							aux += pvar->prod_rho_alpha.get_object(k,j);
						} else {
							aux += pvar->prod_rho_alpha.get_object(k,j)/(denomAvgContext+1.0);
						}
					}
					if(!param->flag_avgContext) {
						aux += pvar->prod_rho_alpha.get_object(k,i);
					} else {
						aux += pvar->prod_rho_alpha.get_object(k,i)/(denomAvgContext+1.0);
					}
					if(aux>maximo) {
						maximo = aux;
						*argmax = k;
					}
				}
			}
			// NoCuda: Up to here
			eta += maximo;
		}

		// Return
		return eta;
	}

	static void substract_contribution_sum_alpha(int t, int i, double y, my_data &data, const my_param &param, my_pvar &pvar, double *vec) {
		y = 1.0;
		// Remove contribution from sum_lambdas
		for(int k=0; k<param.K; k++) {
			vec[k] = pvar.sum_alpha.get_object(t,k)-y*pvar.alpha.get_object(i,k).z;
		}
	}

	static void substract_contribution_sum_alpha(int t, int i, double y, my_data *data, const my_param *param, my_pvar *pvar, double *vec) {
		y = 1.0;
		// Remove contribution from sum_lambdas
		for(int k=0; k<param->K; k++) {
			vec[k] = pvar->sum_alpha.get_object(t,k) - y * pvar->alpha.get_object(i,k).z;
		}
	}

	static void set_sum_alpha(int t, my_data &data, const my_param &param, my_pvar &pvar, double *vec) {
		for(int k=0; k<param.K; k++) {
			vec[k] = pvar.sum_alpha.get_object(t,k);
		}
	}

	static void set_sum_alpha(int t, my_data *data, const my_param *param, my_pvar *pvar, double *vec) {
		for(int k=0; k<param->K; k++) {
			vec[k] = pvar->sum_alpha.get_object(t,k);
		}
	}

	static void compute_test_performance(bool writeFinalFile, int duration, my_data &data, const my_hyper &hyper, const my_param &param, my_pvar &pvar) {
		if(param.noTest) {
			return;
		}

		ctpl::thread_pool my_threads(param.Nthreads);
		double sum_llh = 0.0;
		int T = static_cast<int>(data.obs_test.T);
		double *llh = new double[T];
		int n_valid_lines = 0;
		int n_valid_trans = 0;

		// Create a vector specifying how many valid lines per transactions
		// there are in test.tsv. A line is valid if both the item and the user
		// appear in train.tsv (the user is necessary only when -userVec != 0)
		// Also, mark each line of test.tsv with a flag indicating whether it is valid
		Matrix1D<int> test_valid_lines_per_trans = Matrix1D<int>(data.test_Ntrans);
		Matrix1D<bool> test_valid_lines = Matrix1D<bool>(T);
		set_to_zero(test_valid_lines_per_trans);
		set_to_zero(test_valid_lines);
		for(int ll=0; ll<T; ll++) {
			int u = data.obs_test.y_user[ll];
			int i = data.obs_test.y_item[ll];
			int t = data.obs_test.y_trans[ll];
			int y = data.obs_test.y_rating[ll];
			if(y>0 && data.lines_per_item.get_object(i).size()>0) {
				if(param.flag_userVec==0 || data.lines_per_user.get_object(u).size()>0) {
					int aux = 1+test_valid_lines_per_trans.get_object(t);
					test_valid_lines_per_trans.set_object(t,aux);
					test_valid_lines.set_object(ll,true);
					n_valid_lines++;
				}
			}
		}

		// Count n_valid_trans
		for(int t=0; t<data.test_Ntrans; t++) {
			if(test_valid_lines_per_trans.get_object(t)>0) {
				n_valid_trans++;
			}
		}

		// Locate checkout item
		int i_checkout = -1;
		if(param.flag_checkout) {
			unsigned long long mid = ULLONG_MAX;
			i_checkout = data.item_ids.find(mid)->second;
		}

		// Ensure that sum_alpha is up-to-date
		/* 
		compute_sum_alpha(data,hyper,param,pvar);
		*/

		// For each line of test.tsv
		for(int ll=0; ll<T; ll++) {
			my_threads.push(compute_test_likelihood_ll, &data, &hyper, &param, &pvar, i_checkout, ll, llh, &test_valid_lines);
		}
		my_threads.stop(true);

		// Print to file (all lines)
		if(writeFinalFile) {
			string fname = param.outdir+"/test_all.tsv";
			char buffer[500];
			for(int ll=0; ll<T; ll++) {
				if(test_valid_lines.get_object(ll)) {
					sprintf(buffer,"%.9f",llh[ll]);
				} else {
					sprintf(buffer,"0");
				}
				my_output::write_line(fname,string(buffer));
			}
		}

		// Compute average
		for(int ll=0; ll<T; ll++) {
			if(test_valid_lines.get_object(ll)) {
				sum_llh += llh[ll];
			}
		}
		double llh_avg = sum_llh/static_cast<double>(n_valid_lines);

		// Print to file (avg)
		string fname = param.outdir+"/test.tsv";
		char buffer[500];
		sprintf(buffer,"%d\t%d\t%.9f\t%d\t%d",param.it,duration,llh_avg,n_valid_trans,n_valid_lines);
		my_output::write_line(fname,string(buffer));

		// Free memory
		delete [] llh;
	}

	static void compute_test_performance_baskets(bool writeFinalFile, int duration, my_data &data, const my_hyper &hyper, const my_param &param, my_pvar &pvar) {
		if(param.noTest) {
			return;
		}

		ctpl::thread_pool my_threads(param.Nthreads);
		double sum_llh_checkout = 0.0;
		double sum_llh_nocheckout = 0.0;
		int T = static_cast<int>(data.obs_test.T);
		double *llh_checkout = new double[T+data.test_Ntrans];
		double *llh_nocheckout = new double[T];
		int n_valid_lines = 0;
		int n_valid_trans = 0;

		// Create a vector specifying how many valid lines per transactions
		// there are in test.tsv. A line is valid if both the item and the user
		// appear in train.tsv (the user is necessary only when -userVec != 0)
		// Also, mark each line of test.tsv with a flag indicating whether it is valid
		Matrix1D<int> test_valid_lines_per_trans = Matrix1D<int>(data.test_Ntrans);
		Matrix1D<bool> test_valid_lines = Matrix1D<bool>(T);
		set_to_zero(test_valid_lines_per_trans);
		set_to_zero(test_valid_lines);
		for(int ll=0; ll<T; ll++) {
			int u = data.obs_test.y_user[ll];
			int i = data.obs_test.y_item[ll];
			int t = data.obs_test.y_trans[ll];
			int y = data.obs_test.y_rating[ll];
			if(y>0 && data.lines_per_item.get_object(i).size()>0) {
				if(param.flag_userVec==0 || data.lines_per_user.get_object(u).size()>0) {
					int aux = 1+test_valid_lines_per_trans.get_object(t);
					test_valid_lines_per_trans.set_object(t,aux);
					test_valid_lines.set_object(ll,true);
					n_valid_lines++;
				}
			}
		}

		// Count n_valid_trans
		for(int t=0; t<data.test_Ntrans; t++) {
			if(test_valid_lines_per_trans.get_object(t)>0) {
				n_valid_trans++;
			}
		}

		// Locate checkout item
		int i_checkout = -1;
		if(param.flag_checkout) {
			unsigned long long mid = ULLONG_MAX;
			i_checkout = data.item_ids.find(mid)->second;
		}

		// For each transaction in test.tsv
		for(int t=0; t<data.test_Ntrans; t++) {
			my_threads.push(compute_test_baskets_likelihood_tt, &data, &hyper, &param, &pvar, i_checkout, t, llh_checkout, llh_nocheckout, &test_valid_lines, &test_valid_lines_per_trans);
		}
		my_threads.stop(true);

		// Print to file (all lines)
		if(writeFinalFile) {
			int t;
			int i;
			int u;
			int s;
			for(int type_llh=0; type_llh<2; type_llh++) {
				double *llh_print = nullptr;
				string fname;
				char buffer[500];
				if(type_llh==0) {
					fname = param.outdir+"/test_baskets_all_noChkout.tsv";
					llh_print = llh_nocheckout;
				} else {
					fname = param.outdir+"/test_baskets_all.tsv";
					llh_print = llh_checkout;
				}
				int ll_max = T;
				if(type_llh!=0) {
					ll_max += data.test_Ntrans;
				}
				for(int ll=0; ll<ll_max; ll++) {
					// Get (user,session)
					u = data.test_user_per_trans.get_object(t);
					s = data.test_session_per_trans.get_object(t);
					// Get (trans,item)
					if(ll<T) {
						t = data.obs_test.y_trans[ll];
						i = data.obs_test.y_item[ll];
					} else {
						t = ll-T;
						i = i_checkout;
					}
					// Get the id's 
					unsigned long long u_id = data.find_by_value(data.user_ids,u);
					unsigned long long s_id = data.find_by_value(data.session_ids,s);
					unsigned long long i_id = data.find_by_value(data.item_ids,i);
					// If valid line, print
					if(llh_print[ll]<=0.0) {
						sprintf(buffer,"%llu\t%llu\t%llu\t%.9f",u_id,i_id,s_id,llh_print[ll]);
					} else {
						sprintf(buffer,"0\t0\t0\t0");
					}
					my_output::write_line(fname,string(buffer));
				}
			}
		}

		// Compute average
		for(int ll=0; ll<T+data.test_Ntrans; ll++) {
			if(ll<T) {
				if(llh_checkout[ll]<=0.0) {
					sum_llh_checkout += llh_checkout[ll];
				}
				if(llh_nocheckout[ll]<=0.0) {
					sum_llh_nocheckout += llh_nocheckout[ll];
				}
			} else if(ll>=T && llh_checkout[ll]<=0.0) {
				sum_llh_checkout += llh_checkout[ll];
			}
		}
		double llh_avg_checkout = sum_llh_checkout/static_cast<double>(n_valid_trans);
		double llh_avg_nocheckout = sum_llh_nocheckout/static_cast<double>(n_valid_trans);

		// Print to file (avg)
		char buffer[500];
		string fname = param.outdir+"/test_baskets.tsv";
		sprintf(buffer,"%d\t%d\t%.9f\t%d\t%d",param.it,duration,llh_avg_checkout,n_valid_trans,n_valid_lines);
		my_output::write_line(fname,string(buffer));

		fname = param.outdir+"/test_baskets_noChkout.tsv";
		sprintf(buffer,"%d\t%d\t%.9f\t%d\t%d",param.it,duration,llh_avg_nocheckout,n_valid_trans,n_valid_lines);
		my_output::write_line(fname,string(buffer));

		// Free memory
		delete [] llh_checkout;
		delete [] llh_nocheckout;
	}

	static double compute_val_likelihood(bool writeFinalFile, int duration, my_data &data, const my_param &param, const my_hyper &hyper, my_pvar &pvar) {
		if(param.noVal) {
			return 0.0;
		}

		ctpl::thread_pool my_threads(param.Nthreads);
		int count = 0;
		double sum_llh = 0.0;
		double *llh = new double[data.obs_val.T];

		// Locate checkout item
		int i_checkout = -1;
		if(param.flag_checkout) {
			unsigned long long mid = ULLONG_MAX;
			i_checkout = data.item_ids.find(mid)->second;
		}

		// Ensure that sum_alpha is up-to-date
		/* 
		compute_sum_alpha(data,hyper,param,pvar);
		*/

		// For each line of validation.tsv
		for(unsigned int ll=0; ll<data.obs_val.T; ll++) {
			my_threads.push(compute_val_likelihood_ll, &data, &hyper, &param, &pvar, i_checkout, ll, llh);
		}
		my_threads.stop(true);

		// Print to file (all lines)
		if(writeFinalFile) {
			string fname = param.outdir+"/validation_all.tsv";
			char buffer[500];
			for(unsigned int ll=0; ll<data.obs_val.T; ll++) {
				if(llh[ll] != myINFINITY) {
					sprintf(buffer,"%.9f",llh[ll]);
				} else {
					sprintf(buffer,"0");
				}
				my_output::write_line(fname,string(buffer));
			}
		}

		// Compute average llh
		for(unsigned int ll=0; ll<data.obs_val.T; ll++) {
			if(llh[ll] != myINFINITY) {
				sum_llh += llh[ll];
				count += 1;
			}
		}
		double llh_avg = sum_llh/static_cast<double>(count);

		// Print to file (avg)
		string fname = param.outdir+"/validation.tsv";
		char buffer[500];
		sprintf(buffer,"%d\t%d\t%.9f\t%d",param.it,duration,llh_avg,count);
		my_output::write_line(fname,string(buffer));

		// Free memory
		delete [] llh;

		// Return
		return llh_avg;
	}

	static void prepare_test_valid(gsl_rng *semilla, my_data &data, const my_hyper &hyper, const my_param &param, my_pvar &pvar) {
		// Sample from everything
		set_to_mean_all(data,hyper,param,pvar);

		// Create fake transaction_list
		std::vector<int> transaction_list = std::vector<int>(data.Ntrans);
		for(int t=0; t<data.Ntrans; t++){
			transaction_list.at(t) = t;
		} 

		// Compute products rho*alpha, theta*alpha, etc.
		compute_prod_all(data,hyper,param,pvar,transaction_list);
	}

	static double compute_avg_norm(Matrix2D<var_pointmass> &m) {
		double suma = 0.0;
		for(int i=0; i<m.get_size1(); i++) {
			for(int j=0; j<m.get_size2(); j++) {
				suma += my_pow2(m.get_object(i,j).z);
			}
		}
		return(suma/static_cast<double>(m.get_size1()*m.get_size2()));
	}

	static double compute_avg_norm(Matrix2D<var_gaussian> &m) {
		double suma = 0.0;
		for(int i=0; i<m.get_size1(); i++) {
			for(int j=0; j<m.get_size2(); j++) {
				suma += my_pow2(m.get_object(i,j).z);
			}
		}
		return(suma/static_cast<double>(m.get_size1()*m.get_size2()));
	}

	static double compute_avg_norm(Matrix2D<var_gamma> &m) {
		double suma = 0.0;
		for(int i=0; i<m.get_size1(); i++) {
			for(int j=0; j<m.get_size2(); j++) {
				suma += my_pow2(m.get_object(i,j).z);
			}
		}
		return(suma/static_cast<double>(m.get_size1()*m.get_size2()));
	}

	static void set_to_zero(double *vec, int K) {
		for(int k=0; k<K; k++) {
			vec[k] = 0.0;
		}
	}

	static void set_to_zero(Matrix1D<int> &M) {
		for(int i=0; i<M.get_size1(); i++) {
			M.set_object(i,0);
		}
	}

	static void set_to_zero(Matrix1D<bool> &M) {
		for(int i=0; i<M.get_size1(); i++) {
			M.set_object(i,false);
		}
	}

	static void set_to_zero(Matrix2D<double> &M) {
		for(int i=0; i<M.get_size1(); i++) {
			for(int j=0; j<M.get_size2(); j++) {
				M.set_object(i,j,0.0);
			}
		}
	}

	static void set_to_zero(Matrix3D<int> &M) {
		for(int i=0; i<M.get_size1(); i++) {
			for(int j=0; j<M.get_size2(); j++) {
				for(int k=0; k<M.get_size3(); k++) {
					M.set_object(i,j,k,0);
				}
			}
		}
	}

	static void take_grad_step(double logp, const my_data &data, const my_hyper &hyper, const my_param &param, my_pvar &pvar) {
		for(int i=0; i<data.Nitems; i++) {
			for(int k=0; k<param.K; k++) {
				if(!param.flag_symmetricRho) {
					pvar.rho.get_object(i,k).update_param_var(logp, param.eta, param.flag_step_schedule);
				}
				pvar.alpha.get_object(i,k).update_param_var(logp, param.eta, param.flag_step_schedule);
			}
			if(param.flag_itemIntercept) {
				pvar.lambda0.get_object(i).update_param_var(logp, param.eta, param.flag_step_schedule);
			}
		}
		if(param.flag_userVec>0) {
			for(int u=0; u<data.Nusers; u++) {
				for(int k=0; k<param.K; k++) {
					pvar.theta.get_object(u,k).update_param_var(logp, param.eta, param.flag_step_schedule);
				}
			}
		}
		for(int k=0; k<param.flag_price; k++) {
			for(int u=0; u<data.Nusers; u++) {
				pvar.gamma.get_object(u,k).update_param_var(logp, param.eta, param.flag_step_schedule);
			}
			for(int g_i=0; g_i<data.NitemGroups; g_i++) {
				pvar.beta.get_object(g_i,k).update_param_var(logp, param.eta, param.flag_step_schedule);
			}
		}
		for(int k=0; k<param.flag_day; k++) {
			for(int d=0; d<data.Ndays; d++) {
				pvar.delta.get_object(d,k).update_param_var(logp, param.eta, param.flag_step_schedule);
			}
			for(int g_i=0; g_i<data.NitemGroups; g_i++) {
				pvar.mu.get_object(g_i,k).update_param_var(logp, param.eta, param.flag_step_schedule);
			}
		}
	}

	static void increase_double(double *logp, std::mutex *logp_mutex, double val) {
		logp_mutex->lock();
		*logp += val;
		logp_mutex->unlock();
	}

};


void increase_gradients_t(int thread_id, my_data *data, const my_hyper *hyper, const my_param *param, my_pvar *pvar, \
						  int t, int batchsize, gsl_rng *semilla, std::mutex *semilla_mutex, \
						  double *logp, std::mutex *logp_mutex) {
	int u;
	int s;
	int i_prime;
	int argmax;
	int argmax_prime;
	double *suma = new double[param->K];
	double mm;
	double mm_prime;
	double price;
	double price_prime;
	double dL_deta;
	double sviFactor;
	double sigmoid;
	int count_i;
	std::vector<int> context_items;
	double *eta_base = new double[data->Nitems];

	// Get number of copies
	int Ncopies = 1;
	if(param->flag_shuffle>0) {
		Ncopies = param->flag_shuffle;
	}

	// Get user and session
	u = data->user_per_trans.get_object(t);
	s = data->session_per_trans.get_object(t);

	// Compute eta base for all items
	my_infer::compute_eta_base(data,hyper,param,pvar,t,u,s,eta_base);

	for(int copy=0; copy<Ncopies; copy++) {
		// Get the items in this transaction
		std::vector<int> items_sorted;
		for(int &idx_t : data->lines_per_trans.get_object(t)) {
			int i = data->obs.y_item[idx_t];
			items_sorted.push_back(i);
		}
		// Shuffle the items in this transaction
		if(param->flag_shuffle>0) {
			semilla_mutex->lock();
			gsl_ran_shuffle(semilla,items_sorted.data(),data->lines_per_trans.get_object(t).size(),sizeof(int));
			semilla_mutex->unlock();
		}
		// If checkout, append the checkout item to the list of items
		if(param->flag_checkout && param->flag_shuffle>=0) {
			unsigned long long mid = ULLONG_MAX;
			int i = data->item_ids.find(mid)->second;
			items_sorted.push_back(i);
		}

		// Set suma to zero
		my_infer::set_to_zero(suma, param->K);
		context_items.clear();

		// For each item
		for(int &i : items_sorted) {
			// Get the price
			price = data->get_price(i,s,param);

			// Auxiliary operations for conditionally specified model
			if(param->flag_shuffle==-1) {
				// Set suma accordingly
				my_infer::substract_contribution_sum_alpha(t,i,1.0,data,param,pvar,suma);

				// Set context_items
				context_items.clear();
				for(const int &idx_j : data->items_per_trans.get_object(t)) {
					if(idx_j!=i) {
						context_items.push_back(idx_j);
					}
				}
			}

			// Bernoulli model
			if(param->flag_likelihood==0) {
				// ** POSITIVES **

				// Set sviFactor
				sviFactor = static_cast<double>(data->Ntrans)/static_cast<double>(batchsize*Ncopies);

				// Compute the sigmoid
				mm = my_sigmoid(my_infer::compute_mean(data,param,pvar,t,i,u,s,&context_items,&argmax,eta_base));
				dL_deta = 1.0-mm;
				dL_deta *= sviFactor;

				// Increase log_p
				my_infer::increase_double(logp,logp_mutex,sviFactor*my_log(mm));

				// Increase gradients
				my_infer::increase_gradients_t_i(data,hyper,param,pvar,suma,i,u,s,t,price,dL_deta,&context_items,argmax);

				// ** NEGATIVES **

				// Set sviFactor
				sviFactor *= (data->Nitems-static_cast<double>(data->lines_per_trans.get_object(t).size()))/static_cast<double>(param->negsamples);
				sviFactor *= param->zeroFactor;

				// Sample negative items from the corresponding distribution
				count_i = 0;
				while(count_i<param->negsamples) {
					i_prime = my_infer::take_negative_sample(i,semilla,semilla_mutex,data,param);
					// If this is indeed a negative sample
					if(i_prime!=i && !my_infer::elem_in_vector(&context_items,i_prime)) {
						// Increase the counts of processed negative samples
						count_i++;
						// Compute the sigmoid
						price_prime = data->get_price(i_prime,s,param);
						mm_prime = my_sigmoid(my_infer::compute_mean(data,param,pvar,t,i_prime,u,s,&context_items,&argmax_prime,eta_base));
						dL_deta = -mm_prime;
						dL_deta *= sviFactor;
						// Increase log_p
						my_infer::increase_double(logp,logp_mutex,sviFactor*my_log(1.0-mm_prime));
						// Increase gradients
						my_infer::increase_gradients_t_i(data,hyper,param,pvar,suma,i_prime,u,s,t,price_prime,dL_deta,&context_items,argmax_prime);
					}
				}
			// One-vs-each bound
			} else if(param->flag_likelihood==1) {
				// Set sviFactor
				sviFactor = static_cast<double>(data->Ntrans)/static_cast<double>(batchsize*Ncopies);
				sviFactor *= (data->Nitems-static_cast<double>(context_items.size())-1.0)/static_cast<double>(param->negsamples);

				// ** POSITIVES **

				// Compute eta
				mm = my_infer::compute_mean(data,param,pvar,t,i,u,s,&context_items,&argmax,eta_base);

				// ** NEGATIVES **
				count_i = 0;
				while(count_i<param->negsamples) {
					i_prime = my_infer::take_negative_sample(i,semilla,semilla_mutex,data,param);
					// If this is indeed a negative sample
					if(i_prime!=i && !my_infer::elem_in_vector(&context_items,i_prime)) {
						// Increase the counts of processed negative samples
						count_i++;
						// Compute eta for i_prime
						price_prime = data->get_price(i_prime,s,param);
						mm_prime = my_infer::compute_mean(data,param,pvar,t,i_prime,u,s,&context_items,&argmax_prime,eta_base);
						// Compute the sigmoid
						sigmoid = my_sigmoid(mm-mm_prime);
						dL_deta = 1.0-sigmoid;
						dL_deta *= sviFactor;
						// Increase log_p
						my_infer::increase_double(logp,logp_mutex,sviFactor*my_log(sigmoid));
						// Increase gradients
						my_infer::increase_gradients_t_i(data,hyper,param,pvar,suma,i,u,s,t,price,dL_deta,&context_items,argmax);
						my_infer::increase_gradients_t_i(data,hyper,param,pvar,suma,i_prime,u,s,t,price_prime,-dL_deta,&context_items,argmax_prime);
					}
				}
			// Within-group softmax
			} else if(param->flag_likelihood==3) {
				// Set sviFactor
				sviFactor = static_cast<double>(data->Ntrans)/static_cast<double>(batchsize*Ncopies);
				std::vector<double> probs;
				std::vector<int> neg_items;
				std::vector<int> neg_argmax;
				std::vector<double> price_neg_items;
				double maximo = -myINFINITY;

				// ** POSITIVES **

				// Get itemgroup
				int g_i = data->group_per_item.get_object(i);

				// Compute eta
				mm = my_infer::compute_mean(data,param,pvar,t,i,u,s,&context_items,&argmax,eta_base);
				probs.insert(probs.begin(), mm);
				maximo = my_max(mm,maximo);

				// ** NEGATIVES **
				for(int &jj: data->items_per_group.get_object(g_i)) {
					i_prime = jj;
					// If this is indeed a negative sample
					if(i_prime!=i && !my_infer::elem_in_vector(&context_items,i_prime)) {
						// Compute eta for i_prime
						price_prime = data->get_price(i_prime,s,param);
						mm_prime = my_infer::compute_mean(data,param,pvar,t,i_prime,u,s,&context_items,&argmax_prime,eta_base);
						// Keep track of this item
						maximo = my_max(mm_prime,maximo);
						neg_items.insert(neg_items.begin(), i_prime);
						neg_argmax.insert(neg_argmax.begin(), argmax_prime);
						price_neg_items.insert(price_neg_items.begin(), price_prime);
						probs.insert(probs.begin(), mm_prime);
						// Increase the counts of processed negative samples
						count_i++;
					}
				}
				// Normalize
				double sum_exp = 0.0;
				for(unsigned int ns=0; ns<probs.size(); ns++) {
					probs.at(ns) = my_exp(probs.at(ns)-maximo);
					sum_exp += probs.at(ns);
				}
				for(unsigned int ns=0; ns<probs.size(); ns++) {
					probs.at(ns) /= sum_exp;
				}
				// Increase logp
				my_infer::increase_double(logp,logp_mutex,sviFactor*my_log(probs.back()));
				// Increase grads
				dL_deta = sviFactor*(1.0-probs.back());
				my_infer::increase_gradients_t_i(data,hyper,param,pvar,suma,i,u,s,t,price,dL_deta,&context_items,argmax);
				for(unsigned int ns=0; ns<neg_items.size(); ns++) {
					dL_deta = -sviFactor*probs.at(ns);
					i_prime = neg_items.at(ns);
					argmax_prime = neg_argmax.at(ns);
					price_prime = price_neg_items.at(ns);
					my_infer::increase_gradients_t_i(data,hyper,param,pvar,suma,i_prime,u,s,t,price_prime,dL_deta,&context_items,argmax_prime);
				}
			// Exact softmax
			} else if(param->flag_likelihood==4) {
				// Set sviFactor
				sviFactor = static_cast<double>(data->Ntrans)/static_cast<double>(batchsize*Ncopies);
				std::vector<double> probs;
				std::vector<int> neg_items;
				std::vector<int> neg_argmax;
				std::vector<double> price_neg_items;
				double maximo = -myINFINITY;

				// ** POSITIVES **

				// Compute eta
				mm = my_infer::compute_mean(data,param,pvar,t,i,u,s,&context_items,&argmax,eta_base);
				probs.insert(probs.begin(), mm);
				maximo = my_max(mm,maximo);

				// ** NEGATIVES **
				for(i_prime=0; i_prime<data->Nitems; i_prime++) {
					// If this is indeed a negative sample
					if(i_prime!=i && !my_infer::elem_in_vector(&context_items,i_prime)) {
						// Compute eta for i_prime
						price_prime = data->get_price(i_prime,s,param);
						mm_prime = my_infer::compute_mean(data,param,pvar,t,i_prime,u,s,&context_items,&argmax_prime,eta_base);
						// Keep track of this item
						maximo = my_max(mm_prime,maximo);
						neg_items.insert(neg_items.begin(), i_prime);
						neg_argmax.insert(neg_argmax.begin(), argmax_prime);
						price_neg_items.insert(price_neg_items.begin(), price_prime);
						probs.insert(probs.begin(), mm_prime);
						// Increase the counts of processed negative samples
						count_i++;
					}
				}
				// Normalize
				double sum_exp = 0.0;
				for(unsigned int ns=0; ns<probs.size(); ns++) {
					probs.at(ns) = my_exp(probs.at(ns)-maximo);
					sum_exp += probs.at(ns);
				}
				for(unsigned int ns=0; ns<probs.size(); ns++) {
					probs.at(ns) /= sum_exp;
				}
				// Increase logp
				my_infer::increase_double(logp,logp_mutex,sviFactor*my_log(my_max(1.0e-12,probs.back())));
				// Increase grads
				dL_deta = sviFactor*(1.0-probs.back());
				my_infer::increase_gradients_t_i(data,hyper,param,pvar,suma,i,u,s,t,price,dL_deta,&context_items,argmax);
				for(unsigned int ns=0; ns<neg_items.size(); ns++) {
					dL_deta = -sviFactor*probs.at(ns);
					i_prime = neg_items.at(ns);
					argmax_prime = neg_argmax.at(ns);
					price_prime = price_neg_items.at(ns);
					my_infer::increase_gradients_t_i(data,hyper,param,pvar,suma,i_prime,u,s,t,price_prime,dL_deta,&context_items,argmax_prime);
				}
			} else {
				std::cerr << "[ERR] The 'likelihood' parameter cannot take value " << std::to_string(param->flag_likelihood) << endl;
				assert(0);
			}

			// Increase suma
			if(param->flag_shuffle!=-1) {
				context_items.push_back(i);
				for(int k=0; k<param->K; k++) {
					suma[k] += pvar->alpha.get_object(i,k).z;
				}
			}
		}
	}
	delete [] suma;
	delete [] eta_base;
}

void compute_val_likelihood_ll(int thread_id, my_data *data, const my_hyper *hyper, const my_param *param, my_pvar *pvar, \
							   int i_checkout, int ll, double *llh) {
	int u;
	int i;
	int s;
	int t;
	double *p_item = new double[data->Nitems];
	double *eta_base = new double[data->Nitems];
	int argmax;
	bool exclude_checkout = true;
	std::vector<int> elem_context;

	u = data->obs_val.y_user[ll];
	i = data->obs_val.y_item[ll];
	s = data->obs_val.y_sess[ll];
	t = data->obs_val.y_trans[ll];

	// If the transaction and the item are found in train.tsv
	if(t>=0 && data->lines_per_item.get_object(i).size()>0) {
		// Compute eta_base for all items
		my_infer::compute_eta_base(data,hyper,param,pvar,t,u,s,eta_base);
		// Compute the mean for all items
		elem_context = data->items_per_trans.get_object(t);
		for(int j=0; j<data->Nitems; j++) {
			if(j!=i_checkout || !exclude_checkout) {
				p_item[j] = my_infer::compute_mean(data,param,pvar,t,j,u,s,&elem_context,&argmax,eta_base);
			} else {
				p_item[j] = -myINFINITY;
			}
		}
		// Compute log-lik
		llh[ll] = p_item[i] - my_logsumexp(p_item,data->Nitems);
	} else {
		llh[ll] = myINFINITY;
	}

	delete [] p_item;
	delete [] eta_base;
}

void compute_test_likelihood_ll(int thread_id, my_data *data, const my_hyper *hyper, const my_param *param, my_pvar *pvar, \
								int i_checkout, int ll, double *llh, Matrix1D<bool> *test_valid_lines) {
	int u;
	int i;
	int s;
	int t;
	double *p_item = new double[data->Nitems];
	double *eta_base = new double[data->Nitems];
	int argmax;
	bool exclude_checkout = true;
	std::vector<int> elem_context;

	u = data->obs_test.y_user[ll];
	i = data->obs_test.y_item[ll];
	s = data->obs_test.y_sess[ll];
	t = data->obs_test.y_trans[ll];

	// If the line is valid
	if(test_valid_lines->get_object(ll)) {
		// Compute eta_base for all items
		my_infer::compute_eta_base(data,hyper,param,pvar,t,u,s,eta_base);
		// Find elements in the context
		for(int &ll_j : data->test_lines_per_trans.get_object(t)) {
			int j = data->obs_test.y_item[ll_j];
			if(j!=i && test_valid_lines->get_object(ll_j)) {
				elem_context.push_back(j);
			}
		}
		// Compute the mean for all items
		for(int j=0; j<data->Nitems; j++) {
			if(j!=i_checkout || !exclude_checkout) {
				p_item[j] = my_infer::compute_mean(data,param,pvar,t,j,u,s,&elem_context,&argmax,eta_base);
			} else {
				p_item[j] = -myINFINITY;
			}
		}
		// Compute log-lik
		llh[ll] = p_item[i] - my_logsumexp(p_item,data->Nitems);
	} else {
		llh[ll] = myINFINITY;
	}

	delete [] p_item;
	delete [] eta_base;
}

void compute_test_baskets_likelihood_tt(int thread_id, my_data *data, const my_hyper *hyper, const my_param *param, my_pvar *pvar, \
										int i_checkout, int t, double *llh_checkout, double *llh_nocheckout, \
										Matrix1D<bool> *test_valid_lines, Matrix1D<int> *test_valid_lines_per_trans) {
	int u;
	int i;
	int s;
	double *p_item = new double[data->Nitems];
	double *eta_base = new double[data->Nitems];
	int argmax;
	std::vector<int> elem_context;
	int T = static_cast<int>(data->obs_test.T);

	// Get lines in test.tsv corresponding to transaction t
	std::vector<int> lines_t = data->test_lines_per_trans.get_object(t);

	// Check if the transaction is valid
	if(test_valid_lines_per_trans->get_object(t)==0) {
		for(int &ll : lines_t) {
			llh_nocheckout[ll] = myINFINITY;
			llh_checkout[ll] = myINFINITY;
		}
		int ll = T+t;
		llh_checkout[ll] = myINFINITY;
		return;
	}

	// Get user and session
	u = data->obs_test.y_user[lines_t.at(0)];
	s = data->obs_test.y_sess[lines_t.at(0)];

	// Compute eta_base for all items
	my_infer::compute_eta_base(data,hyper,param,pvar,t,u,s,eta_base);

	// For each line in transaction t
	for(int &ll : lines_t) {
		// If the line is valid
		if(test_valid_lines->get_object(ll)) {
			// Get item
			i = data->obs_test.y_item[ll];
			// Compute the mean for all items
			for(int j=0; j<data->Nitems; j++) {
				p_item[j] = my_infer::compute_mean(data,param,pvar,t,j,u,s,&elem_context,&argmax,eta_base);
			}
			// Compute log-lik
			llh_checkout[ll] = p_item[i] - my_logsumexp(p_item,data->Nitems);
			// Compute log-lik without checkout item
			if(i_checkout>=0) {
				p_item[i_checkout] = -myINFINITY;
			}
			llh_nocheckout[ll] = p_item[i] - my_logsumexp(p_item,data->Nitems);
			// Add item i to context
			elem_context.push_back(i);
		} else {
			llh_checkout[ll] = myINFINITY;
			llh_nocheckout[ll] = myINFINITY;
		}
	}

	// For the checkout item
	if(i_checkout>=0) {
		i = i_checkout;
		// Compute the mean for all items
		for(int j=0; j<data->Nitems; j++) {
			p_item[j] = my_infer::compute_mean(data,param,pvar,t,j,u,s,&elem_context,&argmax,eta_base);
		}
		// Compute log-lik
		llh_checkout[t+T] = p_item[i] - my_logsumexp(p_item,data->Nitems);
	} else {
		llh_checkout[t+T] = myINFINITY;
	}

	// Free memory
	delete [] p_item;
	delete [] eta_base;
}



#endif
