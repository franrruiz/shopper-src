#include "my_headers.hpp"
#include "my_gsl_utilities.hpp"
#include "my_matrices.hpp"
#include "var_utilities.hpp"
#include "var_pointmass.hpp"
#include "var_gaussian.hpp"
#include "var_gamma.hpp"
#include "var_pos_gaussian.hpp"
#include "emb_params.hpp"
#include "emb_io.hpp"
#include "emb_inference.hpp"

int main (int argc, char *argv[]) {
	my_param param;
	my_hyper hyper;
	my_data data;

	/******** Read options from input (argc, argv) ********/
	std::cout << "Initializing program..." << endl;
	my_input::read_input_from_command_line(argc,argv,data,param,hyper);

	/******** Read data from file ********/
	std::cout << "Reading data..." << endl;
	// Read train.tsv, validation.tsv and train.tsv
	my_input::read_data_file(data,param);
	// Create auxiliary data structures
	std::cout << "Computing auxiliary data structures..." << endl;
	data.create_transactions_train(param);
	data.create_sessions_per_user(param);
	data.create_other_data_structs(param);
	std::cout << "Computing auxiliary data structures for test set..." << endl;
	data.create_transactions_test(param);
	// Create output folder
	my_output::create_output_folder(data,param);
	// Print vocabulary
	my_output::write_vocab(data,param);
	// Read prices
	if(param.flag_price>0) {
		std::cout << "Reading prices..." << endl;
		my_input::read_price_file(data,param);
		my_input::normalize_prices(data,param);
		my_output::write_normalized_prices(data,param);
		my_input::take_log_prices(data,param);
	}
	// Read itemgroups
	if(param.flag_day>0 || param.flag_price>0 || param.Kgroup>0 || param.flag_nsFreq>1 || param.flag_likelihood==3) {
		// read itemGroup.tsv
		std::cout << "Reading item groups..." << endl;
		my_input::read_itemgroups(data,param);
	}
	// Read sess_days
	if(param.flag_day>0) {
		// read sess_days.tsv
		std::cout << "Reading session/days mapping..." << endl;
		my_input::read_sess_days(data,param);
	}
	// Create auxiliary data structures (continued)
	data.compute_unigram_distributions(param);

	/******** Write log file ********/
	my_output::create_log_file(data,param,hyper);

	/******** Set the seeds ********/
	srand48(param.seed);
	gsl_rng *semilla = gsl_rng_alloc(gsl_rng_taus);
	gsl_rng_set(semilla,param.seed);

	/******** Allocate memory for variational parameters ********/
	std::cout << "Initializing latent parameters..." << endl;
	my_pvar pvar(data,param);
	// Initialize variational parameters
	pvar.initialize_all(semilla,data,param,hyper);

	/******** Inference algorithm ********/
	std::cout << "Running inference algorithm..." << endl;
	time_t t_ini_abs = time(NULL);
	time_t t_ini;
	time_t t_end;
	bool stop = false;
	param.it = 0;
	int why = -1;
	double val_llh;
	int duration;
	while(!stop) {
		std::cout << " +Iteration " << param.it << "..." << endl;
		t_ini = time(NULL);	// Measure time elapsed

		// Run an inference step
		my_infer::inference_step(data,hyper,param,pvar,semilla);

		// Write time elapsed
		t_end = time(NULL);	// Measure time elapsed
		my_output::write_telapsed(param.outdir,param.it,t_ini,t_end);
		duration = difftime(t_end,t_ini_abs);

		// Compute and write norm
		double auxNormTheta = 0.0;
		double auxNormRho = 0.0;
		if(param.flag_userVec>0) {
			auxNormTheta = my_infer::compute_avg_norm(pvar.theta);
		}
		if(!param.flag_symmetricRho) {
			auxNormRho = my_infer::compute_avg_norm(pvar.rho);
		}
		my_output::write_norm(param.outdir,param.it,auxNormRho,my_infer::compute_avg_norm(pvar.alpha),auxNormTheta);

		// Compute llh + Check stop criteria
		if((param.it%param.rfreq)==0 && param.it>0) {
			my_infer::prepare_test_valid(semilla,data,hyper,param,pvar);
			val_llh = my_infer::compute_val_likelihood(false,duration,data,param,hyper,pvar);
			my_infer::compute_test_performance(false,duration,data,hyper,param,pvar);
			if(!param.noVal && (param.it>3*param.rfreq)) {
				std::cout << " (validation llh relative change (abs value): " << fabs((val_llh-param.prev_val_llh)/param.prev_val_llh) << ")" << endl;
                /*
				// If log likelihood increased, is not zero, and it increased less than 0.000001 of the previous value, set why to zero
				if(val_llh>=param.prev_val_llh && param.prev_val_llh!=0 && fabs((val_llh-param.prev_val_llh)/param.prev_val_llh)<param.valTolerance) {
					stop = true;
					why = 0;
				} else if(val_llh<param.prev_val_llh) {
					// Count the number of times in a row that the likelihood decreased
					(param.n_val_decr)++;
				} else if(val_llh>param.prev_val_llh) {
					param.n_val_decr = 0;
				}
				if(param.n_val_decr>param.valConsecutive) {
					stop = true;
					why = 1;
				}
                */
			}
			param.prev_val_llh = val_llh;
		}
		if(param.it+1>=param.Niter) {
			stop = true;
			why = 2;
		}
		if(!stop) {
			(param.it)++;
			// Save results to file if desired
			if((param.it%param.saveCycle)==0) {
				my_output::write_all_matrices("_it"+std::to_string(param.it),data,hyper,param,pvar);
			}
		}
	}

	/******** Print output ********/
    // Print output matrices
    my_output::write_all_matrices("",data,hyper,param,pvar);
	// Compute final validation llh & Print max file
	my_infer::prepare_test_valid(semilla,data,hyper,param,pvar);
	val_llh = my_infer::compute_val_likelihood(true,duration,data,param,hyper,pvar);
	my_output::write_max_file(param,duration,val_llh,why);
	my_infer::compute_test_performance(true,duration,data,hyper,param,pvar);
	my_infer::compute_test_performance_baskets(true,duration,data,hyper,param,pvar);

	/******** Free memory ********/
	gsl_rng_free(semilla);

	/******** Return ********/
	return 0;
};

