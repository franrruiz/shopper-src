#ifndef REP_VAR_UTILITIES
#define REP_VAR_UTILITIES

/* Gaussian distribution */
class var_stepsize {
public:

	static double update_G(double grad, double currentG, int step_schedule) {
		// step_schedule: 0=advi, 1=rmsprop, 2=adagrad
		double aux = 0.0;
		if(step_schedule==0) {
			aux = 0.1*my_pow2(grad) + 0.9*currentG;
		} else if(step_schedule==1) {
			aux = 0.1*my_pow2(grad) + 0.9*currentG;
		} else if(step_schedule==2) {
			aux = my_pow2(grad) + currentG;
		} else {
			std::cerr << "[ERR] Wrong value of step_schedule: " << std::to_string(step_schedule) << endl;
			assert(0);
		}
		return aux;
	}

	static double get_stepsize(double eta, double G, int step_schedule) {
		double aux = 0.0;
		if(step_schedule==0) {
			aux = eta/(1.0+sqrt(G));
		} else if(step_schedule==1) {
			aux = eta/sqrt(1.0e-8+G);
		} else if(step_schedule==2) {
			aux = eta/sqrt(1.0e-8+G);
		} else {
			std::cerr << "[ERR] Wrong value of step_schedule: " << std::to_string(step_schedule) << endl;
			assert(0);
		}
		return aux;
	}

};

#endif
