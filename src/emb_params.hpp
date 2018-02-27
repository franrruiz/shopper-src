#ifndef PEMB_PARAMS_HPP
#define PEMB_PARAMS_HPP

class my_param {
public:

	string datadir;
	string outdir;
	int K;
	unsigned long int seed;
	int rfreq;
	int saveCycle;
	int Niter;
	int negsamples;
	double zeroFactor;
	int batchsize;
	string label;
	bool flag_itemIntercept;
	int flag_step_schedule;			// step_schedule: (0=advi), 1=rmsprop, 2=adagrad
	int flag_likelihood;			// likelihood: 0=bernoulli, 1=one-vs-each, 3=within-group softmax, 4=exact softmax
	int flag_shuffle;				// shuffle: -1=conditionally specified, 0=don't, 1+=yes
	int flag_nsFreq;				// nsFreq: -1=uniform; 0=unigram; 1=unigram^(3/4); 2+=biased to item group
	int flag_userVec;
	bool flag_avgContext;
	int flag_price;
	int flag_day;
	bool flag_normPrice;
	bool flag_normPrice_min;
	bool flag_checkout;
	int flag_lookahead;
	bool flag_symmetricRho;			// alpha==rho?
	int Nthreads;
	double eta;
	double gamma;
	double stdIni;
	double valTolerance;
	string iniPath;
	string iniFromGroup;
	int Kgroup;
	bool flag_fixKgroup;
	bool flag_iniThetaVal;
	double iniThetaVal;
	bool flag_iniPriceVal;
	double iniPriceVal;
	int valConsecutive;
	bool noVal;
	bool noTest;

	int lf_keepOnly;
	int lf_keepAbove;
	int lf_flag;

	int it;
	int n_val_decr;
	double prev_val_llh;
		
	my_param() {
		datadir = ".";
		outdir = ".";
		K = 50;
		seed = 0;
		rfreq = 1000;
		saveCycle = 5000;
		negsamples = 50;
		zeroFactor = 0.1;
		batchsize = 1000;
		Niter = 12000;
		flag_avgContext = true;
		flag_itemIntercept = true;
		flag_step_schedule = 1;
		flag_nsFreq = -1;
		flag_likelihood = 1;
		flag_shuffle = 1;
		flag_userVec = 3;
		flag_price = 10;
		flag_day = 10;
		flag_normPrice = true;
		flag_normPrice_min = false;
		flag_checkout = true;
		flag_lookahead = 0;
		flag_symmetricRho = false;
		label = "";
		eta = 0.01;
		gamma = 0.9;
		stdIni = -1.0;
		valTolerance = 0.000001;
		valConsecutive = 5;
		noVal = false;
		noTest = false;
		iniPath = "";
		iniFromGroup = "";
		Kgroup = 0;
		flag_fixKgroup = false;
		flag_iniThetaVal = false;
		iniThetaVal = 0.0;
		flag_iniPriceVal = false;
		iniPriceVal = 0.0;
		Nthreads = 1;

		lf_keepOnly = -1;
		lf_keepAbove = -1;
		lf_flag = 0;

		it = 0;
		prev_val_llh = 0.0;
		n_val_decr = 0;
	}
};

class my_hyper {
public:

	double s2rho;
	double s2alpha;
	double s2theta;
	double s2delta;
	double s2lambda;
	double s2mu;
	double rte_gamma;
	double shp_gamma;
	double rte_beta;
	double shp_beta;

	my_hyper() {
		s2rho = 1.0;
		s2alpha = 1.0;
		s2theta = 1.0;
		s2lambda = 1.0;
		s2delta = 0.01;
		s2mu = 0.01;
		rte_gamma = 0.1/my_pow2(0.01);
		shp_gamma = my_pow2(0.1/0.01);
		rte_beta = 0.1/my_pow2(0.01);
		shp_beta = my_pow2(0.1/0.01);
	}
};

class my_data_aux {
public:
	unsigned int T;			// Number of (user,session,item) triplets
	unsigned int *y_user;	// User idx per (user,session,item) triplet
	unsigned int *y_item;	// Item idx per (user,session,item) triplet
	unsigned int *y_sess;	// Session idx per (user,session,item) triplet
	unsigned int *y_rating;	// Value (rating or #units) per (user,session,item) triplet
	unsigned int *y_trans;	// Transaction idx per (user,session,item) triplet

	my_data_aux() {
		T = 0;
		y_user = nullptr;
		y_item = nullptr;
		y_sess = nullptr;
		y_rating = nullptr;
		y_trans = nullptr;
	}

	my_data_aux(unsigned int T_) {
		allocate_all(T_);
	}

	~my_data_aux() {
		delete_all();
	}

	inline void allocate_all(unsigned int T_) {
		T = T_;
		y_user = new unsigned int[T];
		y_item = new unsigned int[T];
		y_sess = new unsigned int[T];
		y_rating = new unsigned int[T];
		y_trans = new unsigned int[T];
	}

	inline void delete_all() {
		delete [] y_user;
		delete [] y_item;
		delete [] y_sess;
		delete [] y_rating;
		delete [] y_trans;
	}

	my_data_aux & operator=(const my_data_aux &rhs) {
		// Check for self-assignment!
		if(this!=&rhs) {
			// deallocate memory
			delete_all();
			// allocate memory for the contents of rhs
			allocate_all(rhs.T);
			// copy values from rhs
			for(unsigned int n=0; n<rhs.T; n++) {
				y_user[n] = rhs.y_user[n];
				y_item[n] = rhs.y_item[n];
				y_sess[n] = rhs.y_sess[n];
				y_rating[n] = rhs.y_rating[n];
				y_trans[n] = rhs.y_trans[n];
			}
		}
		return *this;
	}
};

class hpf_trans_aux {
public:
	int u;
	int s;

	hpf_trans_aux & operator=(const hpf_trans_aux &rhs) {
		// Check for self-assignment!
		if (this!=&rhs) {
			u = rhs.u;
			s = rhs.s;
		}
		return *this;
	}

	hpf_trans_aux(int uu, int ss) {
		u = uu;
		s = ss;
	}

	bool operator==(const hpf_trans_aux &b) const {
		return (u==b.u)&&(s==b.s);
	}

	bool operator!=(const hpf_trans_aux &b) const {
		return (u!=b.u)||(s!=b.s);
	}

	bool operator<(const hpf_trans_aux &b) const {
		return (u<b.u)||(u==b.u && s<b.s);
	}

	bool operator<=(const hpf_trans_aux &b) const {
		return (u<b.u)||(u==b.u && s<=b.s);
	}

	bool operator>(const hpf_trans_aux &b) const {
		return (u>b.u)||(u==b.u && s>b.s);
	}

	bool operator>=(const hpf_trans_aux &b) const {
		return (u>b.u)||(u==b.u && s>=b.s);
	}
};

class my_data {
public:

	// Integers
	int Nitems;		// Number of items
	int Nusers;		// Number of users
	int Nsessions;	// Number of sessions
	int Ntrans;		// Number of transactions (a transaction is a (user,session) pair)
	int Ndays;		// Number of calendar days
	int Nweekdays;  // Number of weekdays (typically 7)
	int NuserGroups;	// Number of user groups
	int NitemGroups;	// Number of item groups

	// Structs with the actual data
	my_data_aux obs;			// Observations (train)
	my_data_aux obs_test;		// Observations (test)
	my_data_aux obs_val;		// Observations (validation)

	// Prices for each session and item
	Matrix2D<double> price_is;	  // Prices
	Matrix1D<double> price_avg_i; // Average price for each item (across all sessions)
	Matrix1D<double> price_min_i; // Minimum price for each item (across all sessions)

	// sum(log(y!))
	double sum_log_yfact;
	Matrix1D<double> sum_log_yfact_per_trans;

	// Mapping from ids to indices
	std::map<unsigned long long, int> item_ids;		// Map containing the item id's
	std::map<unsigned long long, int> user_ids;		// Map containing the user id's
	std::map<unsigned long long, int> session_ids;	// Map containing the session id's
	std::map<hpf_trans_aux,int> trans_ids;			// Map containing the transaction id's (for train+validation)
	std::map<unsigned long long,int> day_ids;			// Map containing the day id's
	std::map<unsigned long long,int> weekday_ids;		// Map containing the weekday id's
	std::map<unsigned long long,int> usergroup_ids;		// Map containing the usergroups id's
	std::map<unsigned long long,int> itemgroup_ids;		// Map containing the itemgroups id's

	// Misc lists
	Matrix1D<std::vector<int>> sessions_per_user;	// For each user, list of sessions in which that user appears
	Matrix1D<std::vector<int>> trans_per_item;		// For each item, list of transactions in which that item appears
	Matrix1D<std::vector<int>> trans_per_user;		// For each user, list of transactions in which that user appears
	Matrix1D<int> user_per_trans;					// For each transaction, user to which it corresponds
	Matrix1D<int> session_per_trans;				// For each transaction, session to which it corresponds
	Matrix1D<std::vector<int>> lines_per_trans;		// For each transaction, list of "lines" in train.tsv referring to that transaction
	Matrix1D<std::vector<int>> items_per_trans;		// For each transaction, list of items in that transaction
	int maxNi;										// Maximum #items of all transactions
	Matrix1D<std::vector<int>> lines_per_item;		// For each item, list of "lines" in train.tsv in which that item appears
	Matrix1D<std::vector<int>> lines_per_user;		// For each user, list of "lines" in train.tsv in which that user appears
	Matrix1D<int> Nitems_per_trans;					// For each transaction, sum(y)
	Matrix1D<int> sum_sizetrans_per_item;			// For each item, sum_t(length(t)) for all the transactions t in which that item appears
	Matrix1D<int> group_per_user;					// For each user, group to which she belongs
	Matrix1D<int> group_per_item;					// For each item, group to which it belongs
	Matrix1D<std::vector<int>> users_per_group;		// For each usergroup, list of the users that belong to that group
	Matrix1D<std::vector<int>> items_per_group;		// For each itemgroup, list of the items that belong to that group
	Matrix1D<int> day_per_session;					// For each session, day_id to which it corresponds
	Matrix1D<int> weekday_per_session;				// For each session, weekday_id to which it corresponds
	Matrix1D<double> hour_per_session;				// For each session, hour of the day
	Matrix1D<std::vector<int>> sessions_per_day;	// For each day, a list of sessions
	Matrix1D<std::vector<int>> sessions_per_weekday;// For each weekday, a list of sessions
	Matrix3D<std::vector<int>> lines_per_xday;		// For each day, a list of "lines" in train.tsv

	// Valid list of items
	std::vector<unsigned long long> valid_items;	// Used to remove low-frequency items

	// Lists for transactions in test set (same definition as above, but for test set only)
	int test_Ntrans;
	Matrix1D<std::vector<int>> test_lines_per_trans;
	Matrix1D<std::vector<int>> test_items_per_trans;
	Matrix1D<int> test_Nitems_per_trans;
	Matrix1D<int> test_user_per_trans;
	Matrix1D<int> test_session_per_trans;
	Matrix1D<std::vector<int>> test_sessions_per_user;
	std::map<hpf_trans_aux,int> test_trans_ids;			// Map containing the transaction id's (for test)

	// Frequency and counts of items
	double *uniform_dist;
	double *unigram_dist;
	double *unigram_dist_power;
	gsl_ran_discrete_t *negsampling_dis;

	// negsampling distribution specific for itemgroups
	std::vector<gsl_ran_discrete_t*> negsampling_dis_per_group;

	my_data() {
		Nitems = 0;
		Nusers = 0;
		Nsessions = 0;
		Ntrans = 0;
		Ndays = 0;
		Nweekdays = 0;
		NuserGroups = 0;
		NitemGroups = 0;
		test_Ntrans = 0;
		maxNi = 0;
		uniform_dist = nullptr;
		unigram_dist = nullptr;
		unigram_dist_power = nullptr;
		negsampling_dis = nullptr;
	}

	~my_data() {
		delete [] uniform_dist;
		delete [] unigram_dist;
		delete [] unigram_dist_power;
		gsl_ran_discrete_free(negsampling_dis);
		for(auto &d : negsampling_dis_per_group) {
			gsl_ran_discrete_free(d);
		}
	}

	inline unsigned long long find_by_value(const std::map<unsigned long long,int> &ids, int val) {
		for(std::map<unsigned long long,int>::const_iterator iter_i=ids.begin(); iter_i!=ids.end(); ++iter_i) {
			if(iter_i->second == val) {
				return iter_i->first;
			}
		}
		return 0;
	}

	inline double get_price(int i, int s, const my_param &param) {
		return((param.flag_price>0)?price_is.get_object(i,s):0.0);
	}

	inline double get_price(int i, int s, const my_param *param) {
		return((param->flag_price>0)?price_is.get_object(i,s):0.0);
	}

	inline int get_order_in_trans(int i, int t) {
		bool found = false;
		int y = -1;
		int j;
		int count = 0;
		std::vector<int>::iterator iter = items_per_trans.get_object(t).begin();
		while(!found && iter!=items_per_trans.get_object(t).end()) {
			j = *iter;
			if(j==i) {
				found = true;
				y = count;
			}
			count++;
			++iter;
		}
		return y;
	} 

	inline double get_items_in_trans(int i, int t) {
		bool found = false;
		int ll;
		double y = -1.0;
		std::vector<int>::iterator iter = lines_per_trans.get_object(t).begin();
		while(!found && iter!=lines_per_trans.get_object(t).end()) {
			ll = *iter;
			if(i==static_cast<int>(obs.y_item[ll])) {
				found = true;
				y = static_cast<double>(obs.y_rating[ll]);
			}
			++iter;
		}
		return y;
	}

	void compute_unigram_distributions(const my_param &param) {
		uniform_dist = new double[Nitems];
		unigram_dist = new double[Nitems];
		unigram_dist_power = new double[Nitems];

		// Initialize to zero
		for(int i=0; i<Nitems; i++) {
			unigram_dist[i] = 0.0;
		}

		// Count item occurrences
		double suma = 0.0;
		for(unsigned int t=0; t<obs.T; t++) {
			int i = obs.y_item[t];
			unigram_dist[i] += 1.0;
			suma += 1.0;
		}
		if(param.flag_checkout) {
			unsigned long long mid = ULLONG_MAX;
			int i = item_ids.find(mid)->second;
			unigram_dist[i] = static_cast<double>(Ntrans);
			suma += static_cast<double>(Ntrans);
		}
		// Normalize counts
		for(int i=0; i<Nitems; i++) {
			unigram_dist[i] /= suma;
			uniform_dist[i] = 1.0/static_cast<double>(Nitems);
		}
		// Raise to the power of 3/4
		suma = 0.0;
		for(int i=0; i<Nitems; i++) {
			unigram_dist_power[i] = pow(unigram_dist[i],0.75);
			suma += unigram_dist_power[i];
		}
		// Normalize the power'ed counts
		for(int i=0; i<Nitems; i++) {
			unigram_dist_power[i] /= suma;
		}
		// Create preprocessed gsl_ran_discrete_t variables for faster negative sampling
		if(param.flag_nsFreq==-1) {
			negsampling_dis = gsl_ran_discrete_preproc(Nitems,uniform_dist);
		} else if(param.flag_nsFreq==0) {
			negsampling_dis = gsl_ran_discrete_preproc(Nitems,unigram_dist);
		} else if(param.flag_nsFreq==1) {
			negsampling_dis = gsl_ran_discrete_preproc(Nitems,unigram_dist_power);
		} else if(param.flag_nsFreq<2) {
			std::cerr << "[ERR] Wrong value for -nsFreq: " << std::to_string(param.flag_nsFreq) << endl;
			assert(0);
		}

		// Now compute the per-group negsampling distribution
		if(param.flag_nsFreq>1){
			for(int g=0; g<NitemGroups; g++) {
				double *aux_dis = new double[Nitems];
				suma = 0.0;
				for(int i=0; i<Nitems; i++) {
					if(group_per_item.get_object(i)==g) {
						aux_dis[i] = static_cast<double>(param.flag_nsFreq);
					} else {
						aux_dis[i] = 1.0;
					}
					suma += aux_dis[i];
				}
				for(int i=0; i<Nitems; i++) {
					aux_dis[i] /= suma;
				}
				gsl_ran_discrete_t *n_dis = gsl_ran_discrete_preproc(Nitems,aux_dis);
				negsampling_dis_per_group.push_back(n_dis);
				delete [] aux_dis;
			}
		}
	}

	void create_transactions_train(const my_param &param) {
		int u;
		int s;
		Ntrans = 0;
		for(unsigned int t=0; t<obs.T; t++) {
			u = obs.y_user[t];
			s = obs.y_sess[t];
			hpf_trans_aux tid = hpf_trans_aux(u,s);
			if(trans_ids.find(tid)==trans_ids.end()) {
				trans_ids.insert(std::pair<hpf_trans_aux,int>(tid,Ntrans));
				Ntrans += 1;
			}
		}
	}

	void create_sessions_per_user(const my_param &param) {
		sessions_per_user = Matrix1D<std::vector<int>>(Nusers);
		int u;
		int s;
		for(auto const &it : trans_ids) {
			u = it.first.u;
			s = it.first.s;
			sessions_per_user.get_object(u).push_back(s);
		}
	}

	void create_other_data_structs(const my_param &param) {
		int u;
		int i;
		int idx_trans;
		int s;

		// Compute sum_log_yfact
		sum_log_yfact = 0.0;
		for(unsigned int t=0; t<obs.T; t++) {
			sum_log_yfact += my_logfactorial(obs.y_rating[t]);
		}

		// Create session_per_trans, user_per_trans
		session_per_trans = Matrix1D<int>(Ntrans);
		user_per_trans = Matrix1D<int>(Ntrans);
		for(auto const &it : trans_ids) {
			u = it.first.u;
			s = it.first.s;
			idx_trans = it.second;
			user_per_trans.set_object(idx_trans,u);
			session_per_trans.set_object(idx_trans,s);
		}

		// Create lists mapping from lines<-->trans, and create mapping item-->transactions
		trans_per_item = Matrix1D<std::vector<int>>(Nitems);
		lines_per_item = Matrix1D<std::vector<int>>(Nitems);
		lines_per_user = Matrix1D<std::vector<int>>(Nusers);
		lines_per_trans = Matrix1D<std::vector<int>>(Ntrans);
		items_per_trans = Matrix1D<std::vector<int>>(Ntrans);
		Nitems_per_trans = Matrix1D<int>(Ntrans);
		set_to_zero(Nitems_per_trans);
		for(unsigned int t=0; t<obs.T; t++) {
			u = obs.y_user[t];
			s = obs.y_sess[t];
			i = obs.y_item[t];

			idx_trans = get_transaction(u,s);
			trans_per_item.get_object(i).push_back(idx_trans);
			lines_per_trans.get_object(idx_trans).push_back(t);
			items_per_trans.get_object(idx_trans).push_back(i);
			lines_per_item.get_object(i).push_back(t);
			lines_per_user.get_object(u).push_back(t);
			obs.y_trans[t] = idx_trans;

			int aux_int = Nitems_per_trans.get_object(idx_trans);
			aux_int += obs.y_rating[t];
			Nitems_per_trans.set_object(idx_trans,aux_int);
		}

		// Compute sum_log_yfact_per_trans
		sum_log_yfact_per_trans = Matrix1D<double>(Ntrans);
		for(int t=0; t<Ntrans; t++) {
			double sum = 0.0;
			for(int &ll : lines_per_trans.get_object(t)) {
				sum += my_logfactorial(obs.y_rating[ll]);
			}
			sum_log_yfact_per_trans.set_object(t,sum);
		}

		// Set maxNi
		for(int t=0; t<Ntrans; t++) {
			int Nt = items_per_trans.get_object(t).size();
			maxNi = my_max(Nt,maxNi);
		}

		// Create obs_val.y_trans[]
		if(!param.noVal){
			for(unsigned int t=0; t<obs_val.T; t++) {
				u = obs_val.y_user[t];
				s = obs_val.y_sess[t];
				i = obs_val.y_item[t];

				idx_trans = get_transaction(u,s);
				if(idx_trans<0) {
					std::cerr << "[WARN] Line " << (t+1) << " of validation.tsv contains a (user,session) pair that is not present in train.tsv." \
							  << " This line will be ignored" << endl;
				}
				obs_val.y_trans[t] = idx_trans;
			}
		}

		// Create trans_per_user
		trans_per_user = Matrix1D<std::vector<int>>(Nusers);
		for(int t=0; t<Ntrans; t++) {
			u = user_per_trans.get_object(t);
			trans_per_user.get_object(u).push_back(t);
		}

		// Create sum_sizetrans_per_item
		sum_sizetrans_per_item = Matrix1D<int>(Nitems);
		for(int i=0; i<Nitems; i++) {
			int cc = 0;
			// for all transactions in which item i appears
			for(int t : trans_per_item.get_object(i)) {
				cc += items_per_trans.get_object(t).size();
			}
			sum_sizetrans_per_item.set_object(i,cc);
		}
	}

	void create_transactions_test(const my_param &param) {
		if(param.noTest) {
			return;
		}

		int u;
		int s;
		int i;
		int aux;

		// Create test_trans_ids
		test_Ntrans = 0;
		for(unsigned int t=0; t<obs_test.T; t++) {
			u = obs_test.y_user[t];
			s = obs_test.y_sess[t];
			hpf_trans_aux tid = hpf_trans_aux(u,s);
			if(test_trans_ids.find(tid)==test_trans_ids.end()) {
				test_trans_ids.insert(std::pair<hpf_trans_aux,int>(tid,test_Ntrans));
				test_Ntrans += 1;
			}
		}

		// Create test_sessions_per_user
		test_sessions_per_user = Matrix1D<std::vector<int>>(Nusers);
		for(auto const &it : test_trans_ids) {
			u = it.first.u;
			s = it.first.s;
			test_sessions_per_user.get_object(u).push_back(s);
		}

		// Create test_user_per_trans and test_session_per_trans
		test_user_per_trans = Matrix1D<int>(test_Ntrans);
		test_session_per_trans = Matrix1D<int>(test_Ntrans);
		for(auto const &it : test_trans_ids) {
			u = it.first.u;
			s = it.first.s;
			int t = it.second;
			test_user_per_trans.set_object(t,u);
			test_session_per_trans.set_object(t,s);
		}

		// Create test_lines_per_trans, test_items_per_trans, and obs_test.y_trans
		test_lines_per_trans = Matrix1D<std::vector<int>>(test_Ntrans);
		test_items_per_trans = Matrix1D<std::vector<int>>(test_Ntrans);
		test_Nitems_per_trans = Matrix1D<int>(test_Ntrans);
		set_to_zero(test_Nitems_per_trans);
		for(unsigned int ll=0; ll<obs_test.T; ll++) {
			u = obs_test.y_user[ll];
			s = obs_test.y_sess[ll];
			i = obs_test.y_item[ll];
			int t = test_get_transaction(u,s);
			if(t<0) {
				std::cerr << "[ERR] Transaction not found (my_data::create_transactions_test)" << endl;
				assert(0);
			}
			obs_test.y_trans[ll] = t;
			test_lines_per_trans.get_object(t).push_back(ll);
			test_items_per_trans.get_object(t).push_back(i);
			aux = test_Nitems_per_trans.get_object(t);
			aux += static_cast<int>(obs_test.y_rating[ll]);
			test_Nitems_per_trans.set_object(t,aux);
		}
	}

	inline int get_transaction(int u, int s) {
		int t = -1;
		hpf_trans_aux tid = hpf_trans_aux(u,s);
		std::map<hpf_trans_aux,int>::const_iterator iter = trans_ids.find(tid);
		if(iter!=trans_ids.end()) {
			t = iter->second;
		}
		return t;
	}

	inline int test_get_transaction(int u, int s) {
		int t = -1;
		hpf_trans_aux tid = hpf_trans_aux(u,s);
		std::map<hpf_trans_aux,int>::const_iterator iter = test_trans_ids.find(tid);
		if(iter!=test_trans_ids.end()) {
			t = iter->second;
		}
		return t;
	}

	inline void set_to_zero(Matrix1D<int> &m) {
		for(int n=0; n<m.get_size1(); n++) {
			m.set_object(n,0);
		}
	}

	inline void set_to_zero(Matrix2D<int> &m) {
		for(int n=0; n<m.get_size1(); n++) {
			for(int k=0; k<m.get_size2(); k++) {
				m.set_object(n,k,0);
			}
		}
	}
};

class my_pvar {
public:
	Matrix2D<var_gaussian> rho;	  // embedding vectors
	Matrix2D<var_gaussian> alpha;	// context vectors
	Matrix2D<var_gaussian> theta;	// user vectors
	Matrix1D<var_gaussian> lambda0;  // item intercepts
	Matrix2D<double> sum_alpha;	  // auxiliary variable (sum of alpha's in the context)
	Matrix2D<var_gamma> gamma;	   // price sensitivity vectors (per-user)
	Matrix2D<var_gamma> beta;		// price sensitivity vectors (per-item)
	Matrix2D<var_gaussian> delta;	// seasonal effects
	Matrix2D<var_gaussian> mu;	   // seasonal effects (per-item)
	Matrix2D<double> prod_rho_alpha;		// products rho*alpha
	double *d_prod_rho_alpha;				// products rho*alpha (device vector)
	Matrix2D<double> prod_theta_alpha;		// products theta*alpha
	Matrix2D<double> prod_gamma_beta;		// products gamma*beta
	Matrix2D<double> prod_delta_mu;			// products delta*mu

	my_pvar(const my_data &data, const my_param &param) {
		int sizeRho = data.Nitems;
		int sizeSum = param.K;
		
		rho = Matrix2D<var_gaussian>(sizeRho,param.K);
		alpha = Matrix2D<var_gaussian>(data.Nitems,param.K);
		sum_alpha = Matrix2D<double>(data.Ntrans,sizeSum);
		prod_rho_alpha = Matrix2D<double>(data.Nitems,data.Nitems);
		d_allocate(&d_prod_rho_alpha, data.Nitems*data.Nitems);
		if(param.flag_userVec>0) {
			theta = Matrix2D<var_gaussian>(data.Nusers,param.K);
			prod_theta_alpha = Matrix2D<double>(data.Nusers,data.Nitems);
		}
		if(param.flag_itemIntercept) {
			lambda0 = Matrix1D<var_gaussian>(data.Nitems);
		}
		if(param.flag_price>0) {
			gamma = Matrix2D<var_gamma>(data.Nusers, param.flag_price);
			beta = Matrix2D<var_gamma>(data.NitemGroups, param.flag_price);
			prod_gamma_beta = Matrix2D<double>(data.Nusers,data.NitemGroups);
		}
		if(param.flag_day>0) {
			delta = Matrix2D<var_gaussian>(data.Ndays, param.flag_day);
			mu = Matrix2D<var_gaussian>(data.NitemGroups, param.flag_day);
			prod_delta_mu = Matrix2D<double>(data.Ndays,data.NitemGroups);
		}
	}

	~my_pvar() {
		cudaFree(d_prod_rho_alpha);
	}

	void initialize_all(gsl_rng *semilla, my_data &data, const my_param &param, const my_hyper &hyper) {
		double val_ini = 0.0;
		// initialize rho
		if(!param.flag_symmetricRho) {
			if(param.iniFromGroup!="") {
				std::cerr << "[ERR] iniFromGroup is not implemented" << endl;
				assert(0);
			  	//initialize_from_group_file(param.iniFromGroup+"/param_rho.tsv",data.item_ids,data.itemgroup_ids,data.group_per_item,rho,param,val_ini,false,semilla);
			} else if(param.iniPath=="") {
				// initialize randomly
				initialize_matrix_randomly(semilla,rho,val_ini,param.stdIni,false);
			} else {
				// initialize from file
			  	initialize_from_file(param.iniPath+"/param_rho.tsv",data.item_ids,rho,param,false);
			}
		}

		// initialize alpha
		if(param.iniFromGroup!="") {
			std::cerr << "[ERR] iniFromGroup is not implemented" << endl;
			assert(0);
		  	//initialize_from_group_file(param.iniFromGroup+"/param_alpha.tsv",data.item_ids,data.itemgroup_ids,data.group_per_item,alpha,param,val_ini,false,semilla);
		} else if(param.iniPath=="") {
			// initialize randomly
			initialize_matrix_randomly(semilla,alpha,val_ini,param.stdIni,false);
		} else {
			// initialize from file
			initialize_from_file(param.iniPath+"/param_alpha.tsv",data.item_ids,alpha,param,false);
		}

		// initialize theta
		if(param.flag_userVec>0) {
			if(param.flag_iniThetaVal) {
				// initialize to fixed value
				for(int u=0; u<data.Nusers; u++) {
					for(int k=0; k<param.K; k++) {
						theta.get_object(u,k).initialize(param.iniThetaVal,false);
					}
				}
			} else if(param.iniPath=="") {
				// initialize randomly
				initialize_matrix_randomly(semilla,theta,val_ini,param.stdIni,false);
			} else {
				// initialize from file
				initialize_from_file(param.iniPath+"/param_theta.txt",data.user_ids,theta,param,false);
			}
		}
		
		// initialize intercepts lambda0
		if(param.flag_itemIntercept) {
			if(param.iniPath=="") {
				// initialize randomly
				initialize_matrix_randomly(semilla,lambda0,val_ini,param.stdIni,false);
			} else {
				// initialize from file
				initialize_from_file(param.iniPath+"/param_lambda0.txt",data.item_ids,lambda0,param,false);
			}
		}

		// initialize price sensitivities (beta, gamma)
		if(param.flag_price>0) {
			if(param.flag_iniPriceVal) {
				// initialize to fixed value
				for(int u=0; u<data.Nusers; u++) {
					for(int k=0; k<param.flag_price; k++) {
						gamma.get_object(u,k).initialize(param.iniPriceVal,true);
					}
				}
				for(int i=0; i<data.Nitems; i++) {
					for(int k=0; k<param.flag_price; k++) {
						beta.get_object(i,k).initialize(param.iniPriceVal,true);
					}
				}
			} else if(param.iniPath=="") {
				// initialize randomly
				initialize_matrix_randomly(semilla,gamma,0.1/sqrt(param.flag_price),param.stdIni,true);
				initialize_matrix_randomly(semilla,beta,0.1/sqrt(param.flag_price),param.stdIni,true);
			} else {
				// initialize from file
				initialize_from_file(param.iniPath+"/param_gamma.txt",data.user_ids,gamma,param,true);
				initialize_from_file(param.iniPath+"/param_beta.txt",data.itemgroup_ids,beta,param,true);
			}

			// initialize per-day effect variables
			if(param.flag_day>0) {
				if(param.iniPath=="") {
					// initialize randomly
					initialize_matrix_randomly(semilla,mu,val_ini,param.stdIni,false);
					initialize_matrix_randomly(semilla,delta,val_ini,param.stdIni,false);
				} else {
					// initialize from file
					initialize_from_file(param.iniPath+"/param_mu.txt",data.itemgroup_ids,mu,param,true);
					initialize_from_file(param.iniPath+"/param_delta.txt",data.day_ids,delta,param,true);
				}
			}
		}
	}

	static void initialize_matrix_randomly(gsl_rng *semilla, Matrix1D<var_pointmass> &M, double vv, double ss, bool pp) {
		for(int i=0; i<M.get_size1(); i++) {
			M.get_object(i).initialize_random(semilla,vv,pp,0.01);
		}
	}

	static void initialize_matrix_randomly(gsl_rng *semilla, Matrix1D<var_gaussian> &M, double vv, double ss, bool pp) {
		for(int i=0; i<M.get_size1(); i++) {
			M.get_object(i).initialize_random(semilla,vv,ss,0.01);
		}
	}

	static void initialize_matrix_randomly(gsl_rng *semilla, Matrix1D<var_gamma> &M, double vv, double ss, bool pp) {
		for(int i=0; i<M.get_size1(); i++) {
			M.get_object(i).initialize_random(semilla,vv,ss,0.01);
		}
	}

	static void initialize_matrix_randomly(gsl_rng *semilla, Matrix2D<var_pointmass> &M, double vv, double ss, bool pp) {
		for(int i=0; i<M.get_size1(); i++) {
			for(int k=0; k<M.get_size2(); k++) {
				M.get_object(i,k).initialize_random(semilla,vv,pp,0.01);
			}
		}
	}

	static void initialize_matrix_randomly(gsl_rng *semilla, Matrix2D<var_gaussian> &M, double vv, double ss, bool pp) {
		for(int i=0; i<M.get_size1(); i++) {
			for(int k=0; k<M.get_size2(); k++) {
				M.get_object(i,k).initialize_random(semilla,vv,ss,0.01);
			}
		}
	}

	static void initialize_matrix_randomly(gsl_rng *semilla, Matrix2D<var_gamma> &M, double vv, double ss, bool pp) {
		for(int i=0; i<M.get_size1(); i++) {
			for(int k=0; k<M.get_size2(); k++) {
				M.get_object(i,k).initialize_random(semilla,vv,ss,0.01);
			}
		}
	}

	static void initialize_from_file(string fname, const std::map<unsigned long long, int> &ids, Matrix1D<var_pointmass> &M, const my_param &param, bool pp) {
		FILE *fin = fopen(fname.c_str(),"r");
	  	if(!fin) {
	  		std::cerr << "[ERR] Unable to open " << fname << endl;
	  		assert(0);
	  	}
	  	unsigned long long id;
	  	int ll;
	  	int idx;
	  	double vv;
		while(!feof(fin)){
			// Read a line
			fscanf(fin,"%d\t%llu\t",&ll,&id);
			std::map<unsigned long long,int>::const_iterator iter = ids.find(id);
			if(iter==ids.end()) {
		  		std::cerr << "[ERR] Error reading line " << ll << " of " << fname << endl;
		  		std::cerr << "	  Index not found in data files" << endl;
		  		assert(0);
			}
			idx = iter->second;
			// Read the value
			fscanf(fin,"%lf\n",&vv);
			M.get_object(idx).initialize(vv,pp);
		}
	  	fclose(fin);
	}

	static void initialize_from_file(string fname, const std::map<unsigned long long, int> &ids, Matrix1D<var_gaussian> &M, const my_param &param, bool pp) {
		string fname1 = fname + "_mean.tsv";
		string fname2 = fname + "_std.tsv";
		FILE *fin1 = fopen(fname1.c_str(),"r");
	  	if(!fin1) {
	  		std::cerr << "[ERR] Unable to open " << fname1 << endl;
	  		assert(0);
	  	}
		FILE *fin2 = fopen(fname2.c_str(),"r");
	  	if(!fin2) {
	  		std::cerr << "[ERR] Unable to open " << fname2 << endl;
	  		assert(0);
	  	}
	  	unsigned long long id;
	  	int ll;
	  	int idx;
	  	double mm;
	  	double ss;
		while(!feof(fin1)){
			// Read a line
			fscanf(fin1,"%d\t%llu\t",&ll,&id);
			fscanf(fin2,"%d\t%llu\t",&ll,&id);
			std::map<unsigned long long,int>::const_iterator iter = ids.find(id);
			if(iter==ids.end()) {
		  		std::cerr << "[ERR] Error reading line " << ll << " of " << fname2 << endl;
		  		std::cerr << "	  Index not found in data files" << endl;
		  		assert(0);
			}
			idx = iter->second;
			// Read the value
			fscanf(fin1,"%lf\n",&mm);
			fscanf(fin2,"%lf\n",&ss);
			M.get_object(idx).initialize(mm,ss);
		}
	  	fclose(fin1);
	  	fclose(fin2);
	}

	static void initialize_from_file(string fname, const std::map<unsigned long long, int> &ids, Matrix1D<var_gamma> &M, const my_param &param, bool pp) {
		string fname1 = fname + "_shp.tsv";
		string fname2 = fname + "_rte.tsv";
		FILE *fin1 = fopen(fname1.c_str(),"r");
	  	if(!fin1) {
	  		std::cerr << "[ERR] Unable to open " << fname1 << endl;
	  		assert(0);
	  	}
		FILE *fin2 = fopen(fname2.c_str(),"r");
	  	if(!fin2) {
	  		std::cerr << "[ERR] Unable to open " << fname2 << endl;
	  		assert(0);
	  	}
	  	unsigned long long id;
	  	int ll;
	  	int idx;
	  	double ss;
	  	double rr;
		while(!feof(fin1)){
			// Read a line
			fscanf(fin1,"%d\t%llu\t",&ll,&id);
			fscanf(fin2,"%d\t%llu\t",&ll,&id);
			std::map<unsigned long long,int>::const_iterator iter = ids.find(id);
			if(iter==ids.end()) {
		  		std::cerr << "[ERR] Error reading line " << ll << " of " << fname2 << endl;
		  		std::cerr << "	  Index not found in data files" << endl;
		  		assert(0);
			}
			idx = iter->second;
			// Read the value
			fscanf(fin1,"%lf\n",&ss);
			fscanf(fin2,"%lf\n",&rr);
			M.get_object(idx).initialize(ss,ss/rr);
		}
	  	fclose(fin1);
	  	fclose(fin2);
	}

	static void initialize_from_file(string fname, const std::map<unsigned long long, int> &ids, Matrix2D<var_pointmass> &M, const my_param &param, bool pp) {
		FILE *fin = fopen(fname.c_str(),"r");
	  	if(!fin) {
	  		std::cerr << "[ERR] Unable to open " << fname << endl;
	  		assert(0);
	  	}
	  	unsigned long long id;
	  	int ll;
	  	int idx;
	  	double vv;
		while(!feof(fin)){
			// Read a line
			fscanf(fin,"%d\t%llu\t",&ll,&id);
			std::map<unsigned long long,int>::const_iterator iter = ids.find(id);
			if(iter==ids.end()) {
		  		std::cerr << "[ERR] Error reading line " << ll << " of " << fname << endl;
		  		std::cerr << "	  Index not found in data files" << endl;
		  		assert(0);
			}
			idx = iter->second;
			for(int k=0; k<M.get_size2(); k++) {
				// Read a value
				if(k<M.get_size2()-1) {
					fscanf(fin,"%lf\t",&vv);
				} else {
					fscanf(fin,"%lf\n",&vv);
				}
				M.get_object(idx,k).initialize(vv,pp);
			}

		}
	  	fclose(fin);
	}

	static void initialize_from_file(string fname, const std::map<unsigned long long, int> &ids, Matrix2D<var_gaussian> &M, const my_param &param, bool pp) {
		string fname1 = fname + "_mean.tsv";
		string fname2 = fname + "_std.tsv";
		FILE *fin1 = fopen(fname1.c_str(),"r");
	  	if(!fin1) {
	  		std::cerr << "[ERR] Unable to open " << fname1 << endl;
	  		assert(0);
	  	}
		FILE *fin2 = fopen(fname2.c_str(),"r");
	  	if(!fin2) {
	  		std::cerr << "[ERR] Unable to open " << fname2 << endl;
	  		assert(0);
	  	}
	  	unsigned long long id;
	  	int ll;
	  	int idx;
	  	double mm;
	  	double ss;
		while(!feof(fin1)){
			// Read a line
			fscanf(fin1,"%d\t%llu\t",&ll,&id);
			fscanf(fin2,"%d\t%llu\t",&ll,&id);
			std::map<unsigned long long,int>::const_iterator iter = ids.find(id);
			if(iter==ids.end()) {
		  		std::cerr << "[ERR] Error reading line " << ll << " of " << fname2 << endl;
		  		std::cerr << "	  Index not found in data files" << endl;
		  		assert(0);
			}
			idx = iter->second;
			for(int k=0; k<M.get_size2(); k++) {
				// Read a value
				if(k<M.get_size2()-1) {
					fscanf(fin1,"%lf\t",&mm);
					fscanf(fin2,"%lf\t",&ss);
				} else {
					fscanf(fin1,"%lf\n",&mm);
					fscanf(fin2,"%lf\n",&ss);
				}
				M.get_object(idx,k).initialize(mm,ss);
			}

		}
	  	fclose(fin1);
	  	fclose(fin2);
	}

	static void initialize_from_file(string fname, const std::map<unsigned long long, int> &ids, Matrix2D<var_gamma> &M, const my_param &param, bool pp) {
		string fname1 = fname + "_shp.tsv";
		string fname2 = fname + "_rte.tsv";
		FILE *fin1 = fopen(fname1.c_str(),"r");
	  	if(!fin1) {
	  		std::cerr << "[ERR] Unable to open " << fname1 << endl;
	  		assert(0);
	  	}
		FILE *fin2 = fopen(fname2.c_str(),"r");
	  	if(!fin2) {
	  		std::cerr << "[ERR] Unable to open " << fname2 << endl;
	  		assert(0);
	  	}
	  	unsigned long long id;
	  	int ll;
	  	int idx;
	  	double ss;
	  	double rr;
		while(!feof(fin1)){
			// Read a line
			fscanf(fin1,"%d\t%llu\t",&ll,&id);
			fscanf(fin2,"%d\t%llu\t",&ll,&id);
			std::map<unsigned long long,int>::const_iterator iter = ids.find(id);
			if(iter==ids.end()) {
		  		std::cerr << "[ERR] Error reading line " << ll << " of " << fname2 << endl;
		  		std::cerr << "	  Index not found in data files" << endl;
		  		assert(0);
			}
			idx = iter->second;
			for(int k=0; k<M.get_size2(); k++) {
				// Read a value
				if(k<M.get_size2()-1) {
					fscanf(fin1,"%lf\t",&ss);
					fscanf(fin2,"%lf\t",&rr);
				} else {
					fscanf(fin1,"%lf\n",&ss);
					fscanf(fin2,"%lf\n",&rr);
				}
				M.get_object(idx,k).initialize(ss,ss/rr);
			}

		}
	  	fclose(fin1);
	  	fclose(fin2);
	}

	/*
	static void initialize_from_group_file(string fname, const std::map<unsigned long long, int> &ids, \
										   const std::map<unsigned long long, int> &group_ids, Matrix1D<int> &group_per_item, \
										   Matrix2D<var_pointmass> &M, const my_param &param, \
										   double valIni, bool pp, gsl_rng *semilla) {
		FILE *fin = fopen(fname.c_str(),"r");
	  	if(!fin) {
	  		std::cerr << "[ERR] Unable to open " << fname << endl;
	  		assert(0);
	  	}
	  	// Read the file
	  	unsigned long long id;
	  	int ll;
	  	int idx;
	  	double vv;
	  	Matrix2D<double> Maux = Matrix2D<double>(group_ids.size(), param.Kgroup);
		while(!feof(fin)){
			// Read a line
			fscanf(fin,"%d\t%llu\t",&ll,&id);
			std::map<unsigned long long,int>::const_iterator iter = group_ids.find(id);
			if(iter==group_ids.end()) {
		  		std::cerr << "[ERR] Error reading line " << ll << " of " << fname << endl;
		  		std::cerr << "	  Index not found in data files" << endl;
		  		assert(0);
			}
			idx = iter->second;
			for(int k=0; k<Maux.get_size2(); k++) {
				// Read a value
				if(k<Maux.get_size2()-1) {
					fscanf(fin,"%lf\t",&vv);
				} else {
					fscanf(fin,"%lf\n",&vv);
				}
				Maux.set_object(idx,k,vv);
			}
		}
	  	fclose(fin);
	  	// Copy to matrix up to Kgroup; initialize randomly the rest
	  	int g_i;
	  	for(int i=0; i<static_cast<int>(ids.size()); i++) {
	  		g_i = group_per_item.get_object(i);
	  		// Copy from matrix and set if it's trainable or not
	  		for(int k=0; k<param.Kgroup; k++) {
	  			vv = Maux.get_object(g_i,k);
	  			M.get_object(i,k).initialize(vv,pp);
	  			M.get_object(i,k).set_trainable(!param.flag_fixKgroup);
			}
	  		// Initialize randomly
	  		for(int k=param.Kgroup; k<param.K; k++) {
	  			M.get_object(i,k).initialize_random(semilla,valIni,pp,0.01);
  			}
	  	}
	}
	*/

};

#endif
