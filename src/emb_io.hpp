#ifndef PEMB_IO_HPP
#define PEMB_IO_HPP

class my_input {
public:

	static void read_input_from_command_line(int argc, char **argv, my_data &data, my_param &param, my_hyper &hyper) {
		int i = 0;
		string val;
		while(i<=argc-1) {
			if(strcmp(argv[i], "-datadir") == 0) {
				val = string(argv[++i]);
				val = remove_final_slash(val);
				param.datadir = val;
			} else if(strcmp(argv[i], "-outdir") == 0) {
				val = string(argv[++i]);
				val = remove_final_slash(val);
				param.outdir = val;
			} else if(strcmp(argv[i], "-K") == 0) {
				val = string(argv[++i]);
				param.K = std::stoi(val);
			} else if(strcmp(argv[i], "-seed") == 0) {
				val = string(argv[++i]);
				param.seed = std::stoi(val);
			} else if(strcmp(argv[i], "-rfreq") == 0) {
				val = string(argv[++i]);
				param.rfreq = std::stoi(val);
			} else if(strcmp(argv[i], "-saveCycle") == 0) {
				val = string(argv[++i]);
				param.saveCycle = std::stoi(val);
			} else if(strcmp(argv[i], "-max-iterations") == 0) {
				val = string(argv[++i]);
				param.Niter = std::stoi(val);
			} else if(strcmp(argv[i], "-label") == 0) {
				val = string(argv[++i]);
				param.label = val;
			} else if(strcmp(argv[i], "-nsFreq") == 0) {
				val = string(argv[++i]);
				param.flag_nsFreq = std::stoi(val);
			} else if(strcmp(argv[i], "-likelihood") == 0) {
				val = string(argv[++i]);
				param.flag_likelihood = std::stoi(val);
			} else if(strcmp(argv[i], "-shuffle") == 0) {
				val = string(argv[++i]);
				param.flag_shuffle = std::stoi(val);
			} else if(strcmp(argv[i], "-negsamples") == 0) {
				val = string(argv[++i]);
				param.negsamples = std::stoi(val);
			} else if(strcmp(argv[i], "-zeroFactor") == 0) {
				val = string(argv[++i]);
				param.zeroFactor = std::stod(val);
			} else if(strcmp(argv[i], "-batchsize") == 0) {
				val = string(argv[++i]);
				param.batchsize = std::stoi(val);
			} else if(strcmp(argv[i], "-userVec") == 0) {
				val = string(argv[++i]);
				param.flag_userVec = std::stoi(val);
			} else if(strcmp(argv[i], "-days") == 0) {
				val = string(argv[++i]);
				param.flag_day = std::stoi(val);
			} else if(strcmp(argv[i], "-itemIntercept") == 0) {
				val = string(argv[++i]);
				if(std::stoi(val)==0) {
					param.flag_itemIntercept = false;
				} else {
					param.flag_itemIntercept = true;
				}
			} else if(strcmp(argv[i], "-price") == 0) {
				val = string(argv[++i]);
				param.flag_price = std::stoi(val);
			} else if(strcmp(argv[i], "-noAvgContext") == 0) {
				param.flag_avgContext = false;
			} else if(strcmp(argv[i], "-noCheckout") == 0) {
				param.flag_checkout = false;
			} else if(strcmp(argv[i], "-lookahead") == 0) {
				param.flag_lookahead = 1;
			} else if(strcmp(argv[i], "-symmetricRho") == 0) {
				param.flag_symmetricRho = true;
			} else if(strcmp(argv[i], "-normPrice") == 0) {
				val = string(argv[++i]);
				if(std::stoi(val)==0) {
					param.flag_normPrice = false;
					param.flag_normPrice_min = false;
				} else if(std::stoi(val)==1) {
					param.flag_normPrice = true;
					param.flag_normPrice_min = false;
				} else if(std::stoi(val)==2) {
					param.flag_normPrice = false;
					param.flag_normPrice_min = true;
				}
			} else if(strcmp(argv[i], "-step_schedule") == 0) {
				val = string(argv[++i]);
				param.flag_step_schedule = std::stoi(val);
			} else if(strcmp(argv[i], "-noVal") == 0) {
				param.noVal = true;
			} else if(strcmp(argv[i], "-noTest") == 0) {
				param.noTest = true;
			} else if(strcmp(argv[i], "-eta") == 0) {
				val = string(argv[++i]);
				param.eta = std::stod(val);
			} else if(strcmp(argv[i], "-keepOnly") == 0) {
				val = string(argv[++i]);
				param.lf_keepOnly = std::stoi(val);
			} else if(strcmp(argv[i], "-keepAbove") == 0) {
				val = string(argv[++i]);
				param.lf_keepAbove = std::stoi(val);
			} else if(strcmp(argv[i], "-threads") == 0) {
				val = string(argv[++i]);
				param.Nthreads = std::stoi(val);
			} else if(strcmp(argv[i], "-s2rho") == 0) {
				val = string(argv[++i]);
				hyper.s2rho = std::stod(val);
			} else if(strcmp(argv[i], "-s2alpha") == 0) {
				val = string(argv[++i]);
				hyper.s2alpha = std::stod(val);
			} else if(strcmp(argv[i], "-s2theta") == 0) {
				val = string(argv[++i]);
				hyper.s2theta = std::stod(val);
			} else if(strcmp(argv[i], "-s2lambda") == 0) {
				val = string(argv[++i]);
				hyper.s2lambda = std::stod(val);
			} else if(strcmp(argv[i], "-s2mu") == 0) {
				val = string(argv[++i]);
				hyper.s2mu = std::stod(val);
			} else if(strcmp(argv[i], "-s2delta") == 0) {
				val = string(argv[++i]);
				hyper.s2delta = std::stod(val);
			} else if(strcmp(argv[i], "-rtegamma") == 0) {
				val = string(argv[++i]);
				hyper.rte_gamma = std::stod(val);
			} else if(strcmp(argv[i], "-rtebeta") == 0) {
				val = string(argv[++i]);
				hyper.rte_beta = std::stod(val);
			} else if(strcmp(argv[i], "-shpgamma") == 0) {
				val = string(argv[++i]);
				hyper.shp_gamma = std::stod(val);
			} else if(strcmp(argv[i], "-shpbeta") == 0) {
				val = string(argv[++i]);
				hyper.shp_beta = std::stod(val);
			} else if(i>0) {
				fprintf(stdout,"[ERR] Unknown option %s\n", argv[i]);
				assert(0);
			} 
			i++;
		};
		// Check for incompatibilities
		if(param.flag_shuffle<0) {
			std::cerr << "[ERR] The input parameter shuffle must be >=0" << endl;
			assert(0);
		}
		if(param.flag_checkout && param.flag_shuffle==-1) {
			std::cerr << "[ERR] Checkout item cannot appear if shuffle==-1" << endl;
			assert(0);
		}
		if(param.flag_normPrice && param.flag_normPrice_min) {
			std::cerr << "[ERR] Choose either -normPrice or -normPriceMin, but not both" << endl;
			assert(0);
		}
		if(param.flag_nsFreq!=-1 && param.flag_nsFreq!=0 && param.flag_nsFreq!=1 && param.flag_nsFreq<2) {
			std::cerr << "[ERR] Wrong argument for -nsFreq: " << std::to_string(param.flag_nsFreq) << endl;
			assert(0);
		}
		if(param.lf_keepAbove>0 && param.lf_keepOnly>0) {
			std::cerr << "[ERR] Choose only one low-frequency method" << endl;
			assert(0);
		}
		if(param.iniFromGroup!="" && param.Kgroup<=0) {
			std::cerr << "[ERR] Kgroup must be a positive integer to initialize from group-level results" << endl;
			assert(0);
		}
		// Set stdIni
		if(param.stdIni<=0.0) {
			param.stdIni = 1.0/sqrt(param.K);
		}
		// Set flag_lf
		if(param.lf_keepAbove>0) {
			param.lf_flag = 1;
		} else if(param.lf_keepOnly>0) {
			param.lf_flag = 2;
		}
	}

	static string remove_final_slash(const string f) {
		string aux;
		if(f.back()=='/') {
			aux = f.substr(0,f.size()-1);
		} else {
			aux = f;
		}
		return aux;
	}

	static void read_data_file(my_data &data, const my_param &param) {
		string trainFname = param.datadir+"/train.tsv";
		string testFname = param.datadir+"/test.tsv";
		string validationFname = param.datadir+"/validation.tsv";
		unsigned int NiTr;
		unsigned int NuTr;

		// First, read train.tsv to count how many times each item appears and remove low-freq items
		pre_read(trainFname,data,param);

		// This is a 2-step process
		// Step 1: Count #users, #items (and #sessions)
		std::cout << " +Stage 1/2..." << endl;
		int n_u = 0;	// Auxiliary index to last user that was read
		int n_i = 0;	// Auxiliary index to last item that was read
		int n_s = 0;	// Auxiliary index to last session that was read
		data.obs.T = count_input_tsv_file(trainFname,data,param,n_u,n_i,n_s,param.flag_checkout);
		NuTr = data.user_ids.size();
		NiTr = data.item_ids.size();
		if(!param.noTest) {
			data.obs_test.T = count_input_tsv_file(testFname,data,param,n_u,n_i,n_s);
		}
		if(!param.noVal) {
			data.obs_val.T = count_input_tsv_file(validationFname,data,param,n_u,n_i,n_s);
		}
		if(data.user_ids.size()>NuTr) {
			std::cerr << "[WARN] There are users in test/validation that do not appear in train" << endl;
		}
		if(data.item_ids.size()>NiTr) {
			std::cerr << "[WARN] There are items in test/validation that do not appear in train" << endl;
		}
		data.Nusers = data.user_ids.size();
		data.Nitems = data.item_ids.size();
		data.Nsessions = data.session_ids.size();

		// Allocate memory to store data
		data.obs = my_data_aux(data.obs.T);
		data.obs_test = my_data_aux(data.obs_test.T);
		data.obs_val = my_data_aux(data.obs_val.T);

		// Step 2: Read the actual data
		std::cout << " +Stage 2/2..." << endl;
		read_input_tsv_file(trainFname,data,data.obs,param);
		if(!param.noTest) {
			read_input_tsv_file(testFname,data,data.obs_test,param);
		}
		if(!param.noVal) {
			read_input_tsv_file(validationFname,data,data.obs_val,param);
		}
	}

	static void read_a_line(const my_param &param, FILE *fin, unsigned long long *uid, unsigned long long *mid, unsigned long long *sid, unsigned int *rating) {
		fscanf(fin,"%llu\t%llu\t%llu\t%u\n",uid,mid,sid,rating);
	}

	static void pre_read(string fname, my_data &data, const my_param &param) {
		if(param.lf_flag<=0) {
			return;
		}

		unsigned long long uid;
		unsigned long long mid;
		unsigned long long sid;
		unsigned int rating;

		std::map<unsigned long long, int> item_counts;
		std::vector<int> all_counts;
		int min_freq;

		// Read the training data
	  	FILE *fin = fopen(fname.c_str(),"r");
	  	if(!fin) {
	  		std::cerr << "[ERR] Unable to open " << fname << endl;
	  		assert(0);
	  	}
		while(!feof(fin)){
			// Read a line
			read_a_line(param,fin,&uid,&mid,&sid,&rating);
			// Append item id to the list of valid items
			auto iter_i = item_counts.find(mid);
			if(iter_i==item_counts.end()) {
				item_counts.insert(std::pair<unsigned long long,int>(mid,1));
			} else {
				iter_i->second = iter_i->second+1;
			}
		}
		fclose(fin);

		// Order by frequency
		for(auto iter_i=item_counts.begin(); iter_i!=item_counts.end(); ++iter_i) {
			all_counts.push_back(iter_i->second);
		}
		std::sort(all_counts.begin(),all_counts.end(),std::greater<int>());
		if(param.lf_keepAbove>0) {
			min_freq = param.lf_keepAbove;
		} else if(param.lf_keepOnly>0) {
			min_freq = all_counts.at(param.lf_keepOnly-1);
		} else {
			std::cerr << "[ERR] This should not happen. Review function 'pre_read'" << endl;
			assert(0);
		}

		// Insert elements to valid_items
		for(auto iter_i=item_counts.begin(); iter_i!=item_counts.end(); ++iter_i) {
			if(iter_i->second >= min_freq) {
				data.valid_items.push_back(iter_i->first);
			}
		}
	}

	static unsigned int count_input_tsv_file(string fname, my_data &data, const my_param &param, int &n_u, int &n_i, int &n_s, bool checkout) {
		unsigned int nlines = count_input_tsv_file(fname,data,param,n_u,n_i,n_s);
		if(checkout) {
			unsigned long long mid = ULLONG_MAX;
			data.item_ids.insert(std::pair<unsigned long long,int>(mid,n_i));
			n_i++;
		}
		return nlines;
	}

	static unsigned int count_input_tsv_file(string fname, my_data &data, const my_param &param, int &n_u, int &n_i, int &n_s) {
		unsigned long long uid;
		unsigned long long mid;
		unsigned long long sid;
		unsigned int rating;
		unsigned int nlines = 0;		// number of lines read

	  	FILE *fin = fopen(fname.c_str(),"r");
	  	if(!fin) {
	  		std::cerr << "[ERR] Unable to open " << fname << endl;
	  		assert(0);
	  	}
		while(!feof(fin)){
			// Read a line
			read_a_line(param,fin,&uid,&mid,&sid,&rating);
			auto iter_i = std::find(data.valid_items.begin(),data.valid_items.end(),mid);
			if(param.lf_flag<=0 || iter_i!=data.valid_items.end()) {
				nlines++;
				// Append user id to the list of user id's
				if(data.user_ids.find(uid) == data.user_ids.end()) {
					data.user_ids.insert(std::pair<unsigned long long,int>(uid,n_u));
					n_u++;
				}
				// Append item id to the list of item id's
				if(data.item_ids.find(mid) == data.item_ids.end()) {
					data.item_ids.insert(std::pair<unsigned long long,int>(mid,n_i));
					n_i++;
				}
				// Append session id to the list of session id's
				if(data.session_ids.find(sid) == data.session_ids.end()) {
					data.session_ids.insert(std::pair<unsigned long long,int>(sid,n_s));
					n_s++;
				}
			}
		}
		fclose(fin);
		return(nlines);
	}

	static void read_input_tsv_file(string fname, const my_data &data, my_data_aux &data_aux, const my_param &param) {
		unsigned long long uid;
		unsigned long long mid;
		unsigned long long sid;
		unsigned int rating;
		unsigned int count = 0;

	  	FILE *fin = fopen(fname.c_str(),"r");
	  	if(!fin) {
	  		std::cerr << "[ERR] Unable to open " << fname << endl;
	  		assert(0);
	  	}

		while(!feof(fin)){
			// Read a line
			read_a_line(param,fin,&uid,&mid,&sid,&rating);
			auto iter_i = std::find(data.valid_items.begin(),data.valid_items.end(),mid);
			if(param.lf_flag<=0 || iter_i!=data.valid_items.end()) {
				// Store values
				data_aux.y_user[count] = data.user_ids.find(uid)->second;
				data_aux.y_item[count] = data.item_ids.find(mid)->second;
				data_aux.y_rating[count] = (rating>0)?1:0;
				data_aux.y_sess[count] = data.session_ids.find(sid)->second;
				// Increase count
				count++;
			}
		}
		fclose(fin);
	}

	static void read_price_file(my_data &data, const my_param &param) {
		// Open file
		string fname = param.datadir+"/item_sess_price.tsv";
		FILE *fin = fopen(fname.c_str(),"r");
	  	if(!fin) {
	  		std::cerr << "[ERR] Unable to open " << fname << endl;
	  		assert(0);
	  	}
	  	// Allocate data structure
		data.price_is = Matrix2D<double>(data.Nitems,data.Nsessions);
		for(int i=0; i<data.Nitems; i++) {
			for(int s=0; s<data.Nsessions; s++) {
				data.price_is.set_object(i,s,1.0);
			}
		}
		// Read file
		int nlines = 0;
		int ll = 0;
		unsigned long long mid;
		unsigned long long sid;
		double p;
		std::map<unsigned long long,int>::const_iterator i_item;
		std::map<unsigned long long,int>::const_iterator i_session;
		int idx_item;
		int idx_session;
		while(!feof(fin)){
			ll++;
			fscanf(fin,"%llu\t%llu\t%lf\n",&mid,&sid,&p);
			// Find the item and session
			i_item = data.item_ids.find(mid);
			i_session = data.session_ids.find(sid);
			// Check if the ids are found in the data files
			if(i_item==data.item_ids.end() || i_session==data.session_ids.end()) {
				std::cerr << "[WARN] Line " << ll << " of item_sess_price.tsv contains a session or item id not found in the data files" << endl;
			} else {
				nlines++;
				idx_item = i_item->second;
				idx_session = i_session->second;
				data.price_is.set_object(idx_item,idx_session,p);
			}
		}
		// Close file
		fclose(fin);
		// Check number of valid lines
		if( (!param.flag_checkout && nlines<data.Nitems*data.Nsessions)
			|| (param.flag_checkout && nlines<(data.Nitems-1)*data.Nsessions) ) {
			std::cerr << "[ERR] The number of lines in 'item_sess_price.tsv'";
			std::cerr << "	  is not consistent with the data files" << endl;
			assert(0);
		}
	}	
	
	static void normalize_prices(my_data &data, const my_param &param) {
		data.price_avg_i = Matrix1D<double>(data.Nitems);
		data.price_min_i = Matrix1D<double>(data.Nitems);
		double aux;

		for(int i=0; i<data.Nitems; i++) {
			// Add up all prices of item i
			aux = 0.0;
			for(int s=0; s<data.Nsessions; s++) {
				aux += data.price_is.get_object(i,s)/static_cast<double>(data.Nsessions);
			}
			data.price_avg_i.set_object(i,aux);
			// Find minimum
			aux = myINFINITY;
			for(int s=0; s<data.Nsessions; s++) {
				aux = my_min(aux,data.price_is.get_object(i,s));
			}
			data.price_min_i.set_object(i,aux);
			// Normalize prices
			if(param.flag_normPrice || param.flag_normPrice_min) {
				for(int s=0; s<data.Nsessions; s++) {
					if(param.flag_normPrice) {
						aux = data.price_is.get_object(i,s)/data.price_avg_i.get_object(i);
					} else if(param.flag_normPrice_min) {
						aux = data.price_is.get_object(i,s)/data.price_min_i.get_object(i);
					}
					data.price_is.set_object(i,s,aux);
				}
			}
		}
	}
	
	static void take_log_prices(my_data &data, const my_param &param) {
		double aux;
		for(int i=0; i<data.Nitems; i++) {
			for(int s=0; s<data.Nsessions; s++) {
				aux = data.price_is.get_object(i,s);
				data.price_is.set_object(i,s,my_log(aux));
			}
		}
	}

	static void read_itemgroups(my_data &data, const my_param &param) {
		unsigned long long iid;
		unsigned long long gid;
		int i;
		int g;
		int n_g = 0;

		data.group_per_item = Matrix1D<int>(data.Nitems);
		string fname = param.datadir+"/itemGroup.tsv";
		// Read the file
		FILE *fin = fopen(fname.c_str(),"r");
	  	if(!fin) {
	  		std::cerr << "[ERR] Unable to open " << fname << endl;
	  		assert(0);
	  	}
		while(!feof(fin)){
			fscanf(fin, "%llu\t%llu\n", &iid, &gid);
			std::map<unsigned long long,int>::iterator iter_i = data.item_ids.find(iid);
			if(iter_i==data.item_ids.end()) {
				i = data.Nitems;
				std::cerr << "[WARN] Item " << iid << " in 'itemGroup.tsv' not found in train/test/validation" << endl;
			} else {
				i = iter_i->second;
			}
			if(i<data.Nitems) {
				std::map<unsigned long long,int>::const_iterator iter_g = data.itemgroup_ids.find(gid);
				if(iter_g==data.itemgroup_ids.end()) {
					data.itemgroup_ids.insert(std::pair<unsigned long long,int>(gid,n_g));
					n_g++;
					iter_g = data.itemgroup_ids.find(gid);
				}
				g = iter_g->second;
				data.group_per_item.set_object(i,g);
			}
		}
		fclose(fin);
		// If checkout, insert "checkout item" in its own "itemgroup"
		if(param.flag_checkout) {
			iid = ULLONG_MAX;
			gid = ULLONG_MAX;
			i = data.item_ids.find(iid)->second;
			data.itemgroup_ids.insert(std::pair<unsigned long long,int>(gid,n_g));
			n_g++;
			std::map<unsigned long long,int>::const_iterator iter_g = data.itemgroup_ids.find(gid);
			g = iter_g->second;
			data.group_per_item.set_object(i,g);
		}
		// Other data variables
		data.NitemGroups = data.itemgroup_ids.size();
		data.items_per_group = Matrix1D<std::vector<int>>(data.NitemGroups);
		for(int ii=0; ii<data.Nitems; ii++) {
			data.items_per_group.get_object(data.group_per_item.get_object(ii)).push_back(ii);
		}
	}

	static void read_usergroups(my_data &data, const my_param &param) {
		unsigned long long uid;
		unsigned long long gid;
		int u;
		int g;
		int n_g = 0;

		data.group_per_user = Matrix1D<int>(data.Nusers);
		string fname = param.datadir+"/userGroup.tsv";
		// Read the file
		FILE *fin = fopen(fname.c_str(),"r");
	  	if(!fin) {
	  		std::cerr << "[ERR] Unable to open " << fname << endl;
	  		assert(0);
	  	}
		while(!feof(fin)){
			fscanf(fin, "%llu\t%llu\n", &uid, &gid);
			std::map<unsigned long long,int>::const_iterator iter_u = data.user_ids.find(uid);
			if(iter_u==data.user_ids.end()) {
				u = data.Nusers;
				std::cerr << "[WARN] User " << uid << " in 'userGroup.tsv' not found in train/test/validation" << endl;
			} else {
				u = iter_u->second;
			}
			if(u<data.Nusers) {
				std::map<unsigned long long,int>::iterator iter_g = data.usergroup_ids.find(gid);
				if(iter_g==data.usergroup_ids.end()) {
					data.usergroup_ids.insert(std::pair<unsigned long long,int>(gid,n_g));
					n_g++;
					iter_g = data.usergroup_ids.find(gid);
				}
				g = iter_g->second;
				data.group_per_user.set_object(u,g);
			}
		}
		fclose(fin);
		// Other data variables
		data.NuserGroups = data.usergroup_ids.size();
		data.users_per_group = Matrix1D<std::vector<int>>(data.NuserGroups);
		for(int uu=0; uu<data.Nusers; uu++) {
			data.users_per_group.get_object(data.group_per_user.get_object(uu)).push_back(uu);
		}
	}

	static void read_sess_days(my_data &data, const my_param &param) {
		unsigned long long did;
		unsigned long long wid;
		unsigned long long sid;
		int s;
		int d;
		int w;
		double hh;
		int n_d = 0;
		int n_w = 0;

		string fname = param.datadir+"/sess_days.tsv";
		// Read the file & create day_per_session
		data.day_per_session = Matrix1D<int>(data.Nsessions);
		data.weekday_per_session = Matrix1D<int>(data.Nsessions);
		data.hour_per_session = Matrix1D<double>(data.Nsessions);
		FILE *fin = fopen(fname.c_str(),"r");
	  	if(!fin) {
	  		std::cerr << "[ERR] Unable to open " << fname << endl;
	  		assert(0);
	  	}
		while(!feof(fin)){
			fscanf(fin, "%llu\t%llu\t%llu\t%lf\n", &sid, &did, &wid, &hh);
			std::map<unsigned long long,int>::iterator iter_s = data.session_ids.find(sid);
			if(iter_s==data.session_ids.end()) {
				s = data.Nsessions;
				std::cerr << "[WARN] Session " << sid << " in 'sess_days.tsv' not found in train/test/validation" << endl;
			} else {
				s = iter_s->second;
			}
			if(s<data.Nsessions) {
				std::map<unsigned long long,int>::const_iterator iter_d = data.day_ids.find(did);
				if(iter_d==data.day_ids.end()) {
					data.day_ids.insert(std::pair<unsigned long long,int>(did,n_d));
					n_d++;
					iter_d = data.day_ids.find(did);
				}
				d = iter_d->second;
				data.day_per_session.set_object(s,d);

				std::map<unsigned long long,int>::const_iterator iter_w = data.weekday_ids.find(wid);
				if(iter_w==data.weekday_ids.end()) {
					data.weekday_ids.insert(std::pair<unsigned long long,int>(wid,n_w));
					n_w++;
					iter_w = data.weekday_ids.find(wid);
				}
				w = iter_w->second;
				data.weekday_per_session.set_object(s,w);

				data.hour_per_session.set_object(s,hh);
			}
		}
		fclose(fin);

		// Count Ndays
		data.Ndays = data.day_ids.size();
		data.Nweekdays = data.weekday_ids.size();

		// Create sessions_per_day
		data.sessions_per_day = Matrix1D<std::vector<int>>(data.Ndays);
		for(int ss=0; ss<data.Nsessions; ss++) {
			d = data.day_per_session.get_object(ss);
			data.sessions_per_day.get_object(d).push_back(ss);
		}

		// Create sessions_per_weekday
		data.sessions_per_weekday = Matrix1D<std::vector<int>>(data.Nweekdays);
		for(int ss=0; ss<data.Nsessions; ss++) {
			w = data.weekday_per_session.get_object(ss);
			data.sessions_per_weekday.get_object(w).push_back(ss);
		}
	}

};

class my_output {
public:

	static void create_output_folder(const my_data &data, my_param &param) {
		ostringstream sa;
		sa << "t" << data.Ntrans << "-";
		sa << "n" << data.Nusers << "-";
		sa << "m" << data.Nitems << "-";
		sa << "k" << param.K;
		if(param.flag_itemIntercept) {
			sa << "-intercept";
		}
		if(param.flag_userVec>0) {
			sa << "-users" << param.flag_userVec;
		}
		if(param.flag_price>0) {
			sa << "-price" << param.flag_price;
			if(param.flag_normPrice) {
				sa << "norm";
			}
		}
		if(param.flag_day>0) {
			sa << "-days" << param.flag_day;
		}
		sa << "-lik" << param.flag_likelihood;
		if(param.flag_lookahead!=0) {
			sa << "-lookahead" << param.flag_lookahead;
		}
		if(param.flag_symmetricRho) {
			sa << "-symmetricRho";
		}
		sa << "-avgCtxt" << param.flag_avgContext;
		sa << "-shuffle" << param.flag_shuffle;
		sa << "-eta" << param.eta;
		if(param.zeroFactor>=0) {
			sa << "-zF" << param.zeroFactor;
		}
		if(param.negsamples>=0) {
			sa << "-nS" << param.negsamples;
		}
		if(param.flag_nsFreq!=-1) {
			sa << "-nsFreq" << param.flag_nsFreq;
		}
		if(param.batchsize>0) {
			sa << "-batch" << param.batchsize;
		}
		if(param.flag_checkout) {
			sa << "-chkout";
		}
		if(param.label!="") {
			sa << "-" << param.label;
		}

		string prefix = sa.str();
		struct stat buffer;
		std::cout << "Output directory label: " << prefix << endl;
		int out = stat((param.outdir+"/"+prefix).c_str(), &buffer);
		if(out != 0) {
			std::cout << "Creating directory " << param.outdir << "/" << prefix << endl;
			if(mkdir((param.outdir+"/"+prefix).c_str(), S_IRUSR | S_IWUSR | S_IXUSR) != 0) {
				std::cerr << "[ERR] Cannot create output folder" << endl;
				assert(0);
			}
		}

		// Keep the new outdir
		param.outdir = param.outdir+"/"+prefix;
	}

	static void create_log_file(const my_data &data, const my_param &param, const my_hyper &hyper) {
		// Write data
		write_log(param.outdir,"Data:");
		write_log(param.outdir," +datadir="+param.datadir);
		write_log(param.outdir," +Nusers="+std::to_string(data.Nusers));
		write_log(param.outdir," +Nitems="+std::to_string(data.Nitems));
		write_log(param.outdir," +Nsessions="+std::to_string(data.Nsessions));
		write_log(param.outdir," +Ntrans="+std::to_string(data.Ntrans));
		write_log(param.outdir," +Ntrans (test)="+std::to_string(data.test_Ntrans));
		write_log(param.outdir," +Ndays="+std::to_string(data.Ndays));
		write_log(param.outdir," +Nweekdays="+std::to_string(data.Nweekdays));
		write_log(param.outdir," +NuserGroups="+std::to_string(data.NuserGroups));
		write_log(param.outdir," +NitemGroups="+std::to_string(data.NitemGroups));
		write_log(param.outdir," +Lines of train.tsv="+std::to_string(data.obs.T));
		write_log(param.outdir," +Lines of test.tsv="+std::to_string(data.obs_test.T));
		write_log(param.outdir," +Lines of validation.tsv="+std::to_string(data.obs_val.T));

		// Write parameters
		write_log(param.outdir,"Parameters:");
		write_log(param.outdir," +outdir="+param.outdir);
		write_log(param.outdir," +K="+std::to_string(param.K));
		write_log(param.outdir," +Kgroup="+std::to_string(param.Kgroup));
		write_log(param.outdir," +fixKgroup="+std::to_string(param.flag_fixKgroup));
		write_log(param.outdir," +seed="+std::to_string(param.seed));
		write_log(param.outdir," +rfreq="+std::to_string(param.rfreq));
		write_log(param.outdir," +saveCycle="+std::to_string(param.saveCycle));
		write_log(param.outdir," +max-iterations="+std::to_string(param.Niter));
		write_log(param.outdir," +negsamples="+std::to_string(param.negsamples));
		write_log(param.outdir," +nsFreq="+std::to_string(param.flag_nsFreq));
		write_log(param.outdir," +likelihood="+std::to_string(param.flag_likelihood));
		write_log(param.outdir," +lookahead="+std::to_string(param.flag_lookahead));
		write_log(param.outdir," +avgContext="+std::to_string(param.flag_avgContext));
		write_log(param.outdir," +symmetricRho="+std::to_string(param.flag_symmetricRho));
		write_log(param.outdir," +checkout="+std::to_string(param.flag_checkout));
		write_log(param.outdir," +shuffle="+std::to_string(param.flag_shuffle));
		write_log(param.outdir," +zeroFactor="+std::to_string(param.zeroFactor));
		write_log(param.outdir," +batchsize="+std::to_string(param.batchsize));
		write_log(param.outdir," +userVec="+std::to_string(param.flag_userVec));
		write_log(param.outdir," +itemIntercept="+std::to_string(param.flag_itemIntercept));
		write_log(param.outdir," +price="+std::to_string(param.flag_price));
		write_log(param.outdir," +day="+std::to_string(param.flag_day));
		write_log(param.outdir," +normPrice="+std::to_string(param.flag_normPrice));
		write_log(param.outdir," +normPriceMin="+std::to_string(param.flag_normPrice_min));
		write_log(param.outdir," +step_schedule="+std::to_string(param.flag_step_schedule));
		write_log(param.outdir," +eta="+std::to_string(param.eta));
		write_log(param.outdir," +gamma="+std::to_string(param.gamma));
		write_log(param.outdir," +valTolerance="+std::to_string(param.valTolerance));
		write_log(param.outdir," +valConsecutive="+std::to_string(param.valConsecutive));
		write_log(param.outdir," +keepOnly="+std::to_string(param.lf_keepOnly));
		write_log(param.outdir," +keepAbove="+std::to_string(param.lf_keepAbove));
		write_log(param.outdir," +threads="+std::to_string(param.Nthreads));

		// Write initialization
		write_log(param.outdir,"Initialization:");
		write_log(param.outdir," +stdIni="+std::to_string(param.stdIni));
		write_log(param.outdir," +iniPath="+param.iniPath);
		write_log(param.outdir," +iniFromGroup="+param.iniFromGroup);
		if(param.flag_iniThetaVal) {
			write_log(param.outdir," +iniThetaVal="+std::to_string(param.iniThetaVal));
		}
		if(param.flag_iniPriceVal) {
			write_log(param.outdir," +iniPriceVal="+std::to_string(param.iniPriceVal));
		}

		// Write hyperparameters
		write_log(param.outdir,"Hyperparameters:");
		write_log(param.outdir," +s2rho="+std::to_string(hyper.s2rho));
		write_log(param.outdir," +s2alpha="+std::to_string(hyper.s2alpha));
		write_log(param.outdir," +s2theta="+std::to_string(hyper.s2theta));
		write_log(param.outdir," +s2lambda="+std::to_string(hyper.s2lambda));
		write_log(param.outdir," +rtegamma="+std::to_string(hyper.rte_gamma));
		write_log(param.outdir," +shpgamma="+std::to_string(hyper.shp_gamma));
		write_log(param.outdir," +rtebeta="+std::to_string(hyper.rte_beta));
		write_log(param.outdir," +shpbeta="+std::to_string(hyper.shp_beta));
		write_log(param.outdir," +s2delta="+std::to_string(hyper.s2delta));
		write_log(param.outdir," +s2mu="+std::to_string(hyper.s2mu));
	}

	static void write_log(const string folder, const string str) {
		write_line(folder+"/log.txt",str);
	}

	static void write_line(const string file, const string str) {
		ofstream myfile;
		myfile.open(file,ios::out|ios::app);	// Append content to the end of existing file
		if(!myfile.is_open()) {
			std::cerr << "[ERR] File '" << file << "' could not be opened" << endl;
			assert(0);
		}
		myfile << str << endl;
		myfile.close();
	}

	static void write_normalized_prices(my_data &data, const my_param &param) {
		string filenameAvg = param.outdir+"/mean_prices.txt";
		string filenameMin = param.outdir+"/min_prices.txt";
		char buffer[200];
		unsigned long long iName;
		int i;
		string aux;

		int count = 0;
		for(auto const &it : data.item_ids) {
			iName = it.first;
			i = it.second;

			sprintf(buffer,"%d\t%llu\t%.12f",count,iName,data.price_avg_i.get_object(i));
			aux = string(buffer);
			write_line(filenameAvg,aux);

			sprintf(buffer,"%d\t%llu\t%.12f",count,iName,data.price_min_i.get_object(i));
			aux = string(buffer);
			write_line(filenameMin,aux);

			count++;
		}
	}

	static void write_telapsed(const string folder, int iter, time_t t_ini, time_t t_end) {
		double elapsed_secs = difftime(t_end,t_ini);
		write_line(folder+"/telapsed.txt",std::to_string(iter)+"\t"+std::to_string(elapsed_secs));
	}

	static void write_norm(const string outdir, int iter, double norm_rho, double norm_alpha) {
		write_line(outdir+"/avg_norm.txt",std::to_string(iter)+"\t"+std::to_string(norm_rho)+"\t"+std::to_string(norm_alpha));
	}

	static void write_norm(const string outdir, int iter, double norm_rho, double norm_alpha, double norm_theta) {
		write_line(outdir+"/avg_norm.txt",std::to_string(iter)+"\t"+std::to_string(norm_rho)+"\t"+std::to_string(norm_alpha)+"\t"+std::to_string(norm_theta));
	}

	static void write_max_file(const my_param &param, int duration, double val_llh, int why) {
		char buffer[200];
		sprintf(buffer,"%d\t%d\t%.9f\t%d",param.it,duration,val_llh,why);
		write_line(param.outdir+"/max.txt",string(buffer));
	}
	
	static void write_matrix(string filename, std::map<unsigned long long,int> &ids, Matrix1D<var_pointmass> &M) {
		char buffer[200];
		string aux;
		int i;
		unsigned long long iName;
		std::map<unsigned long long,int>::iterator it;

		// Print values
		int count = 0;
		for(auto const &it : ids) {
			iName = it.first;
			i = it.second;

			sprintf(buffer,"%d\t%llu\t%.12f",count,iName,M.get_object(i).e_x);
			count++;
			aux = string(buffer);
			write_line(filename+"_mean.tsv",aux);
		}
	}

	static void write_matrix(string filename, std::map<unsigned long long,int> &ids, Matrix1D<var_gaussian> &M) {
		char buffer[200];
		string aux;
		int i;
		unsigned long long iName;
		std::map<unsigned long long,int>::iterator it;

		// Print values
		int count = 0;
		for(auto const &it : ids) {
			iName = it.first;
			i = it.second;

			sprintf(buffer,"%d\t%llu\t%.12f",count,iName,M.get_object(i).mean);
			aux = string(buffer);
			write_line(filename+"_mean.tsv",aux);

			sprintf(buffer,"%d\t%llu\t%.12f",count,iName,M.get_object(i).sigma);
			aux = string(buffer);
			write_line(filename+"_std.tsv",aux);

			count++;
		}
	}

	static void write_matrix(string filename, std::map<unsigned long long,int> &ids, Matrix1D<var_gamma> &M) {
		char buffer[200];
		string aux;
		int i;
		unsigned long long iName;
		std::map<unsigned long long,int>::iterator it;

		// Print values
		int count = 0;
		for(auto const &it : ids) {
			iName = it.first;
			i = it.second;

			sprintf(buffer,"%d\t%llu\t%.12f",count,iName,M.get_object(i).shape);
			aux = string(buffer);
			write_line(filename+"_shp.tsv",aux);

			sprintf(buffer,"%d\t%llu\t%.12f",count,iName,M.get_object(i).rate);
			aux = string(buffer);
			write_line(filename+"_rte.tsv",aux);

			sprintf(buffer,"%d\t%llu\t%.12f",count,iName,M.get_object(i).mean);
			aux = string(buffer);
			write_line(filename+"_mean.tsv",aux);

			count++;
		}
	}

	static void write_matrix(string filename, std::map<unsigned long long,int> &ids, Matrix2D<var_pointmass> &M) {
		char buffer[100];
		string aux;
		int i;
		unsigned long long iName;
		std::map<unsigned long long,int>::iterator it;

		// Print values
		int count = 0;
		for(const auto &it : ids) {
			iName = it.first;
			i = it.second;

			sprintf(buffer,"%d\t%llu\t",count,iName);
			count++;
			aux = string(buffer);
			for(int k=0; k<M.get_size2()-1; k++) {
				sprintf(buffer,"%.12f\t",M.get_object(i,k).e_x);
				aux += string(buffer);
			}
			sprintf(buffer,"%.12f",M.get_object(i,M.get_size2()-1).e_x);
			aux += string(buffer);
			write_line(filename+"_mean.tsv",aux);
		}
	}

	static void write_matrix(string filename, std::map<unsigned long long,int> &ids, Matrix2D<var_gaussian> &M) {
		char buffer[200];
		string aux;
		int i;
		unsigned long long iName;
		std::map<unsigned long long,int>::iterator it;

		// Print values
		int count = 0;
		for(const auto &it : ids) {
			iName = it.first;
			i = it.second;

			sprintf(buffer,"%d\t%llu\t",count,iName);
			aux = string(buffer);
			for(int k=0; k<M.get_size2()-1; k++) {
				sprintf(buffer,"%.12f\t",M.get_object(i,k).mean);
				aux += string(buffer);
			}
			sprintf(buffer,"%.12f",M.get_object(i,M.get_size2()-1).mean);
			aux += string(buffer);
			write_line(filename+"_mean.tsv",aux);

			sprintf(buffer,"%d\t%llu\t",count,iName);
			aux = string(buffer);
			for(int k=0; k<M.get_size2()-1; k++) {
				sprintf(buffer,"%.12f\t",M.get_object(i,k).sigma);
				aux += string(buffer);
			}
			sprintf(buffer,"%.12f",M.get_object(i,M.get_size2()-1).sigma);
			aux += string(buffer);
			write_line(filename+"_std.tsv",aux);

			count++;
		}
	}

	static void write_matrix(string filename, std::map<unsigned long long,int> &ids, Matrix2D<var_gamma> &M) {
		char buffer[200];
		string aux;
		int i;
		unsigned long long iName;
		std::map<unsigned long long,int>::iterator it;

		// Print values
		int count = 0;
		for(const auto &it : ids) {
			iName = it.first;
			i = it.second;

			sprintf(buffer,"%d\t%llu\t",count,iName);
			aux = string(buffer);
			for(int k=0; k<M.get_size2()-1; k++) {
				sprintf(buffer,"%.12f\t",M.get_object(i,k).shape);
				aux += string(buffer);
			}
			sprintf(buffer,"%.12f",M.get_object(i,M.get_size2()-1).shape);
			aux += string(buffer);
			write_line(filename+"_shp.tsv",aux);

			sprintf(buffer,"%d\t%llu\t",count,iName);
			aux = string(buffer);
			for(int k=0; k<M.get_size2()-1; k++) {
				sprintf(buffer,"%.12f\t",M.get_object(i,k).rate);
				aux += string(buffer);
			}
			sprintf(buffer,"%.12f",M.get_object(i,M.get_size2()-1).rate);
			aux += string(buffer);
			write_line(filename+"_rte.tsv",aux);

			sprintf(buffer,"%d\t%llu\t",count,iName);
			aux = string(buffer);
			for(int k=0; k<M.get_size2()-1; k++) {
				sprintf(buffer,"%.12f\t",M.get_object(i,k).mean);
				aux += string(buffer);
			}
			sprintf(buffer,"%.12f",M.get_object(i,M.get_size2()-1).mean);
			aux += string(buffer);
			write_line(filename+"_mean.tsv",aux);

			count++;
		}
	}

	/*
	static void write_matrix_xday(string filename, const std::map<unsigned long long,int> &ids_g, const std::map<unsigned long long,int> &ids_i, const std::map<unsigned long long,int> &ids_d, Matrix3D<var_pointmass> &M) {
		char buffer[300];
		string aux;
		int g;
		int i;
		int d;

		std::map<unsigned long long,int>::const_iterator iter_g;
		std::map<unsigned long long,int>::const_iterator iter_i;
		std::map<unsigned long long,int>::const_iterator iter_d;

		// Print values
		int count = 0;
		for(iter_g=ids_g.begin(); iter_g!=ids_g.end(); ++iter_g) {
			g = iter_g->second;
			for(iter_i=ids_i.begin(); iter_i!=ids_i.end(); ++iter_i) {
				i = iter_i->second;
				for(iter_d=ids_d.begin(); iter_d!=ids_d.end(); ++iter_d) {
					d = iter_d->second;
					// Print mean
					sprintf(buffer,"%d\t%llu\t%llu\t%llu\t%.16f",count,iter_g->first,iter_i->first,iter_d->first,M.get_object(g,i,d).e_x);
					aux = string(buffer);
					write_line(filename,aux);
					// Increase line number
					count++;
				}
			}
		}
	}
	*/

	/*
	static void write_matrix_users_sess(string filename, my_data &data, Matrix2D<var_pointmass> &M) {
		char buffer[100];
		int u;
		int s;
		int t;
		string aux;

		// Build auxiliary user_ids
		std::vector<unsigned long long> user_ids = std::vector<unsigned long long>(data.Nusers);
		for(auto const &it : data.user_ids) {
			user_ids.at(it.second) = it.first;
		}

		// Build auxiliary session_ids
		std::vector<unsigned long long> session_ids = std::vector<unsigned long long>(data.Nsessions);
		for(auto const &it : data.session_ids) {
			session_ids.at(it.second) = it.first;
		}

		// Print values
		int count = 0;
		for(const auto &it : data.trans_ids) {
			u = it.first.u;
			s = it.first.s;
			t = it.second;

			sprintf(buffer,"%d\t%llu\t%llu\t",count,user_ids.at(u),session_ids.at(s));
			aux = string(buffer);
			for(int k=0; k<M.get_size2()-1; k++) {
				sprintf(buffer,"%.12f\t",M.get_object(t,k).e_x);
				aux += string(buffer);
			}
			sprintf(buffer,"%.12f",M.get_object(t,M.get_size2()-1).e_x);
			aux += string(buffer);
			write_line(filename,aux);
			count++;
		}
	}
	*/

	static void write_all_matrices(string label, my_data &data, const my_hyper &hyper, const my_param &param, my_pvar &pvar) {
		// Print output matrices
		write_matrix(param.outdir+"/param_alpha"+label,data.item_ids,pvar.alpha);
		if(!param.flag_symmetricRho) {
			write_matrix(param.outdir+"/param_rho"+label,data.item_ids,pvar.rho);
		} else {
			write_matrix(param.outdir+"/param_rho"+label,data.item_ids,pvar.alpha);
		}
		if(param.flag_userVec) {
			write_matrix(param.outdir+"/param_theta"+label,data.user_ids,pvar.theta);
		}
		if(param.flag_itemIntercept) {
			write_matrix(param.outdir+"/param_lambda0"+label,data.item_ids,pvar.lambda0);
		}
		if(param.flag_price>0) {
			write_matrix(param.outdir+"/param_gamma"+label,data.user_ids,pvar.gamma);
			write_matrix(param.outdir+"/param_beta"+label,data.itemgroup_ids,pvar.beta);
		}
		if(param.flag_day) {
			write_matrix(param.outdir+"/param_mu"+label,data.itemgroup_ids,pvar.mu);
			write_matrix(param.outdir+"/param_delta"+label,data.day_ids,pvar.delta);
		}
	}

	static void write_objective_function(const my_param &param, double logp) {
		char buffer[100];
		sprintf(buffer,"%d\t%.6f",param.it,logp);
		string aux = string(buffer);
		write_line(param.outdir+"/obj_function.txt",aux);
	}

	static void write_vocab(my_data &data, const my_param &param) {
		char buffer[300];
		string filename = param.outdir+"/vocab.txt";
		unsigned long long iName;
		int i;
		for(const auto &it : data.item_ids) {
			iName = it.first;
			i = it.second;
			if(data.lines_per_item.get_object(i).size()>0) {
				sprintf(buffer,"%llu\t%lu",iName,data.lines_per_item.get_object(i).size());
				write_line(filename,string(buffer));
			}
		}
	}
};

#endif
