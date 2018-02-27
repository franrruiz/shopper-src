#ifndef MATRICES_HPP
#define MATRICES_HPP

template<class T>
class Matrix1D {
private:
	int N;
	T *p;

public:
	Matrix1D() {
		N = 0;
		p = nullptr;
	}

	Matrix1D(const Matrix1D<T> &obj) {
		N = obj.N;
		p = new T[N];
		for(long long int n=0; n<N; n++) {
			p[n] = obj.p[n];
		}
	}

	Matrix1D(int N__) {
		N = N__;
		p = new T[N];
	}

	~Matrix1D() {
		delete [] p;
	}

	inline T& get_object(int n) {
		return p[n];
	}

	inline void set_object(int n, T obj) {
		p[n] = obj;
	}

	inline T *get_pointer(unsigned long long n) const {
		return(p+n);
	}

	inline int get_size1() {
		return N;
	}

	Matrix1D<T> & operator=(const Matrix1D<T> &rhs) {
		// Check for self-assignment!
		if (this!=&rhs) {
			delete [] p;			// deallocate memory that Matrix1D uses internally
			p = new T[rhs.N];	// Allocate memory to hod the contents of rhs
			N = rhs.N;				// Copy values from rhs
			for(long long int n=0; n<N; n++) {
				p[n] = *rhs.get_pointer(n);
			}
		}
		return *this;
	}
};

template<class T>
class Matrix2D {
private:
	int N;
	int M;
	T *p;

public:
	Matrix2D() {
		N = 0;
		M = 0;
		p = nullptr;
	}

	Matrix2D(const Matrix2D<T> &obj) {
		N = obj.N;
		M = obj.M;
		p = new T[N*M];
		for(long long int n=0; n<N*M; n++) {
			p[n] = obj.p[n];
		}
	}

	Matrix2D(int N__, int M__) {
		N = N__;
		M = M__;
		p = new T[N*M];
	}

	~Matrix2D() {
		delete [] p;
	}

	inline T& get_object(int n, int m) {
		return p[n*M+m];
	}

	inline void set_object(int n, int m, T obj) {
		p[n*M+m] = obj;
	}

	inline T *get_pointer(unsigned long long n) const {
		return(p+n);
	}

	inline int get_size1() {
		return N;
	}

	inline int get_size2() {
		return M;
	}

	Matrix2D<T> & operator=(const Matrix2D<T> &rhs) {
		// Check for self-assignment!
		if (this!=&rhs) {
			delete [] p;			// deallocate memory that Matrix1D uses internally
			p = new T[rhs.N*rhs.M];	// Allocate memory to hod the contents of rhs
			N = rhs.N;				// Copy values from rhs
			M = rhs.M;
			for(long long int n=0; n<N*M; n++) {
				p[n] = *rhs.get_pointer(n);
			}
		}
		return *this;
	}
};

template<class T>
class Matrix3D {
private:
	int N;
	int M;
	int P;
	T *p;

public:
	Matrix3D() {
		N = 0;
		M = 0;
		P = 0;
		p = nullptr;
	}

	Matrix3D(const Matrix3D<T> &obj) {
		N = obj.N;
		M = obj.M;
		P = obj.P;
		p = new T[N*M*P];
		for(long long int n=0; n<N*M*P; n++) {
			p[n] = obj.p[n];
		}
	}

	Matrix3D(int N__, int M__, int P__) {
		N = N__;
		M = M__;
		P = P__;
		p = new T[N*M*P];
	}

	~Matrix3D() {
		delete [] p;
	}

	inline T& get_object(int n, int m, int p_) {
		return p[P*M*n+P*m+p_];
	}

	inline void set_object(int n, int m, int p_, T obj) {
		p[P*M*n+P*m+p_] = obj;
	}

	inline T *get_pointer(unsigned long long n) const {
		return(p+n);
	}

	inline int get_size1() {
		return N;
	}

	inline int get_size2() {
		return M;
	}

	inline int get_size3() {
		return P;
	}

	Matrix3D<T> & operator=(const Matrix3D<T> &rhs) {
		// Check for self-assignment!
		if (this!=&rhs) {
			delete [] p;					// deallocate memory that Matrix1D uses internally
			p = new T[rhs.N*rhs.M*rhs.P];	// Allocate memory to hod the contents of rhs
			N = rhs.N;						// Copy values from rhs
			M = rhs.M;
			P = rhs.P;
			for(long long int n=0; n<N*M*P; n++) {
				p[n] = *rhs.get_pointer(n);
			}
		}
		return *this;
	}
};

#endif
