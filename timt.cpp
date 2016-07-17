#include <iostream>
#include <armadillo>
#include <vector>
#include <boost/random.hpp>
#include <boost/random/gamma_distribution.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <getopt.h>
#include <math.h>

//#define DEBUG

using namespace std;
using namespace arma;
using namespace boost;

extern "C" {
// qr update for rank 1 matrices
void dqr1up_(int* m, int* n, int* k, double Q[], int* ldq, double R[], int* ldr, double u[], double v[], double w[]);
}

double gamrnd(double shape, double scale, mt19937& rng) {
	gamma_distribution<> gd(shape, scale);
	variate_generator<mt19937&, gamma_distribution<> > g(rng, gd);

	return g();
}

double gampdf(double x, double shape, double scale, bool asLog) {
	double p = 0.0;
	if (asLog) {
		p = -log(tgamma(shape)) - shape * log(scale) + (shape - 1) * log(x) - 1.0 / scale * x;
	} else {
		p = 1.0 / (tgamma(shape) * pow(scale, shape)) * pow(x, shape - 1) * exp(-x / scale);
	}
	return p;
}

double unifrnd(mt19937& rng) {
	uniform_01<> ud;
	variate_generator<mt19937&, uniform_01<> > u(rng, ud);

	return u();
}

void unifrnd(vec* r, mt19937& rng) {
	uniform_01<> ud;
	variate_generator<mt19937&, uniform_01<> > u(rng, ud);

	for (uint i = 0; i < (*r).n_rows; ++i) {
		(*r)(i) = u();
	}
}

//void binomrnd(vec* r, int node, double p, mt19937& rng) {
//	binomial_distribution<> bd(1, p);
//	variate_generator<mt19937&, binomial_distribution<> > b(rng, bd);
//
//	bool repeat = true;
//
//	while (repeat == true) {
//		for (uint i = 0; i < (*r).n_rows; ++i) {
//			(*r)(i) = b();
//			if (repeat == true) {
//				if (i != node && (*r)(i) > 0) {
//					repeat = false;
//				}
//			}
//		}
//	}
//
//}

void binomrnd(vector<int>* v, int n, int node, double p, mt19937& rng) {
	binomial_distribution<> bd(1, p);
	variate_generator<mt19937&, binomial_distribution<> > b(rng, bd);

	bool repeat = true;

	while (repeat == true) {
		for (int i = 0; i < n; ++i) {
			if (b() > 0 && i != node) {
				v->push_back(i);
				if (repeat == true) {
					repeat = false;
				}
			}
		}
	}

}

void randomWalk(int* node, int* flipNode, int* offset, int* offsetDiagonal, mat* W, int* nEdge, mt19937& rng) {

#ifdef DEBUG
	cout << "DEBUG: begin randomWalk" << endl;
#endif

	int n = (*W).n_rows;

	*flipNode = *node;

	while (*flipNode == *node) {
		*flipNode = trunc(unifrnd(rng) * n);
	}

	double r = unifrnd(rng);

	int currentValue = (*W)(*node, *flipNode);
	int nextValue = -2;

	if (currentValue == 0) {
		if (r <= 0.5) {
			nextValue = -1;
		} else {
			nextValue = 1;
		}
		*offsetDiagonal = 1; // edge is added
		(*nEdge)++;
	} else if (currentValue == -1) {
		if (r <= 0.5) {
			nextValue = 0;
			*offsetDiagonal = -1; // edge is removed
			(*nEdge)--;
		} else {
			nextValue = 1;
			*offsetDiagonal = 0; // edge is maintained
		}
	} else if (currentValue == 1) {
		if (r <= 0.5) {
			nextValue = -1;
			*offsetDiagonal = 0; // edge is maintained
		} else {
			nextValue = 0;
			*offsetDiagonal = -1; // edge is removed
			(*nEdge)--;
		}
	}

	(*W)(*node, *flipNode) = nextValue;
	(*W)(*flipNode, *node) = nextValue;
	(*W)(*node, *node) += *offsetDiagonal;
	(*W)(*flipNode, *flipNode) += *offsetDiagonal;

	*offset = nextValue - currentValue;

#ifdef DEBUG
	cout << "DEBUG: end randomWalk" << endl;
#endif
}

void updateQR1(int node, int flipNode, int offset, int offsetDiagonal, int n, mat* Q, mat* R, vec* u, vec* v, vec* w) {

#ifdef DEBUG
	cout << "DEBUG: begin updateQR1" << endl;
#endif

	(*u).fill(0.0);
	(*v).fill(0.0);
	(*w).fill(0.0);

	if (offset == 1 || offset == -1) { // 1 qr update

#ifdef DEBUG
			cout << "DEBUG: 1x QR update" << endl;
#endif

		if (offset == 1 && offsetDiagonal == 1) {
			(*u)(node) = 1;
			(*u)(flipNode) = 1;
			(*v)(node) = 1;
			(*v)(flipNode) = 1;
			//cout << "case 1: offset = " << offset << ", offsetDiagonal = " << offsetDiagonal << ", u * v' =\n" << u * v.t() << endl;
		} else if (offset == 1 && offsetDiagonal == -1) {
			(*u)(node) = 1;
			(*u)(flipNode) = -1;
			(*v)(node) = -1;
			(*v)(flipNode) = 1;
			//cout << "case 2: offset = " << offset << ", offsetDiagonal = " << offsetDiagonal << ", u * v' =\n" << u * v.t() << endl;
		} else if (offset == -1 && offsetDiagonal == 1) {
			(*u)(node) = -1;
			(*u)(flipNode) = 1;
			(*v)(node) = -1;
			(*v)(flipNode) = 1;
			//cout << "case 3: offset = " << offset << ", offsetDiagonal = " << offsetDiagonal << ", u * v' =\n" << u * v.t() << endl;
		} else if (offset == -1 && offsetDiagonal == -1) {
			(*u)(node) = 1;
			(*u)(flipNode) = 1;
			(*v)(node) = -1;
			(*v)(flipNode) = -1;
			//cout << "case 4: offset = " << offset << ", offsetDiagonal = " << offsetDiagonal << ", u * v' =\n" << u * v.t() << endl;
		}

		dqr1up_(&n, &n, &n, (*Q).memptr(), &n, (*R).memptr(), &n, (*u).memptr(), (*v).memptr(), (*w).memptr());
	} else { // 2 qr updates

#ifdef DEBUG
		cout << "DEBUG: 2x QR update" << endl;
#endif

		(*u)(node) = offset;
		(*v)(flipNode) = 1;

		dqr1up_(&n, &n, &n, (*Q).memptr(), &n, (*R).memptr(), &n, (*u).memptr(), (*v).memptr(), (*w).memptr());

		(*w).fill(0.0);

		(*u)(node) = 0; // reset
		(*v)(flipNode) = 0; // reset

		(*u)(flipNode) = offset;
		(*v)(node) = 1;

		dqr1up_(&n, &n, &n, (*Q).memptr(), &n, (*R).memptr(), &n, (*u).memptr(), (*v).memptr(), (*w).memptr());
	}

#ifdef DEBUG
	cout << "DEBUG: end updateQR1" << endl;
#endif

}

void updateQR2(int node, int flipNode, int offset, int offsetDiagonal, double* d1tW1, vec* vecW1, vec* vec1tWS, double* beta, int n, mat* Q, mat* R, mat* S, vec* u, vec* v, vec* a, vec* b, vec* u1,
		vec* v1, vec* w) {

#ifdef DEBUG
	cout << "DEBUG: begin updateQR2" << endl;
#endif

	(*u).fill(0.0);
	(*v).fill(0.0);
	(*w).fill(0.0);

	if (offset == 1 || offset == -1) { // 2 qr update

#ifdef DEBUG
			cout << "DEBUG: 2x QR update" << endl;
#endif

		double d1tu = 0.0;
		double dvt1 = 0.0;

		if (offset == 1 && offsetDiagonal == 1) {
//			cout << "case 1" << endl;
			(*u)(node) = 1;
			(*u)(flipNode) = 1;
			d1tu = 2.0;
			(*v)(node) = 1;
			(*v)(flipNode) = 1;
			dvt1 = 2.0;
		} else if (offset == 1 && offsetDiagonal == -1) {
//			cout << "case 2" << endl;
			(*u)(node) = 1;
			(*u)(flipNode) = -1;
			(*v)(node) = -1;
			(*v)(flipNode) = 1;
		} else if (offset == -1 && offsetDiagonal == 1) {
//			cout << "case 3" << endl;
			(*u)(node) = -1;
			(*u)(flipNode) = 1;
			(*v)(node) = -1;
			(*v)(flipNode) = 1;
		} else if (offset == -1 && offsetDiagonal == -1) {
//			cout << "case 4" << endl;
			(*u)(node) = -1;
			(*u)(flipNode) = -1;
			d1tu = -2.0;
			(*v)(node) = 1;
			(*v)(flipNode) = 1;
			dvt1 = 2.0;

		}

		double factor = 1.0 / (*d1tW1);
		*d1tW1 += d1tu * dvt1; //update
		factor -= 1.0 / (*d1tW1);

		*u1 = 0.5 * (*beta) * (factor * (*vecW1) - 1.0 / (*d1tW1) * dvt1 * (*u));
		*v1 = *vec1tWS;

		// 1st qr update
		dqr1up_(&n, &n, &n, (*Q).memptr(), &n, (*R).memptr(), &n, (*u1).memptr(), (*v1).memptr(), (*w).memptr());

		(*w).fill(0.0);

		*u1 = 0.5 * (*beta) * ((*u) - 1.0 / (*d1tW1) * d1tu * ((*vecW1) + dvt1 * (*u)));
		*v1 = (*v)(node) * (*S).col(node) + (*v)(flipNode) * (*S).col(flipNode);

		// 2nd qr update
		dqr1up_(&n, &n, &n, (*Q).memptr(), &n, (*R).memptr(), &n, (*u1).memptr(), (*v1).memptr(), (*w).memptr());

		if (offset + offsetDiagonal != 0) {
			*vec1tWS += (offset + offsetDiagonal) * ((*S).col(node) + (*S).col(flipNode)); // update
			(*vecW1)(node) += offset + offsetDiagonal; // update
			(*vecW1)(flipNode) += offset + offsetDiagonal; // update
		}

	} else if (offset == 2 || offset == -2) { // 3 qr updates, offsetDiagonal == 0

#ifdef DEBUG
			cout << "DEBUG: 3x QR update" << endl;
#endif

		(*a).fill(0.0);
		(*b).fill(0.0);

		(*u)(node) = offset;
		double d1tu = offset;
		(*v)(flipNode) = 1;

		(*a)(flipNode) = offset;
		double d1ta = offset;
		(*b)(node) = 1;

		double factor = 1.0 / (*d1tW1);
		*d1tW1 += 2 * offset; // update
		factor -= 1.0 / (*d1tW1);

		*u1 = 0.5 * (*beta) * (factor * (*vecW1) - 1.0 / (*d1tW1) * (*u) - 1.0 / (*d1tW1) * (*a));
		*v1 = *vec1tWS;

		// 1st qr update
		dqr1up_(&n, &n, &n, (*Q).memptr(), &n, (*R).memptr(), &n, (*u1).memptr(), (*v1).memptr(), (*w).memptr());

		(*w).fill(0.0);

		*u1 = 0.5 * (*beta) * ((*u) - 1.0 / (*d1tW1) * d1tu * ((*vecW1) + (*u) + (*a)));
		*v1 = (*S).col(flipNode);

		// 2nd qr update
		dqr1up_(&n, &n, &n, (*Q).memptr(), &n, (*R).memptr(), &n, (*u1).memptr(), (*v1).memptr(), (*w).memptr());

		(*w).fill(0.0);

		*u1 = 0.5 * (*beta) * ((*a) - 1.0 / (*d1tW1) * d1ta * ((*vecW1) + (*u) + (*a)));
		*v1 = (*S).col(node);

		// 3rd qr update
		dqr1up_(&n, &n, &n, (*Q).memptr(), &n, (*R).memptr(), &n, (*u1).memptr(), (*v1).memptr(), (*w).memptr());

		*vec1tWS += offset * ((*S).col(node) + (*S).col(flipNode)); // update
		(*vecW1)(node) += offset; // update
		(*vecW1)(flipNode) += offset; // update

	}

#ifdef DEBUG
	cout << "DEBUG: end updateQR2" << endl;
#endif
}

int main(int argc, char** argv) {

	mt19937 rng;
	rng.seed(time(NULL) + getpid());
	//rng.seed(1);

#ifdef DEBUG
	cout << "INFO: DEBUG MODE" << endl;
#endif

	double expScale = 15; //50.0; smaller -> more sparse

	int nIterBatch = 1000;

	int nBatch = 15;

	const char *Sfilename = NULL;

	int input;
	static struct option long_options[] = { };
	int option_index = 0;

	while ((input = getopt_long(argc, argv, "e:s:i:b:", long_options, &option_index)) != -1) {
		switch (input) {
		case 0:
			/* If this option set a flag, do nothing else now. */
			if (long_options[option_index].flag != 0)
				break;
			printf("option %s", long_options[option_index].name);
			if (optarg)
				printf(" with arg %s", optarg);
			printf("\n");
			break;
		case 's':
			Sfilename = optarg;
			break;
		case 'e':
			expScale = atof(optarg);
			break;
		case 'i':
			nIterBatch = atoi(optarg);
			break;
		case 'b':
			nBatch = atoi(optarg);
			break;
		default: /* '?' */
			fprintf(stderr, "Usage: %s -e 10 -s S.txt\n", argv[0]);
			return 1;
		}
	}

	if (Sfilename == NULL) {
		cout << "INFO: No S matrix supplied!" << endl;
		//Sfilename = "S.txt";
		Sfilename = "/home/david/Software/timt_pathways/K_Bhatta_XII.txt";
		//return 0;
	}
	cout << "INFO: Using S matrix from file: " << Sfilename << endl;

	mat S, Q, W, WProposal, WQS;
	S.load(Sfilename);

	int n = S.n_rows;

	cout << "INFO: n = " << n << endl;

	cout << "INFO: expScale = " << expScale << endl;

	// rank 1 update vectors
	vec u = zeros<vec>(n);
	vec v = zeros<vec>(n);
	vec a = zeros<vec>(n);
	vec b = zeros<vec>(n);

	vec u1 = zeros<vec>(n);
	vec v1 = zeros<vec>(n);
	vec w = zeros<vec>(2 * n); // workspace vector for qr update

	double diagTerm = 5e-1;

	W = -1.0 * ones<mat>(n, n);
	for (int i = 0; i < n; ++i) {
		W(i, i) = n - 1 + diagTerm;
	}
	int nEdge = 0.5 * n * (n - 1);

	int nEdgeProposal = nEdge;

	WProposal = W;

	//double d = 2.0 * n; // initial d
	double d = 1.0 * n; // initial d

	mat Q1, R1, Q1Proposal, R1Proposal, Q2, R2, Q2Proposal, R2Proposal;

	double vFactor = 2.0; // factor which determines the balance between both deterimants, must obey: vFactor > 1 + (n - 2)/d
	cout << "INFO: vFactor = " << vFactor << endl;
	//double scaleF = 3.0; // parameter for transition probability of beta and beta proposal
	double s = 150.0; //75.0; // parameter for beta proposal
	double gammaShape = 5.0; // prior for beta
	double gammaScale = 0.8; // prior for beta

	double kGamma = 1.1; // shape parameter for sparsity inducing gamma prior

	double beta = 2.0;
	double betaProposal = -1.0;

	double logPosteriorBeta = 0.0;
	double logPosteriorBetaProposal = 0.0;

	double logAcceptanceBeta = 0.0;
	int acceptedBeta = 0;
	int acceptedFlip = 0;
	int totalAcceptedFlip = 0;

	int flipNode = -1;
	int offset = -2;
	int offsetDiagonal = -2;

	qr(Q1, R1, W);

	double d1tW1 = sum(sum(W));
	double d1tW1Proposal = d1tW1;
	vec vecW1 = sum(W, 1);
	vec vecW1Proposal = vecW1;
	vec vec1tWS = (vecW1.t() * S).t();
	vec vec1tWSProposal = vec1tWS;

	int burninBatch = round(0.5 * nBatch) - 1;

	bool annealFlag = true;
	double annealingStopThreshold = 0.1; // stop annealing when below ... accepted flips in %
	double annealingFactor = 6.0; //12.0; // the final d will be annealingFactor * d

	cout << "INFO: nBatch = " << nBatch << endl;
	cout << "INFO: nIterBatch = " << nIterBatch << endl;

	double sfactor = pow(annealingFactor, 1.0 / (nIterBatch * nBatch)); // annealing

	double logLikelihood;
	double logLikelihoodProposal;
	double logPrior;
	double logPriorProposal;
	double logPosterior;
	double logPosteriorProposal;

	vec sumW = zeros<vec>(n);
	mat WBatch = zeros<mat>(n, n);

	//vec logLikelihoodVector = zeros<vec>(nBatch * nIterBatch);
	//int counter = 0;

	wall_clock timer;

	timer.tic();

	for (int batch = 0; batch < nBatch; ++batch) {

		cout << "batch " << batch + 1;

		if (batch <= burninBatch - 1) {
			cout << ", burnin, d = " << d << endl;
		} else {
			cout << ", averaging, d = " << d << endl;
		}

		bool alteredW = true;

		acceptedFlip = 0;

		for (int iter = 0; iter < nIterBatch; ++iter) {
			//cout << "batch = " << batch << ", iter = " << iter << endl;

			if (annealFlag) {
				d = d * sfactor;
			}

			// beta ----------------------------------
			betaProposal = gamrnd(s, beta / s, rng);

			if (alteredW) { // if no flip was accepted in the previous loop, W, WQS, Q2 and R2 did not change
				//Q = eye<mat>(n, n) - 1.0 / sum(sum(W)) * ones<mat>(n, n) * W;
				//WQS = W * Q * S;
				sumW = sum(W, 1);
				WQS = (W - 1.0 / sum(sumW) * sumW * sumW.t()) * S;

				qr(Q2, R2, eye<mat>(n, n) + 0.5 * beta * WQS);

				alteredW = false;
			}

			// here: 0.5 * beta to make it identical to the R code. recompute, because d changes every iteration
			//logPosteriorBeta = 0.5 * n * d * log(0.5 * beta) - 0.5 * vFactor * d * sum(log(abs(R2.diag()))) + 0.5 * d * gampdf(0.5 * beta, 1.5 * scaleF, 1.0 / scaleF, true);

			// here: beta instead of 0.5 * beta
			logPosteriorBeta = 0.5 * n * d * log(beta) - 0.5 * vFactor * d * sum(log(abs(R2.diag()))) + 0.5 * d * gampdf(beta, gammaShape, gammaScale, true);

			qr(Q2Proposal, R2Proposal, eye<mat>(n, n) + 0.5 * betaProposal * WQS);

			// here: 0.5 * betaProposal to make it identical to the R code
			//logPosteriorBetaProposal = 0.5 * n * d * log(0.5 * betaProposal) - 0.5 * vFactor * d * sum(log(abs(R2Proposal.diag()))) + 0.5 * d * gampdf(0.5 * betaProposal, 1.5 * scaleF, 1.0 / scaleF, true);

			// here: beta instead of 0.5 * beta
			logPosteriorBetaProposal = 0.5 * n * d * log(betaProposal) - 0.5 * vFactor * d * sum(log(abs(R2Proposal.diag()))) + 0.5 * d * gampdf(betaProposal, gammaShape, gammaScale, true);

			// here: 0.5 * beta{Proposal} ... not needed, because factor cancels
			logAcceptanceBeta = logPosteriorBetaProposal - logPosteriorBeta + gampdf(beta, s, betaProposal / s, true) - gampdf(betaProposal, s, beta / s, true);

			if (isinf(logAcceptanceBeta) || isnan(logAcceptanceBeta)) {
				cout << "batch " << batch << ", iter = " << iter << endl;
				cout << "logAcceptanceBeta = " << logAcceptanceBeta << ", beta = " << beta << ", betaProposal = " << betaProposal << endl;
				cout << "logPosteriorBetaProposal = " << logPosteriorBetaProposal << ", logPosteriorBeta = " << logPosteriorBeta << endl;
				cout << "log(prod(abs(R2.diag()))) =\n" << log(prod(abs(R2.diag()))) << endl;
				cout << "R2.diag() =\n" << R2.diag() << endl;
				cout << "sum(log(abs(R2.diag()))) = " << sum(log(abs(R2.diag()))) << endl;
				cout << "det = " << det(Q2 * R2) << endl;
				cout << "det(I_n + 0.5 * beta * WQS) = " << det(eye<mat>(n, n) + 0.5 * beta * WQS) << endl;
				cout << "Q2 =\n" << Q2 << ", R2 =\n" << R2 << endl;
				cout << "det(Q2) = " << det(Q2) << ", det(R2) = " << det(R2) << endl;
				cout << "exiting..." << endl;
				return 0;
			}

			if (log(randu()) < logAcceptanceBeta) {
				beta = betaProposal;
				acceptedBeta++;
				Q2 = Q2Proposal;
				R2 = R2Proposal;

				logPosteriorBeta = logPosteriorBetaProposal;
#ifdef DEBUG
				cout << "DEBUG: accepted betaProposal" << endl;
#endif
			}
			// end beta ----------------------------------

			logLikelihood = -0.5 * d * log(d1tW1) + 0.5 * d * sum(log(abs(R1.diag()))) - 0.5 * vFactor * d * sum(log(abs(R2.diag())));

			//logPrior = - 0.5 * d * double(nEdge) / expScale; // exponential prior
			logPrior = 0.5 * d * gampdf(nEdge, kGamma, expScale, true); // gamma prior
			logPosterior = logLikelihood + logPrior;

			// flip ----------------------------------
			for (int node = 0; node < n; ++node) {

				WProposal = W;
				nEdgeProposal = nEdge;

				Q1Proposal = Q1;
				R1Proposal = R1;
				Q2Proposal = Q2;
				R2Proposal = R2;

				d1tW1Proposal = d1tW1;
				vecW1Proposal = vecW1;
				vec1tWSProposal = vec1tWS;

#ifdef DEBUG
				double logdet1pre_true = log(det(WProposal));
				double logdet1pre_update = sum(log(abs(R1Proposal.diag())));
				if (abs(logdet1pre_true - logdet1pre_update) > 0.001) {
					cout << "DEBUG: logdet1pre_true = " << logdet1pre_true << ", logdet1pre_update = " << logdet1pre_update << endl;
					return 0;
				}
#endif

				randomWalk(&node, &flipNode, &offset, &offsetDiagonal, &WProposal, &nEdgeProposal, rng);

				updateQR1(node, flipNode, offset, offsetDiagonal, n, &Q1Proposal, &R1Proposal, &u, &v, &w);

#ifdef DEBUG
				double logdet1_true = log(det(WProposal));
				double logdet1_update = sum(log(abs(R1Proposal.diag())));
				if (abs(logdet1_true - logdet1_update) > 0.001) {
					cout << "DEBUG: logdet1_true = " << logdet1_true << ", logdet1_update = " << logdet1_update << endl;
					return 0;
				} else {
					cout << "DEBUG: logdet1 passed" << endl;
				}
#endif

				updateQR2(node, flipNode, offset, offsetDiagonal, &d1tW1Proposal, &vecW1Proposal, &vec1tWSProposal, &beta, n, &Q2Proposal, &R2Proposal, &S, &u, &v, &a, &b, &u1, &v1, &w);

#ifdef DEBUG
				mat Q_debug = eye<mat>(n, n) - 1.0 / sum(sum(WProposal)) * ones<mat>(n, n) * WProposal;
				mat WQS_debug = WProposal * Q_debug * S;

				double logdet2_true = log(det(eye<mat>(n, n) + 0.5 * beta * WQS_debug));
				double logdet2_update = sum(log(abs(R2Proposal.diag())));
				if (abs(logdet2_true - logdet2_update) > 0.001) {

					cout << "true:\n" << eye<mat>(n, n) + 0.5 * beta * WQS_debug << endl;
					cout << "update:\n" << Q2Proposal * R2Proposal << endl;

					cout << "DEBUG: logdet2_true = " << logdet2_true << ", logdet2_update = " << logdet2_update << endl;
					return 0;
				} else {
					cout << "DEBUG: logdet2 passed" << endl;
				}
#endif

				logLikelihoodProposal = -0.5 * d * log(d1tW1Proposal) + 0.5 * d * sum(log(abs(R1Proposal.diag()))) - 0.5 * vFactor * d * sum(log(abs(R2Proposal.diag())));

				//logPriorProposal = - 0.5 * d * double(nEdgeProposal) / expScale; // exponential prior
				logPriorProposal = 0.5 * d * gampdf(nEdgeProposal, kGamma, expScale, true); // gamma prior
				logPosteriorProposal = logLikelihoodProposal + logPriorProposal;

				if (isinf(logPosteriorProposal) || isnan(logPosteriorProposal)) {
					cout << "*** DEBUG ***" << endl;
					cout << "logPosteriorProposal = " << logPosteriorProposal << endl;
					cout << "logLikelihoodProposal = " << logLikelihoodProposal << ", logPriorProposal = " << logPriorProposal << endl;
					cout << "log(d1tW1Proposal) = " << log(d1tW1Proposal) << ", log(prod(R1Proposal.diag()) = " << log(prod(R1Proposal.diag())) << ", log(prod(R2Proposal.diag())) = "
							<< log(prod(R2Proposal.diag())) << endl;
					cout << "proposal =\n" << Q2Proposal * R2Proposal << endl;

					Q = eye<mat>(n, n) - 1.0 / sum(sum(WProposal)) * ones<mat>(n, n) * WProposal;
					WQS = WProposal * Q * S;

					cout << "WProposal =\n" << eye<mat>(n, n) + 0.5 * beta * WQS << endl;
					cout << "*** END DEBUG ***" << endl;
					return 0;
				}

				if (isinf(logPosterior) || isnan(logPosterior)) {
					cout << "*** DEBUG ***" << endl;
					cout << "logPosterior = " << logPosterior << endl;
					cout << "logLikelihood = " << logLikelihood << ", logPrior = " << logPrior << endl;
					cout << "log(d1tW1) = " << log(d1tW1Proposal) << ", log(prod(R1.diag()) = " << log(prod(R1Proposal.diag())) << ", log(prod(R2.diag())) = " << log(prod(R2Proposal.diag())) << endl;
					cout << "*** END DEBUG ***" << endl;
					return 0;
				}

				if (log(unifrnd(rng)) < logPosteriorProposal - logPosterior) {
					acceptedFlip++;

					W = WProposal;
					nEdge = nEdgeProposal;

					Q1 = Q1Proposal;
					R1 = R1Proposal;
					Q2 = Q2Proposal;
					R2 = R2Proposal;

					d1tW1 = d1tW1Proposal;
					vecW1 = vecW1Proposal;
					vec1tWS = vec1tWSProposal;

					logPosterior = logPosteriorProposal;

					alteredW = true;

#ifdef DEBUG
					cout << "DEBUG: accepted WProposal" << endl;
#endif
				}
			}
			// end flip ----------------------------------

			if (batch > burninBatch - 1) {
				WBatch += W;
			}
			//logLikelihoodVector(counter) = logLikelihood / d;
			//counter++;

		} // for 1 batch

		totalAcceptedFlip += acceptedFlip;

		if ((batch > 4) && (100.0 * acceptedFlip / double(n * nIterBatch) < annealingStopThreshold) && annealFlag == true) {
			cout << "  accepted flips = " << 100.0 * acceptedFlip / double(n * nIterBatch) << " %, stopped annealing" << endl;
			annealFlag = false;
		} else {
			cout << "  accepted flips = " << 100.0 * acceptedFlip / double(n * nIterBatch) << " %" << endl;
		}
		cout << "  beta = " << beta << endl;
		cout << "  nEdge = " << nEdge << endl;

	} // for all batches

	double n_secs = timer.toc();
	cout << "time: " << n_secs << " seconds elapsed" << endl;

	//WBatch = WBatch / max(max(WBatch));
	WBatch = WBatch / ((nBatch - burninBatch) * nIterBatch);
	WBatch.save("WBatch.txt", csv_ascii);
	//cout << "WBatch =\n" << WBatch << endl;

	//cout << "round(WBatch) =\n" << round(WBatch) << endl;

	//cout << "W =\n" << W << endl;
	//char filename[50];
	//sprintf(filename, "WBatch_e%.3f.txt", expScale);
	//WBatch.save(filename, csv_ascii);

	W.save("W.txt", csv_ascii);

	cout << "total accepted beta = " << 100.0 * double(acceptedBeta) / double(nIterBatch * nBatch) << " %" << endl;
	cout << "total accepted flips = " << 100.0 * double(totalAcceptedFlip) / double(nIterBatch * n * nBatch) << " %" << endl;

	cout << "nEdge = " << nEdge << endl;
	cout << "beta = " << beta << endl;

	cout << "d = " << d << endl;

	return 0;
}
