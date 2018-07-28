/*
 * check.cpp
 *
 *  Created on: 23 Mar 2017
 *      Author: jkiesele
 */




#include <iostream>
#include "friendTreeInjector.h"
#include "rocCurveCollection.h"

int main(){

	//no GUI libs in the miniconda root installation!


	friendTreeInjector in;
	in.addFromFile("/data/shared/BumbleB/20180401_ak8/doubleb_merged_h_q_lessQCD_test_samples_predict/tree_association.txt")

	//in.showList();

	in.createChain();

	std::cout << in.getChain()->GetEntries() <<std::endl;

	//simple Tree->Draw plotting


	rocCurveCollection rocColl;

	rocColl.addROC("Hbb vs QCD", "prob_fj_isHbb", "fj_isH", "fj_isQCD*sample_isQCD", "blue");

	rocColl.printRocs(in.getChain(),"deepdoubleb_roc.pdf");

	//or use the chain as a normal TChain, loop, do plots etc.

}
