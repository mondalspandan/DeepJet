import sys
import os
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pandas as pd
import h5py

def make_plots(outputDir, savedir="Plots"):
	print "Making standard plots"	
	dt = pd.read_pickle(outputDir+'/output.pkl')
	labels =  list(dt.columns)
	truthnames = [label[len("truth"):] for label in labels if label.startswith("truth")]
	def cut(tdf):
		ptlow, pthigh = 300, 2500
		mlow, mhigh = 40, 200
		cdf = tdf[(tdf.fj_pt < pthigh) & (tdf.fj_pt>ptlow) &(tdf.fj_sdmass < mhigh) & (tdf.fj_sdmass>mlow)]
		return cdf
	dt = cut(dt)

	savedir = os.path.join(outputDir,savedir)
	
	def make_dirs(dirname):
	    import os, errno
	    """
	    Ensure that a named directory exists; if it does not, attempt to create it.
	    """
	    try:
		os.makedirs(dirname)
	    except OSError, e:
	        if e.errno != errno.EEXIST:
		    raise

	make_dirs(savedir)           
 
	def dists(xdf, savedir="", log=False):
	    labels =  xdf.columns
	    truths = [label[len("truth"):] for label in labels if label.startswith("truth")]
	    print "Labels: ", truths
	    def distribution(xdf, predict="Hcc", log=False):
	        plt.figure(figsize=(10,7))
	        bins = np.linspace(0,1,70)
	        trus = []
	        for tru in truths:
        	    t = xdf['truth'+tru].values
	            trus.append(t)
            
	        preds = [xdf['predict'+predict].values]*len(truths)
	        plt.hist(preds, bins=bins, weights = trus, alpha=0.8, normed=True, label=truths, stacked=True)
	        plt.xlabel("Probability "+predict)
	        plt.title("Stacked Distributions")
	        if log: plt.semilogy()
	        plt.legend(title="True labels:")
        	make_dirs(savedir)
	        if log: plt.savefig(os.path.join(savedir,'LogProbability_'+predict+'.png'), dpi=400)
	        else:plt.savefig(os.path.join(savedir,'Probability_'+predict+'.png'), dpi=400)
            
	    def overlay_distribution(xdf, predict="Hcc"):
	        plt.figure(figsize=(10,7))
	        bins = np.linspace(0,1,70)
	        trus = []
	        for tru in truths:
	            t = xdf['truth'+tru].values
	            trus.append(t)
            
	        pred = xdf['predict'+predict].values
	        #pred = xdf['predict'+predict].values.transpose()
	        for tru, weight in zip(truths, trus):
	            plt.hist(pred, bins=bins, weights = weight, alpha=0.3, normed=True, label=tru)
	        plt.xlabel("Probability "+predict)
	        plt.title("Normalized Distributions")
	        plt.legend(title="True labels:")
	        make_dirs(savedir)
	        plt.savefig(os.path.join(savedir,'Norm_Probability_'+predict+'.png'), dpi=400)
        
       		# DoubleB
	    def db(xdf, log=False):
	        plt.figure(figsize=(10,7))
        	bins = np.linspace(-1,1,70)
	        trus = []
	        for tru in truths:
	            trus.append(xdf['truth'+tru].values)
	        preds = [xdf['fj_doubleb'].values]*len(truths)

        	plt.hist(preds, bins=bins, weights = trus, alpha=0.8, normed=False, label=truths, stacked=True)
	        plt.xlabel("Probability DoubleB (BDT)")
	        plt.title("Stacked Distributions")
	        if log: plt.semilogy()
	        plt.legend(title="True labels:")
	        if log: plt.savefig(os.path.join(savedir,'LogProbability_DB.png'), dpi=400)
	        else: plt.savefig(os.path.join(savedir,'Probability_DB.png'), dpi=400)
            
	    for pred in truths:
	        distribution(xdf, predict = pred)        
	        distribution(xdf, predict = pred, log=True)
        	overlay_distribution(xdf, predict = pred)
	    db(xdf)
	    db(xdf, log=True)

        def roc_input(xdf, signal=["HCC"], include = ["HCC", "Light", "gBB", "gCC", "HBB"]):       
            # Bkg def - filter unwanted
            bkg = np.zeros(len(eval("xdf.truth"+include[0])))
            for label in include:
                bkg = np.add(bkg,  xdf['truth'+label].values)
            bkg = [bool(x) for x in bkg]
            tdf = xdf[bkg] #tdf for temporary df

            # Signal
            truth = np.zeros(len(eval("tdf.truth"+signal[0])))
            predict = np.zeros(len(eval("tdf.truth"+signal[0])))
            for label in signal:
                truth   += np.add(truth,  tdf['truth'+label].values)
                predict += np.add(predict,  tdf['predict'+label].values)

            db = tdf.fj_doubleb.values
            return truth, predict, db

        def single_roc(frame, sculp_label="", savedir=""):
            labels = frame.columns.values
            labels = [label[len("truth"):] for label in labels if label.startswith("truth")]
            if "cc" in labels:
                sig = ["cc"]
                bkg = [l for l in labels if l not in sig]
		if len(sculp_label) > 0: bkg = [sculp_label]
                truth, predict, db =  roc_input(frame, signal=sig, include = sig+bkg)
            else:
                sig = ["Hcc"]
                bkg = [l for l in labels if l not in sig]
		if len(sculp_label) > 0: bkg = [sculp_label]
                truth, predict, db =  roc_input(frame, signal=sig, include = sig+bkg)

            plt.figure(figsize=(10,7))
            fpr, tpr, threshold = roc_curve(truth, predict)
            plt.plot(tpr, fpr, lw=1, label="DeepDoubleC %s, auc= %.1f%%"%("", auc(fpr,tpr)*100))
            print "DeepDoubleC %s, auc= %.1f%%"%("", auc(fpr,tpr)*100), "Sig:", sig, "Bkg:", bkg
	    plt.xlim(0,1)
            plt.ylim(0.001,1)
            plt.xlabel('Signal: '+", ".join(sig)+' efficiency')
            plt.ylabel('Bkg: '+", ".join(bkg)+' mistag')
            plt.grid()
            plt.semilogy()
            plt.legend(title="Pt ["+str(round((min(frame.fj_pt))))+" - "+str(round((max(frame.fj_pt))))+"]\n" \
                       + "m ["+str(round((min(frame.fj_sdmass))))+" - "+str(round((max(frame.fj_sdmass))))+"]" )
	    plt.savefig(os.path.join(savedir,'ROC_vs_'+sculp_label+'.png'), dpi=400)
  
        for lab in truthnames:
            single_roc(dt, sculp_label=lab, savedir=savedir) # against each specific bkg component
        single_roc(dt, sculp_label="", savedir=savedir) # against all bkg

        def sculpting(tdf, sculp_label='Light', savedir=""):
            def find_nearest(array,value):
                idx = (np.abs(array-value)).argmin()
                return idx, array[idx]

            labels =  tdf.columns
            labels = [label[len("truth"):] for label in labels if label.startswith("truth")]

            truth, predict, db = roc_input(tdf, signal=["Hcc"], include = ["Hcc", sculp_label])
            fpr, tpr, threshold = roc_curve(truth, predict)

            cuts = {}
            for wp in [0.01, 0.05, 0.1, 0.25, 0.5]: # % mistag rate
                idx, val = find_nearest(fpr, wp)
                cuts[str(wp)] = threshold[idx] # threshold for deep double-b corresponding to ~1% mistag rate

            plt.figure(figsize=(10,7))
            bins = np.linspace(40,200,41)
            for wp, cut in reversed(sorted(cuts.iteritems())):
		ctdf = tdf.loc[tdf.predictHcc.values > cut ]
                weight = ctdf['truth'+sculp_label].values
                plt.hist(ctdf['fj_sdmass'].values, bins=bins, weights = weight, normed=True,histtype='step',label=sculp_label+' %i%% mis-tag'%(float(wp)*100.))

            plt.xlabel(r'$m_{\mathrm{SD}}$')
            plt.legend(loc='upper right')
            plt.savefig(os.path.join(savedir,'M_sculpting_'+sculp_label+'.png'), dpi=400)

            plt.figure(figsize=(10,7))
            bins = np.linspace(300,2500,41)
            for wp, cut in reversed(sorted(cuts.iteritems())):

                ctdf = tdf.loc[tdf.predictHcc.values > cut ]
 	        weight = ctdf['truth'+sculp_label].values
		#print weight
		#print ctdf['fj_pt'].values.transpose()[0]
                plt.hist(ctdf['fj_pt'].values, bins=bins, weights = weight, normed=True,histtype='step',label=sculp_label+' %i%% mis-tag'%(float(wp)*100.))

            plt.xlabel(r'$pt$')
            plt.legend(loc='upper right')
            plt.savefig(os.path.join(savedir,'Pt_sculpting_'+sculp_label+'.png'), dpi=400)

	dists(dt, savedir=savedir)
	
	for lab in truthnames:
	    sculpting(dt, sculp_label=lab, savedir=savedir)
	
        def pt_dep(xdf, bkg_label="", savedir=savedir):
            def roc(xdf, ptlow=300, pthigh=2500, verbose=False):
                tdf = xdf[(xdf.fj_pt < pthigh) & (xdf.fj_pt>ptlow)]
            	truth, predict, db = roc_input(tdf, signal=["Hcc"], include = ["Hcc", bkg_label])
	        fpr, tpr, threshold = roc_curve(truth, predict)
                return fpr, tpr

            step = float((2500-300))/10
            pts = np.arange(300+step,2500+step,step)
            efftight, effloose = [], []
            mistight, misloose = [], []
            def find_nearest(array,value):
                    idx = (np.abs(array-value)).argmin()
                    return idx, array[idx]

            for pt in pts:
                fpr, tpr = roc(xdf, pt-step,pt)
                ix, mistag =  find_nearest(fpr, 0.1)
                effloose.append(tpr[ix])
                ix, mistag =  find_nearest(fpr, 0.01)
                efftight.append(tpr[ix])
                ix, eff =  find_nearest(tpr, 0.76)
                misloose.append(fpr[ix])
                ix, eff =  find_nearest(tpr, 0.4)
                mistight.append(fpr[ix])

            fig = plt.figure(figsize=(10,7))
            plt.errorbar(pts-step/2, effloose,  xerr=step/2, fmt='o', label='10% mistag', c='blue')
            plt.errorbar(pts-step/2, efftight,  xerr=step/2, fmt='o', label='1% mistag', c='black')
            plt.ylim(0,1)
            plt.xlim(300,2500)
            plt.ylabel('Efficiency')
            plt.xlabel('$Pt$')
            plt.legend()
            plt.show()
            plt.savefig(os.path.join(savedir,'Efficiency_ptdep_'+bkg_label+'.png'), dpi=400)

            fig = plt.figure(figsize=(10,7))
            plt.errorbar(pts-step/2, misloose,  xerr=step/2, fmt='o', label='10% mistag', c='blue')
            plt.errorbar(pts-step/2, mistight,  xerr=step/2, fmt='o', label='1% mistag', c='black')
            plt.ylim(0,1)
            plt.xlim(300,2500)
            plt.ylabel('Mistag')
            plt.xlabel('$Pt$')
            plt.legend()
            plt.show()
            plt.savefig(os.path.join(savedir,'Mistag_ptdep_'+bkg_label+'.png'), dpi=400)

	for lab in truthnames:
	    pt_dep(dt, bkg_label=lab, savedir=savedir)

	def spectator_val(xdf, savedir="", feature="fj_pt"):
            labels =  xdf.columns
            truths = [label[len("truth"):] for label in labels if label.startswith("truth")]
            print "Labels: ", truths
            def distribution(xdf, predict="Hcc", log=False):
                plt.figure(figsize=(10,7))
                trus = []
                for tru in truths:
                    t = xdf['truth'+tru].values
                    trus.append(t)
	        
		feature_vals = xdf[feature].values
		for tru, weight in zip(truths, trus):
	            plt.hist(feature_vals, bins=70, weights = weight, alpha=0.4, normed=True, label=tru)
	        plt.xlabel(feature)
		plt.ylabel("Events")
	        plt.title("Spectator Distribution")

                plt.savefig(os.path.join(savedir,'Feature_'+feature+'.png'), dpi=400)

	for lab in truthnames:
	    spectator_val(dt, savedir=savedir, feature="fj_pt")
	    spectator_val(dt, savedir=savedir, feature="fj_sdmass")
		


	print "Finished Plotting"	    
#make_plots("/home/anovak/data/dev/devDDCWeights_flatten/eval-all")
