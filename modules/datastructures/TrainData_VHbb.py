from __future__ import print_function

from DeepJetCore.TrainData import TrainData, fileTimeOut
import numpy

class TrainData_VHbb(TrainData):
    
    def __init__(self):
        '''
        This class is meant as a base class for the FatJet studies
        You will not need to edit it for trying out things
        '''
        TrainData.__init__(self)
        
        #define truth:
	self.treename = "tree"
        self.undefTruth=['']
        self.truthclasses=['sig','bkg']
        self.referenceclass='sig' ## used for pt reshaping
        self.registerBranches([])

        self.weightbranchX='H_pt'
        self.weightbranchY='H_mass'

        self.weight_binX = numpy.array([0,3000],dtype=float)
        self.weight_binY = numpy.array([0,3000], dtype=float)

        #this is only needed because the truth definitions are different from deepFlavour
        self.allbranchestoberead=[]
        self.registerBranches(self.undefTruth)
        self.registerBranches(self.truthclasses)
        print("Branches read:", self.allbranchestoberead)
        

#######################################
            
class TrainData_VHbb_bdt(TrainData_VHbb):
    
    def __init__(self):
        '''
        This is an example data format description for FatJet studies
        '''
        TrainData_VHbb.__init__(self)
#	self.remove = False
#	self.weight = False
        
        #example of how to register global branches
        self.addBranches(['sig',
                          'bkg'
                      ])
        
        self.addBranches(['H_pt',
                          'H_mass',
                          'V_pt',
                          'hJets_btagCSV_1',
                          'hJets_btagCSV_0',
                          'Top1_mass_fromLepton_regPT_w4MET',
                          'HVdPhi',
                          #'nAddJet_f',
                          'lepMetDPhi',
                          #'softActivityVH_njets5',
                          'V_mt',
                          'met_pt',
			  'hJets_pt_0',
			'hJets_pt_1',
			'hJets_eta_0',
			'hJets_eta_1',
			'selLeptons_pt_0',
			'selLeptons_eta_0',
			#'selLeptons_rellso_0',
			'jjWPtBalance',
			'AddJets252p9_puid_leadJet_btagCSV'
                          ])

    def readFromRootFile(self,filename,TupleMeanStd, weighter):
        
        #the first part is standard, no changes needed
        from DeepJetCore.preprocessing import MeanNormApply, MeanNormZeroPad, MeanNormZeroPadParticles, ZeroPadParticles
        import numpy
        import ROOT
        
        fileTimeOut(filename,120) #give eos 2 minutes to recover
        rfile = ROOT.TFile(filename)
        tree = rfile.Get("tree")
        self.nsamples=tree.GetEntries()
        
        #the definition of what to do with the branches
        
        # those are the global branches (jet pt etc)
        # they should be just glued to each other in one vector
        # and zero padded (and mean subtracted and normalised)
        #x_global = MeanNormZeroPad(filename,TupleMeanStd,
        #                           [self.branches[0]],
        #                           [self.branchcutoffs[0]],self.nsamples)
        
        # the second part (the pf candidates) should be treated particle wise
        # an array with (njets, nparticles, nproperties) is created
    
        x_glb  = ZeroPadParticles(filename,TupleMeanStd,
                                          self.branches[0],
                                          self.branchcutoffs[0],self.nsamples)

	x_dbr  = ZeroPadParticles(filename,TupleMeanStd, self.branches[1], self.branchcutoffs[1],self.nsamples)

        x_db  = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                         self.branches[1],
                                         self.branchcutoffs[1],self.nsamples)
        
        
        Tuple = self.readTreeFromRootToTuple(filename)
        notremoves=weighter.createNotRemoveIndices(Tuple)
        if self.weight:
            weights=weighter.getJetWeights(Tuple)
        elif self.remove:
            weights=notremoves
        else:
            print('neither remove nor weight')
            weights=numpy.empty(self.nsamples)
            weights.fill(1.)
	empty = numpy.empty(self.nsamples)
            
        # create all collections:
        truthtuple =  Tuple[self.truthclasses]
        alltruth=self.reduceTruth(truthtuple)
        undef=numpy.sum(alltruth,axis=1)
        weights=weights[undef > 0]
        x_glb=x_glb[undef > 0]
        x_db=x_db[undef > 0]
        alltruth=alltruth[undef > 0]

#        print("LENS", len(weights), len(notremoves))
        # remove the entries to get same jet shapes
        if self.remove:
            print('remove')
            notremoves=notremoves[undef > 0]
            weights=weights[notremoves > 0]
            x_glb=x_glb[notremoves > 0]
            x_db=x_db[notremoves > 0]
            alltruth=alltruth[notremoves > 0]
            
        #newnsamp=x_global.shape[0]
        newnsamp=x_glb.shape[0]
	print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')
        self.nsamples = newnsamp
        
        # fill everything
        self.w=[weights]
        self.x=[x_db]
        self.z=[x_glb, x_dbr]
        self.y=[alltruth]

        
