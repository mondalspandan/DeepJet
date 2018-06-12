from __future__ import print_function

from DeepJetCore.TrainData import TrainData, fileTimeOut
import numpy

class TrainData_deepDoubleC(TrainData):
    
    def __init__(self):
        '''
        This class is meant as a base class for the FatJet studies
        You will not need to edit it for trying out things
        '''
        TrainData.__init__(self)
        
        #define truth:
	self.treename = "deepntuplizer/tree"
        self.undefTruth=['isUndefined']
        self.truthclasses=['fj_isNonCC', 'fj_isCC', 'fj_isNonBB', 'fj_isQCD', 'fj_isH']
        self.truthclasses=['fj_isCC', 'fj_isBB', 'fj_isQCD' ]
        self.truthclasses=["label_H_bb", "label_H_cc", "label_QCD_bb",  "label_QCD_cc", "label_QCD_others"]
        #self.truthclasses=["fj_isCC", "fj_isNonCC"]
        #self.referenceclass='label_H_cc' ## used for pt reshaping
        self.referenceclass='lowest' ## used for pt reshaping
        #self.referenceclass='flatten' ## used for pt reshaping
        self.registerBranches(['fj_pt','fj_sdmass'])

        self.weightbranchX='fj_pt'
        self.weightbranchY='fj_sdmass'

        #self.weight_binX = numpy.array([
        #        300,1000,2500],dtype=float)
	self.weight_binX = numpy.array(
                #[300,400,500,600,700, 800, 900, 1000, 
		#1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]
		range(300,1000,50)+range(1000,2600,200),dtype=float)

        self.weight_binY = numpy.array(
            [40,200],
            #range(40,200,40),
            dtype=float
            )
	self.weight=True
	self.remove=False

        #this is only needed because the truth definitions are different from deepFlavour
        self.allbranchestoberead=[]
        self.registerBranches(self.undefTruth)
        self.registerBranches(self.truthclasses)
        self.registerBranches(['fj_isBB', 'fj_isNonBB', 'fj_isNonCC', 'fj_isCC', 'fj_isQCD', 'fj_isH'])
        print("Branches read:", self.allbranchestoberead)
        
    ## categories to use for training     
    def reduceTruth(self, tuple_in):
        import numpy
        self.reducedtruthclasses=['fj_isNonCC','fj_isCC']
        if tuple_in is not None:
            q = tuple_in['fj_isNonCC']
            q = q.view(numpy.ndarray)
            h = tuple_in['fj_isCC']
            h = h.view(numpy.ndarray)
            return numpy.vstack((q,h)).transpose()  
        
class TrainData_deepDoubleC_db(TrainData_deepDoubleC):
    # NOT UP TO DATE 
    
    def __init__(self):
        '''
        This is an example data format description for FatJet studies
        '''
        TrainData_deepDoubleC.__init__(self)

        self.remove = True
        self.weight = False
        #example of how to register global branches
        self.addBranches(['fj_pt',
                          'fj_eta',
                          'fj_sdmass',
                          'fj_n_sdsubjets',
                          'fj_doubleb',
                          'fj_tau21',
                          'fj_tau32',
                          'npv',
                          'npfcands',
                          'ntracks',
                          'nsv'
                      ])
        
        self.addBranches(['fj_jetNTracks',
                          'fj_nSV',
                          'fj_tau0_trackEtaRel_0',
                          'fj_tau0_trackEtaRel_1',
                          'fj_tau0_trackEtaRel_2',
                          'fj_tau1_trackEtaRel_0',
                          'fj_tau1_trackEtaRel_1',
                          'fj_tau1_trackEtaRel_2',
                          'fj_tau_flightDistance2dSig_0',
                          'fj_tau_flightDistance2dSig_1',
                          'fj_tau_vertexDeltaR_0',
                          'fj_tau_vertexEnergyRatio_0',
                          'fj_tau_vertexEnergyRatio_1',
                          'fj_tau_vertexMass_0',
                          'fj_tau_vertexMass_1',
                          'fj_trackSip2dSigAboveBottom_0',
                          'fj_trackSip2dSigAboveBottom_1',
                          'fj_trackSip2dSigAboveCharm_0',
                          'fj_trackSipdSig_0',
                          'fj_trackSipdSig_0_0',
                          'fj_trackSipdSig_0_1',
                          'fj_trackSipdSig_1',
                          'fj_trackSipdSig_1_0',
                          'fj_trackSipdSig_1_1',
                          'fj_trackSipdSig_2',
                          'fj_trackSipdSig_3',
                          'fj_z_ratio',
                          ]) 

        #branches that are used directly in the following function 'readFromRootFile'
        #this is a technical trick to speed up the conversion
        #self.registerBranches(['Cpfcan_erel','Cpfcan_eta','Cpfcan_phi',
        #                       'Npfcan_erel','Npfcan_eta','Npfcan_phi',
        #                       'nCpfcand','nNpfcand',
        #                       'jet_eta','jet_phi'])
        self.registerBranches(['sample_isQCD','fj_isH','fj_isQCD'])
        
        
    #this function describes how the branches are converted
    def readFromRootFile(self,filename,TupleMeanStd, weighter):
        
        #the first part is standard, no changes needed
        from DeepJetCore.preprocessing import MeanNormApply, MeanNormZeroPad, MeanNormZeroPadParticles, ZeroPadParticles
        import numpy
        import ROOT
        
        fileTimeOut(filename,120) #give eos 2 minutes to recover
        rfile = ROOT.TFile(filename)
        tree = rfile.Get("deepntuplizer/tree")
        self.nsamples=tree.GetEntries()
    
        x_glb  = ZeroPadParticles(filename,TupleMeanStd,
                                          self.branches[0],
                                          self.branchcutoffs[0],self.nsamples)

        x_db  = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                         self.branches[1],
                                         self.branchcutoffs[1],self.nsamples)
        
        # now, some jets are removed to avoid pt and eta biases
        
        Tuple = self.readTreeFromRootToTuple(filename)
        #if self.remove:
            # jets are removed until the shapes in eta and pt are the same as
            # the truth class 'fj_isNonBB'
        notremoves=weighter.createNotRemoveIndices(Tuple)
        if self.weight:
            weights=weighter.getJetWeights(Tuple)
        elif self.remove:
            weights=notremoves
        else:
            print('neither remove nor weight')
            weights=numpy.empty(self.nsamples)
            weights.fill(1.)
            
            
        # create all collections:
        #truthtuple =  Tuple[self.truthclasses]
        alltruth=self.reduceTruth(Tuple)
        undef=numpy.sum(alltruth,axis=1)
        #weights=weights[undef > 0]
        #x_glb=x_glb[undef > 0]
        #x_db=x_db[undef > 0]
        #alltruth=alltruth[undef > 0]
        notremoves=notremoves[undef > 0]

        undef=Tuple['fj_isNonCC'] * Tuple['sample_isQCD'] * Tuple['fj_isQCD'] + Tuple['fj_isCC'] * Tuple['fj_isH']

        # remove the entries to get same jet shapes
        if self.remove:
            print('remove')
            weights=weights[notremoves > 0]
            x_glb=x_glb[notremoves > 0]
            x_db=x_db[notremoves > 0]
            alltruth=alltruth[notremoves > 0]
            
        newnsamp=x_glb.shape[0]
        print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')
        self.nsamples = newnsamp
        
        # fill everything
        self.w=[weights]
        self.x=[x_db]
        self.z=[x_glb]
        self.y=[alltruth]        

    def reduceTruth(self, tuple_in):
        import numpy
        self.reducedtruthclasses=['QCD','Hcc']
        if tuple_in is not None:
            q = tuple_in['fj_isQCD']
            q = q.view(numpy.ndarray)
            h = tuple_in['fj_isCC'] * tuple_in['fj_isH']
            h = h.view(numpy.ndarray)

            return numpy.vstack((q,h)).transpose()


#######################################
            
class TrainData_deepDoubleC_db_pf_cpf_sv(TrainData_deepDoubleC):
   # NOT UP TO DATE
    def __init__(self):
        '''
        This is an example data format description for FatJet studies
        '''
        TrainData_deepDoubleC.__init__(self)
        
        #example of how to register global branches
        self.addBranches(['fj_pt',
                          'fj_eta',
                          'fj_sdmass',
                          'fj_n_sdsubjets',
                          'fj_doubleb',
                          'fj_tau21',
                          'fj_tau32',
                          'npv',
                          'npfcands',
                          'ntracks',
                          'nsv'
                      ])
        
        self.addBranches(['fj_jetNTracks',
                          'fj_nSV',
                          'fj_tau0_trackEtaRel_0',
                          'fj_tau0_trackEtaRel_1',
                          'fj_tau0_trackEtaRel_2',
                          'fj_tau1_trackEtaRel_0',
                          'fj_tau1_trackEtaRel_1',
                          'fj_tau1_trackEtaRel_2',
                          'fj_tau_flightDistance2dSig_0',
                          'fj_tau_flightDistance2dSig_1',
                          'fj_tau_vertexDeltaR_0',
                          'fj_tau_vertexEnergyRatio_0',
                          'fj_tau_vertexEnergyRatio_1',
                          'fj_tau_vertexMass_0',
                          'fj_tau_vertexMass_1',
                          'fj_trackSip2dSigAboveBottom_0',
                          'fj_trackSip2dSigAboveBottom_1',
                          'fj_trackSip2dSigAboveCharm_0',
                          'fj_trackSipdSig_0',
                          'fj_trackSipdSig_0_0',
                          'fj_trackSipdSig_0_1',
                          'fj_trackSipdSig_1',
                          'fj_trackSipdSig_1_0',
                          'fj_trackSipdSig_1_1',
                          'fj_trackSipdSig_2',
                          'fj_trackSipdSig_3',
                          'fj_z_ratio'
                          ])
        
        #example of pf candidate branches
        """self.addBranches(['pfcand_ptrel',
                          #'pfcand_erel',
                          'pfcand_phirel',
                          'pfcand_etarel',
                          'pfcand_deltaR',
                          'pfcand_puppiw',
                          'pfcand_drminsv',
                          'pfcand_drsubjet1',
                          #'pfcand_drsubjet2',
                          'pfcand_hcalFrac'
                         ],
                         100) 
	"""
        self.addBranches(['track_ptrel',     
                          'track_erel',     
                          'track_phirel',     
                          'track_etarel',     
                          'track_deltaR',
                          'track_drminsv',     
                          'track_drsubjet1',     
                          'track_drsubjet2',
                          'track_dz',     
                          'track_dzsig',     
                          'track_dxy',     
                          'track_dxysig',     
                          'track_normchi2',     
                          'track_quality',     
                          'track_dptdpt',     
                          'track_detadeta',     
                          'track_dphidphi',     
                          'track_dxydxy',     
                          'track_dzdz',     
                          'track_dxydz',     
                          'track_dphidxy',     
                          'track_dlambdadz',     
                          'trackBTag_EtaRel',     
                          'trackBTag_PtRatio',     
                          'trackBTag_PParRatio',     
                          'trackBTag_Sip2dVal',     
                          'trackBTag_Sip2dSig',     
                          'trackBTag_Sip3dVal',     
                          'trackBTag_Sip3dSig',     
                          'trackBTag_JetDistVal'
                         ],
                         60) 
        
        self.addBranches(['sv_ptrel',
                          'sv_erel',
                          'sv_phirel',
                          'sv_etarel',
                          'sv_deltaR',
                          'sv_pt',
                          'sv_mass',
                          'sv_ntracks',
                          'sv_normchi2',
                          'sv_dxy',
                          'sv_dxysig',
                          'sv_d3d',
                          'sv_d3dsig',
                          'sv_costhetasvpv'
                         ],
                         5)

        #branches that are used directly in the following function 'readFromRootFile'
        #this is a technical trick to speed up the conversion
        #self.registerBranches(['Cpfcan_erel','Cpfcan_eta','Cpfcan_phi',
        #                       'Npfcan_erel','Npfcan_eta','Npfcan_phi',
        #                       'nCpfcand','nNpfcand',
        #                       'jet_eta','jet_phi'])
        self.registerBranches(['sample_isQCD','fj_isH','fj_isQCD'])
        
        
    #this function describes how the branches are converted
    def readFromRootFile(self,filename,TupleMeanStd, weighter):
        
        #the first part is standard, no changes needed
        from DeepJetCore.preprocessing import MeanNormApply, MeanNormZeroPad, MeanNormZeroPadParticles, ZeroPadParticles
        import numpy
        import ROOT
        
        fileTimeOut(filename,120) #give eos 2 minutes to recover
        rfile = ROOT.TFile(filename)
        tree = rfile.Get("deepntuplizer/tree")
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

        x_db  = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                         self.branches[1],
                                         self.branchcutoffs[1],self.nsamples)

        #x_pf  = MeanNormZeroPadParticles(filename,TupleMeanStd,
        #                                 self.branches[2],
        #                                 self.branchcutoffs[2],self.nsamples)
        
        x_cpf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                         self.branches[2],
                                         self.branchcutoffs[2],self.nsamples)
        
        x_sv = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                        self.branches[3],
                                        self.branchcutoffs[3],self.nsamples)
        
        # now, some jets are removed to avoid pt and eta biases
        
        Tuple = self.readTreeFromRootToTuple(filename)
        #if self.remove:
            # jets are removed until the shapes in eta and pt are the same as
            # the truth class 'fj_isNonBB'
	#weighter.setBinningAndClasses([self.weight_binX,self.weight_binY])
        notremoves=weighter.createNotRemoveIndices(Tuple)
            #undef=Tuple[self.undefTruth]
            #notremoves-=undef
        if self.weight:
            weights=weighter.getJetWeights(Tuple)
        elif self.remove:
            weights=notremoves
        else:
            print('neither remove nor weight')
            weights=numpy.empty(self.nsamples)
            weights.fill(1.)
            
            
        # create all collections:
        #truthtuple =  Tuple[self.truthclasses]
        alltruth=self.reduceTruth(Tuple)
        undef=numpy.sum(alltruth,axis=1)
        weights=weights[undef > 0]
        x_glb=x_glb[undef > 0]
        x_db=x_db[undef > 0]
        x_sv=x_sv[undef > 0]
        #x_pf=x_pf[undef > 0]
        x_cpf=x_cpf[undef > 0]
        alltruth=alltruth[undef > 0]

        print("LENS", len(weights), len(notremoves))
        # remove the entries to get same jet shapes
        if self.remove:
            print('remove')
            notremoves=notremoves[undef > 0]
            weights=weights[notremoves > 0]
            x_glb=x_glb[notremoves > 0]
            x_db=x_db[notremoves > 0]
            x_sv=x_sv[notremoves > 0]
            #x_pf=x_pf[notremoves > 0]
            x_cpf=x_cpf[notremoves > 0]
            alltruth=alltruth[notremoves > 0]
            
        #newnsamp=x_global.shape[0]
        newnsamp=x_glb.shape[0]
        print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')
        self.nsamples = newnsamp
        
        # fill everything
        self.w=[weights]
        self.x=[x_db,x_cpf,x_sv]
        self.z=[x_glb]
        self.y=[alltruth]

    def reduceTruth(self, tuple_in):
        import numpy
        self.reducedtruthclasses=['QCD','Hcc']
        if tuple_in is not None:
            q = tuple_in['fj_isQCD']
            q = q.view(numpy.ndarray)
            h = tuple_in['fj_isCC'] * tuple_in['fj_isH']
            h = h.view(numpy.ndarray)

            return numpy.vstack((q,h)).transpose()


#######################################
        
class TrainData_deepDoubleC_db_cpf_sv_reduced(TrainData_deepDoubleC):
    
    def __init__(self):
        '''
        This is an example data format description for FatJet studies
        '''
        TrainData_deepDoubleC.__init__(self)
        
        #example of how to register global branches
        self.addBranches(['fj_pt',
                          'fj_eta',
                          'fj_sdmass',
                          'fj_n_sdsubjets',
                          'fj_doubleb',
                          'fj_tau21',
                          'fj_tau32',
			  # Doesn't work, bad file type - int not, float
                          #'fj_isH',
                          #'fj_isQCD',
                          #'fj_isCC',
                          #'fj_isBB',
                          'npv',
                          'npfcands',
                          'ntracks',
                          'nsv'
                      ])
        
        self.addBranches(['fj_jetNTracks',
                          'fj_nSV',
                          'fj_tau0_trackEtaRel_0',
                          'fj_tau0_trackEtaRel_1',
                          'fj_tau0_trackEtaRel_2',
                          'fj_tau1_trackEtaRel_0',
                          'fj_tau1_trackEtaRel_1',
                          'fj_tau1_trackEtaRel_2',
                          'fj_tau_flightDistance2dSig_0',
                          'fj_tau_flightDistance2dSig_1',
                          'fj_tau_vertexDeltaR_0',
                          'fj_tau_vertexEnergyRatio_0',
                          'fj_tau_vertexEnergyRatio_1',
                          'fj_tau_vertexMass_0',
                          'fj_tau_vertexMass_1',
                          'fj_trackSip2dSigAboveBottom_0',
                          'fj_trackSip2dSigAboveBottom_1',
                          'fj_trackSip2dSigAboveCharm_0',
                          'fj_trackSipdSig_0',
                          'fj_trackSipdSig_0_0',
                          'fj_trackSipdSig_0_1',
                          'fj_trackSipdSig_1',
                          'fj_trackSipdSig_1_0',
                          'fj_trackSipdSig_1_1',
                          'fj_trackSipdSig_2',
                          'fj_trackSipdSig_3',
                          'fj_z_ratio'
                          ])
        
        self.addBranches(['trackBTag_EtaRel',     
                          'trackBTag_PtRatio',     
                          'trackBTag_PParRatio',     
                          'trackBTag_Sip2dVal',     
                          'trackBTag_Sip2dSig',     
                          'trackBTag_Sip3dVal',     
                          'trackBTag_Sip3dSig',     
                          'trackBTag_JetDistVal'
                         ],
                         60) 
        
        self.addBranches(['sv_d3d',
                          'sv_d3dsig',
                         ],
                         5)

        #branches that are used directly in the following function 'readFromRootFile'
        #this is a technical trick to speed up the conversion
        #self.registerBranches(['Cpfcan_erel','Cpfcan_eta','Cpfcan_phi',
        #                       'Npfcan_erel','Npfcan_eta','Npfcan_phi',
        #                       'nCpfcand','nNpfcand',
        #                       'jet_eta','jet_phi'])
        self.registerBranches(['sample_isQCD','fj_isH','fj_isQCD'])
        
        
    #this function describes how the branches are converted
    def readFromRootFile(self,filename,TupleMeanStd, weighter):
        
        #the first part is standard, no changes needed
        from DeepJetCore.preprocessing import MeanNormApply, MeanNormZeroPad, MeanNormZeroPadParticles, ZeroPadParticles
        import numpy
        import ROOT
        
        fileTimeOut(filename,120) #give eos 2 minutes to recover
        rfile = ROOT.TFile(filename)
        tree = rfile.Get("deepntuplizer/tree")
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

        x_db  = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                         self.branches[1],
                                         self.branchcutoffs[1],self.nsamples)
       
        x_db_raw  = ZeroPadParticles(filename,TupleMeanStd,
                                         self.branches[1],
                                         self.branchcutoffs[1],self.nsamples)
 
        x_cpf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                         self.branches[2],
                                         self.branchcutoffs[2],self.nsamples)
        
        x_sv = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                        self.branches[3],
                                        self.branchcutoffs[3],self.nsamples)
        
        # now, some jets are removed to avoid pt and eta biases
        
        Tuple = self.readTreeFromRootToTuple(filename)
        if self.remove or self.weight:
            # jets are removed until the shapes in eta and pt are the same as
            # the truth class 'fj_isNonBB'
            notremoves=weighter.createNotRemoveIndices(Tuple)
            #undef=Tuple[self.undefTruth]
        #notremoves-=undef
        
        if self.weight:
            weights=weighter.getJetWeights(Tuple)
        elif self.remove:
            weights=notremoves #weighter.createNotRemoveIndices(Tuple)
        else:
            print('neither remove nor weight')
            weights=numpy.empty(self.nsamples)
            weights.fill(1.)
	    

	truthtuple =  Tuple[self.truthclasses]
        alltruth=self.reduceTruth(Tuple)
	undef=numpy.sum(alltruth,axis=1)
            
        if self.remove:
	    print('remove')
	    weights=weights[undef > 0]
            x_glb=x_glb[undef > 0]
	    x_db=x_db[undef > 0]
	    x_db_raw=x_db_raw[undef > 0]
	    x_sv=x_sv[undef > 0]
	    x_cpf=x_cpf[undef > 0]
	    alltruth=alltruth[undef > 0]

            # remove the entries to get same jet shapes
            notremoves=notremoves[undef > 0]
            weights=weights[notremoves > 0]
            x_glb=x_glb[notremoves > 0]
            x_db=x_db[notremoves > 0]
            x_db_raw=x_db_raw[notremoves > 0]
            x_sv=x_sv[notremoves > 0]
            x_cpf=x_cpf[notremoves > 0]
            alltruth=alltruth[notremoves > 0]
            
        #newnsamp=x_global.shape[0]
        newnsamp=x_glb.shape[0]
        print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')
        self.nsamples = newnsamp
        
	#print("Subsample composition:")
        #for lab in ['fJ_isQCD', 'fj_isH', 'fj_isCC', 'fj_isBB']:
	#	print(numpy.sum((Tuple[lab].view(numpy.ndarray))), lab)	
        #for lab, stat in zip(self.reducedtruthclasses, stats):
	#	print(lab, ': {}%'.format(stat))

        # fill everything
        self.w=[weights]
        self.x=[x_db,x_cpf,x_sv]
        self.z=[x_glb, x_db_raw]
        self.y=[alltruth]

    def reduceTruth(self, tuple_in):
        import numpy
        self.reducedtruthclasses=['QCD','Hcc']
        if tuple_in is not None:
            q = tuple_in['fj_isQCD']
            q = q.view(numpy.ndarray)
            #h = tuple_in['fj_isCC'] * tuple_in['fj_isH']
            h = tuple_in['fj_isH']
            h = h.view(numpy.ndarray)

            return numpy.vstack((q,h)).transpose()


class TrainData_deepDoubleC_db_cpf_sv_reduced_incglu(TrainData_deepDoubleC_db_cpf_sv_reduced):
    ## categories to use for training
    def reduceTruth(self, tuple_in):
        import numpy
        self.reducedtruthclasses=['Light','cc']
        if tuple_in is not None:
            q = tuple_in['fj_isNonCC'] * tuple_in['fj_isQCD'] * tuple_in['fj_isNonBB'] 
            q = q.view(numpy.ndarray)
            h = tuple_in['fj_isCC'] 
            h = h.view(numpy.ndarray)
            return numpy.vstack((q,h)).transpose()


#######################################
#        double-c vs double-b
#######################################
class TrainData_deepDoubleCvB_db(TrainData_deepDoubleC_db):
    ## categories to use for training
    def reduceTruth(self, tuple_in):
        import numpy
        self.reducedtruthclasses=['Hbb','Hcc']
        if tuple_in is not None:
            q = tuple_in['fj_isBB'] * tuple_in['fj_isH']
            q = q.view(numpy.ndarray)
            h = tuple_in['fj_isCC'] * tuple_in['fj_isH']
            h = h.view(numpy.ndarray)
            return numpy.vstack((q,h)).transpose()

######################################
class TrainData_deepDoubleCvB_db_pf_cpf_sv(TrainData_deepDoubleC_db_pf_cpf_sv):
    ## categories to use for training
    def reduceTruth(self, tuple_in):
        import numpy
        self.reducedtruthclasses=['Hbb','Hcc']
        if tuple_in is not None:
            q = tuple_in['fj_isBB'] * tuple_in['fj_isH']
            q = q.view(numpy.ndarray)
            h = tuple_in['fj_isCC'] * tuple_in['fj_isH']
            h = h.view(numpy.ndarray)
            return numpy.vstack((q,h)).transpose()


#####################################
class TrainData_deepDoubleCvB_db_cpf_sv_reduced(TrainData_deepDoubleC_db_cpf_sv_reduced):
    ## categories to use for training
    def reduceTruth(self, tuple_in):
        import numpy
        self.reducedtruthclasses=['Hbb','Hcc']
        if tuple_in is not None:
            q = tuple_in['fj_isBB'] * tuple_in['fj_isH']
            q = q.view(numpy.ndarray)
            h = tuple_in['fj_isCC'] * tuple_in['fj_isH']
            h = h.view(numpy.ndarray)
            return numpy.vstack((q,h)).transpose()

class TrainData_deepDoubleCvB_db_cpf_sv_reduced_incglu(TrainData_deepDoubleC_db_cpf_sv_reduced):
    ## categories to use for training
    def reduceTruth(self, tuple_in):
        import numpy
        self.reducedtruthclasses=['bb','cc']
        if tuple_in is not None:
            q = tuple_in['fj_isBB'] 
            q = q.view(numpy.ndarray)
            h = tuple_in['fj_isCC'] 
            h = h.view(numpy.ndarray)
            return numpy.vstack((q,h)).transpose()

#######################################
#        double-b vs QCD
#####################################
class TrainData_deepDoubleBvQCD_db_cpf_sv_reduced(TrainData_deepDoubleC_db_cpf_sv_reduced):
    ## categories to use for training
    def reduceTruth(self, tuple_in):
        import numpy
        self.reducedtruthclasses=['QCD','Hbb']
        if tuple_in is not None:
            q = tuple_in['fj_isQCD'] * tuple_in['fj_isNonBB']
            q = q.view(numpy.ndarray)
            h = tuple_in['fj_isBB'] * tuple_in['fj_isH']
            h = h.view(numpy.ndarray)

            return numpy.vstack((q,h)).transpose()
 
######################################
#            Multiclass              #
######################################
class TrainData_deepDoubleC_db_multi(TrainData_deepDoubleC_db):    
    ## categories to use for training 
    def reduceTruth(self, tuple_in):
        import numpy
        self.reducedtruthclasses=['QCD','Hcc', 'Hbb']
        if tuple_in is not None:
            q = tuple_in["fj_isQCD"]
            q = q.view(numpy.ndarray)
            h1 = tuple_in['fj_isCC'] * tuple_in['fj_isH']
            h2 = tuple_in['fj_isBB'] * tuple_in['fj_isH'] 
            h1 = h1.view(numpy.ndarray)
            h2 = h2.view(numpy.ndarray)
            return numpy.vstack((q,h1, h2)).transpose() 

#######################################
class TrainData_deepDoubleC_db_pf_cpf_sv_multi(TrainData_deepDoubleC_db_pf_cpf_sv):        
    ## categories to use for training     
    def reduceTruth(self, tuple_in):
        import numpy
        self.reducedtruthclasses=['QCD','Hcc', 'Hbb']
        if tuple_in is not None:
            q = tuple_in["fj_isQCD"]
            q = q.view(numpy.ndarray)
            h1 = tuple_in['fj_isCC'] * tuple_in['fj_isH']
            h2 = tuple_in['fj_isBB'] * tuple_in['fj_isH'] 
            h1 = h1.view(numpy.ndarray)
            h2 = h2.view(numpy.ndarray)
            return numpy.vstack((q,h1, h2)).transpose()  

#######################################
class TrainData_deepDoubleC_db_cpf_sv_reduced_multi(TrainData_deepDoubleC_db_cpf_sv_reduced):        
    ## categories to use for training     
    def reduceTruth(self, tuple_in):
        import numpy
        self.reducedtruthclasses=['QCD','Hcc', 'Hbb']
        if tuple_in is not None:
            q = tuple_in["fj_isQCD"]
            q = q.view(numpy.ndarray)
            h1 = tuple_in['fj_isCC'] * tuple_in['fj_isH']
            h2 = tuple_in['fj_isBB'] * tuple_in['fj_isH'] 
            h1 = h1.view(numpy.ndarray)
            h2 = h2.view(numpy.ndarray)
            return numpy.vstack((q,h1, h2)).transpose()  

#######################################
class TrainData_deepDoubleC_db_cpf_sv_reduced_multiglue(TrainData_deepDoubleC_db_cpf_sv_reduced):
    ## categories to use for training     
    def reduceTruth(self, tuple_in):
        import numpy
        self.reducedtruthclasses=['Light','Hcc', 'Hbb', 'gcc', 'gbb']
        if tuple_in is not None:
            q = tuple_in["fj_isQCD"] * tuple_in["fj_isNonCC"] * tuple_in["fj_isNonBB"]
            q = q.view(numpy.ndarray)
            h1 = tuple_in['fj_isCC'] * tuple_in['fj_isH']
            h2 = tuple_in['fj_isBB'] * tuple_in['fj_isH']
            q1 = tuple_in['fj_isCC'] * tuple_in['fj_isQCD']
            q2 = tuple_in['fj_isBB'] * tuple_in['fj_isQCD']
            h1 = h1.view(numpy.ndarray)
            h2 = h2.view(numpy.ndarray)
            q1 = q1.view(numpy.ndarray)
            q2 = q2.view(numpy.ndarray)
	    #stats = numpy.around(numpy.array([numpy.sum(q),numpy.sum(h1),numpy.sum(h2),numpy.sum(q1), numpy.sum(q2)])/float(len(h1))*100)
	    #print('Final stats - Light:{}% , Hcc:{}% , Hbb:{}% , gcc:{}% , gbb:{}%'.format(*stats), "Total evts kept: {}".format(len(q)))
	    #print("Sample composition:")
            #for lab, stat in zip(self.reducedtruthclasses, stats):
	    #	print(lab, ': {}%'.format(stat))
	    return numpy.vstack((q,h1, h2, q1, q2)).transpose()

class TrainData_deepDoubleC_flatten(TrainData_deepDoubleC_db_cpf_sv_reduced):
    def __init__(self):
        TrainData_deepDoubleC_db_cpf_sv_reduced.__init__(self)
	self.weight=True
	self.remove=False
	self.referenceclass='flatten'
	self.truthclasses=["fj_isH", "fj_isQCD"]

class TrainData_deepDoubleC_lowest(TrainData_deepDoubleC_db_cpf_sv_reduced):
    def __init__(self):
        TrainData_deepDoubleC_db_cpf_sv_reduced.__init__(self)
	self.weight=True
	self.remove=False
	self.referenceclass='lowest'
	self.truthclasses=["fj_isH", "fj_isQCD"]

class TrainData_deepDoubleC_Hcc(TrainData_deepDoubleC_db_cpf_sv_reduced):
    def __init__(self):
        TrainData_deepDoubleC_db_cpf_sv_reduced.__init__(self)
	self.weight=True
	self.remove=False
    	self.referenceclass='fj_isH'
	self.truthclasses=["fj_isH", "fj_isQCD"]

class TrainData_deepDoubleC_bkg(TrainData_deepDoubleC_db_cpf_sv_reduced):
    def __init__(self):
        TrainData_deepDoubleC_db_cpf_sv_reduced.__init__(self)
	self.weight=True
	self.remove=False
        self.referenceclass='fj_isQCD'
	self.truthclasses=["fj_isH", "fj_isQCD"]

class TrainData_deepDoubleCvB_flatten(TrainData_deepDoubleCvB_db_cpf_sv_reduced):
    def __init__(self):
        TrainData_deepDoubleC_db_cpf_sv_reduced.__init__(self)
        self.referenceclass='flatten'
	self.truthclasses=["fj_isCC", "fj_isBB"]

class TrainData_deepDoubleCvB_lowest(TrainData_deepDoubleCvB_db_cpf_sv_reduced):
    def __init__(self):
        TrainData_deepDoubleC_db_cpf_sv_reduced.__init__(self)
        self.referenceclass='lowest'
	self.truthclasses=["fj_isCC", "fj_isBB"]

class TrainData_deepDoubleCvB_Hcc(TrainData_deepDoubleCvB_db_cpf_sv_reduced):
    def __init__(self):
        TrainData_deepDoubleC_db_cpf_sv_reduced.__init__(self)
        self.referenceclass='fj_isCC'
	self.truthclasses=["fj_isCC", "fj_isBB"]
       
class TrainData_deepDoubleCvB_bkg(TrainData_deepDoubleCvB_db_cpf_sv_reduced):
    def __init__(self):
        TrainData_deepDoubleC_db_cpf_sv_reduced.__init__(self)
        self.referenceclass='fj_isBB'
        self.truthclasses=["fj_isCC", "fj_isBB"]
 
class TrainData_deepDoubleC_multiflatten(TrainData_deepDoubleC_db_cpf_sv_reduced_multiglue):
    def __init__(self):
        TrainData_deepDoubleC_db_cpf_sv_reduced.__init__(self)
        self.referenceclass='flatten'
        self.truthclasses=["label_H_bb", "label_H_cc", "label_QCD_bb",  "label_QCD_cc", "label_QCD_others"]

class TrainData_deepDoubleC_multilowest(TrainData_deepDoubleC_db_cpf_sv_reduced_multiglue):
    def __init__(self):
        TrainData_deepDoubleC_db_cpf_sv_reduced.__init__(self)
        self.referenceclass='lowest'
        self.truthclasses=["label_H_bb", "label_H_cc", "label_QCD_bb",  "label_QCD_cc", "label_QCD_others"]

class TrainData_deepDoubleC_multiHcc(TrainData_deepDoubleC_db_cpf_sv_reduced_multiglue):
    def __init__(self):
        TrainData_deepDoubleC_db_cpf_sv_reduced.__init__(self)
        self.referenceclass='label_H_cc'
        self.truthclasses=["label_H_bb", "label_H_cc", "label_QCD_bb",  "label_QCD_cc", "label_QCD_others"]

class TrainData_deepDoubleC_multibkg(TrainData_deepDoubleC_db_cpf_sv_reduced_multiglue):
    def __init__(self):
        TrainData_deepDoubleC_db_cpf_sv_reduced.__init__(self)
        self.referenceclass='label_QCD_cc'
        self.truthclasses=["label_H_bb", "label_H_cc", "label_QCD_bb",  "label_QCD_cc", "label_QCD_others"]
