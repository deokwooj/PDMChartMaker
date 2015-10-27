# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 23:24:03 2015

@author: deokwooj
"""
from __future__ import division # To forace float point division
from pdm_config import *
from scipy.fftpack import fft, dct
from sklearn import cluster
from sklearn.cluster import Ward
from sklearn.cluster import KMeans
from sklearn.neighbors.kde import KernelDensity
from sklearn import manifold
from sklearn.metrics import euclidean_distances
from sklearn.decomposition import PCA
from sklearn import cluster, covariance, manifold
from scipy.stats import stats 
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx

__author__ = "Deokwoo Jung"
__credits__ = ["Deokwoo Jung"]
__version__ = "0.1.0"
__maintainer__ = "Deokwoo Jung"
__email__ = "deokwooj@gmail.com"
__status__ = "Protoype"


class RawData:
    def __init__(self):
        self.t = [] # time in 
        self.d = [] # data (unitless)


class AccSensor:
    def __init__(self):
        self.addr =None # Address
        # location,0,1,2,3,....9, fixing type,0,1,2,3,  0 for ref, 1 for bad. 
        self.label=None # a lable tuple = (loc, fix, ref)
        self.x=RawData()
        self.y=RawData()
        self.z=RawData()
        self.txyz=None
        self.xyz_mds=None
        self.xyz_avg=None
        self.xyz_std=None
        self.xyz_dct=None
        self.start_date=dt.datetime(1999,12,31)  #  The first time for sensor reading
        self.latest_date=dt.datetime(1999,12,31)  # The latest time for sensor reading.
        self.feature_vec =None # feacture vector for raw data. 
    # push a new data stream from database
    def set_data(self,ort_,t_,d_):
        #import pdb;pdb.set_trace()
        self.__dict__[ort_].t=np.array(t_)
        self.__dict__[ort_].d=np.array(d_)
    
    # Rescale the data
    def set_label(self): # set label index and return it.
        self.label=(self.loc,self.fix,self.ref) 
    
    def set_txyz(self): # set data structrue for (t,x,y,z)
        print 'check consistency of vector size for ', self.label
        assert self.x.t.shape[0] ==self.x.d.shape[0]==self.y.t.shape[0]\
        ==self.y.d.shape[0] == self.z.t.shape[0]  ==self.z.d.shape[0]
        """
        print 'check consistency of time reference for ', i
        assert tuple(self.x.t)== tuple(self.y.t)==tuple(self.z.t)
        """
        # average out time stamp. 
        # TODO: For asynced time stamp, align t and interploate xyz
        t_=np.average(np.c_[self.x.t,self.y.t,self.z.t],axis=1)
        # construct txyz ndarray. 
        self.txyz=np.c_[t_,self.x.d,self.y.d,self.z.d]
    
    def get_t(self):
        return self.txyz[:,0]
    
    def get_x(self):
        return self.txyz[:,1]
    
    def get_y(self):
        return self.txyz[:,2]
    
    def get_z(self):
        return self.txyz[:,3]
    
    def get_xyz(self):
        return self.txyz[:,1:4]
    
    def get_avg_xyz(self):
        print 'Compute AVG for xyz acc data..,'
        self.xyz_avg=self.get_xyz().mean(axis=0)
        return self.xyz_avg
    
    def get_std_xyz(self):
        print 'Compute STD for xyz acc data..,'
        self.xyz_std=self.get_xyz().std(axis=0)
        return self.xyz_std
    
    # get discrete consine tranform (dct) for xyz
    def get_dct_xyz(self):
        print 'Compute DCT for xyz acc data..,'
        self.xyz_dct= \
        np.c_[dct(self.get_x()), dct(self.get_y()), dct(self.get_z())]
        return self.xyz_dct
    
    # compute mds for xyz, and store/rself.xyz_avgeturn it.
    def get_mds_xyz(self):
        print 'Compute 2D mds for xyz acc data..,'
        print '------------------------------------'
        xyz_=self.get_xyz()[0:100,:]
        print 'compute euclidan distance... '
        dst_mat = euclidean_distances(xyz_)
        seed = np.random.RandomState(seed=3)
        mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, random_state=seed,
                           dissimilarity="precomputed", n_jobs=1)
        print  'compute mds embedding... '
        pos = mds.fit(dst_mat).embedding_
        clf = PCA(n_components=2)
        print  'compute pcs transform... '
        pos = clf.fit_transform(pos)
        self.xyz_mds=pos
        return pos


def analyzeSeries(series):
    assert isinstance(series, dict)
    assert 'x' in series 
    assert 'y' in series
    assert 'z' in series 
    assert 'name' in series

    name = series['name']

    acc = AccSensor()
    for axis in 'xyz':
        t, d = zip(*series[axis])
        acc.set_data(axis, t, d)

    print '------------------------------------------------------------------------'
    print  'constrcut txyz ndarray '
    acc.set_txyz()
    acc.get_avg_xyz()
    acc.get_std_xyz()
    acc.get_mds_xyz()
    acc.get_dct_xyz()
        
    print '------------------------------------------------------------------------'
    plt.ioff()
    # 1-D tims plot
    i = 0
    fig_ts=plt.figure('ts_'+str(i),figsize=(6*3.13,4*3.13))
    # 3-D scatter plot
    fig_3d =plt.figure('sct_'+str(i),figsize=(6*3.13,4*3.13))
    ax_3d = fig_3d.add_subplot(1,1,1, projection='3d')
    # 2-D MDS plot
    fig_mds =plt.figure('mds_'+str(i),figsize=(6*3.13,4*3.13))
    ax_mds=fig_mds.add_subplot(1,1,1)
    # 3-D DCT plot
    fig_dct =plt.figure('dct_'+str(i),figsize=(6*3.13,4*3.13))
    ax_dct=fig_dct.add_subplot(1,1,1, projection='3d')
    # DCT spectral plot
    fig_spr =plt.figure('spr_'+str(i),figsize=(6*3.13,4*3.13))
    ax_spr=fig_spr.add_subplot(1,1,1)
    
    ax_ts=fig_ts.add_subplot(2,1,1)
    t_=pmt.unix_to_dtime(acc.get_t())
    ax_ts.plot(t_,acc.get_xyz())
    ax_ts.set_title('Time Series - ' + name)
    ax_ts.tick_params(labelsize='small')

    # 2. 3-D scatter plot
    ref_3d=ax_3d.scatter(acc.get_x(), acc.get_y(), acc.get_z(), c='r')
    
    # 3. MDS plot
    ref_mds_xyz=acc.xyz_mds
    ref_mds=ax_mds.scatter(ref_mds_xyz[:, 0], ref_mds_xyz[:, 1], s=40, c='b',marker='s')
    
    # 4. 3-D DCT plot
    ref_dct_xyz=abs(acc.xyz_dct)
    ref_dct=ax_dct.scatter(ref_dct_xyz[:,0],ref_dct_xyz[:,1],ref_dct_xyz[:,2],c='b',marker='s')
    
    # 5. 1-D Spectral
    # frq compute
    dt_ = np.average(np.diff(acc.get_t()))
    Fs = 1 / dt_  # sampling rate, Fs = 500MHz = 1/2ns
    n = ref_dct_xyz.shape[0]  # length of the signal
    k = np.arange(n)
    T = n / Fs
    frq = k / T  # two sides frequency range
    ref_spr=ax_spr.plot(frq,ref_dct_xyz.sum(axis=1),'b')
    
    plt.xlim([10,90])
    plt.ylim([0,1000])

    # Add lengeds to 3D scatter plot
    ax_3d.legend((ref_3d,),(name,), scatterpoints=1, loc='lower left',fontsize=18)
    ax_3d.set_title( '3D Scatter Plot - ' + name)
    ax_3d.tick_params(labelsize='large')
    ax_3d.set_xlabel('X Acc. (t)')
    ax_3d.set_ylabel('Y Acc. (t)')
    ax_3d.set_zlabel('Z Acc. (t)')
    
    # Add lengeds to MDS scatter plot
    ax_mds.legend((ref_mds, ),(name,), scatterpoints=1, loc='lower left',fontsize=18)
    ax_mds.set_title( 'MDS Scatter Plot - ' + name)
    
    # Add lengeds to DCT plot
    ax_dct.legend((ref_dct, ),('ref', ), scatterpoints=1, loc='lower left',fontsize=18)
    ax_dct.set_title('DCT Plot - ' + name)
    ax_dct.tick_params(labelsize='large')
    ax_dct.set_xlabel('X Acc. (Hz)')
    ax_dct.set_ylabel('Y Acc. (Hz)')
    ax_dct.set_zlabel('Z Acc. (Hz)')
    
    # Add lengeds to SPR plot
    ax_spr.set_title('SPR Plot - ' + name)
    ax_spr.set_xlabel('Freq. (Hz)')
    ax_spr.set_ylabel('Power Spetral')

    # label and save for time seriese plot
    fig_t_label='ts_' + name
    fig_ts.savefig(FIG_OUT_DIR+fig_t_label+'.png', bbox_inches='tight')
    plt.close(fig_ts)
    # label and save for 3d scatter plot
    fig_3d_label='3d_' + name
    fig_3d.savefig(FIG_OUT_DIR+fig_3d_label+'.png', bbox_inches='tight')
    plt.close(fig_3d)
    # label and save for mds plot
    fig_mds_label='mds_' + name
    fig_mds.savefig(FIG_OUT_DIR+fig_mds_label+'.png', bbox_inches='tight')
    plt.close(fig_mds)
    # label and save for DCT scatter plot
    fig_dct_label='dct_' + name
    fig_dct.savefig(FIG_OUT_DIR+fig_dct_label+'.png', bbox_inches='tight')
    plt.close(fig_dct)

    # label and save for freq spectral plot
    ax_spr.tick_params(labelsize='large')
    fig_spr_label='spr_' + name
    fig_spr.savefig(FIG_OUT_DIR+fig_spr_label+'.png', bbox_inches='tight')
    plt.close(fig_spr)

    plt.ion()
    print '------------------------------------------------------------------------'


def analyzeSeriesWithName(seriesName):
    def load(filename):
        res = []
        lines = open(filename, 'r').read().split('\n')
        for l in lines:
            l = l.strip()
            if not l:
                continue

            _, timestamp, value = l.split(' ')
            timestamp = float(timestamp)
            value = float(value)
            res.append((timestamp, value))

        return res

    analyzeSeries({
        'name' : seriesName,
        'x': load('data_in/%s_x.txt' % seriesName),
        'y': load('data_in/%s_y.txt' % seriesName),
        'z': load('data_in/%s_z.txt' % seriesName)
    })


if __name__ == '__main__':
    analyzeSeriesWithName('foo')
